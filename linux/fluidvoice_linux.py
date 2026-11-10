#!/usr/bin/env -S uv run --script --python /usr/bin/python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "onnx-asr[hub]",
#     "onnxruntime-gpu",
#     "numpy",
#     "pynput",
#     "nvidia-cudnn-cu13",
#     "nvidia-cublas",
#     "nvidia-cuda-runtime",
#     "nvidia-cufft",
#     "nvidia-curand",
#     "nvidia-cuda-nvrtc",
# ]
# ///
"""Minimal Linux (X11) dictation: hold AltGr, speak, release to paste.

Runs the same Parakeet TDT v3 model family as the macOS app, via ONNX Runtime on CUDA.
"""
from __future__ import annotations

import atexit
import glob
import json
import os
import re
import signal
import site
import subprocess
import sys
import threading
import time

import numpy as np

# PyGObject is a compiled system package (python3-gi) that uv's isolated environment
# can't see; the script is pinned to the system interpreter so its dist-packages match.
# Appended last so pip-installed packages keep precedence.
sys.path.append("/usr/lib/python3/dist-packages")


def _preload_cuda_libs() -> None:
    # The nvidia-* pip wheels ship the CUDA/cuDNN libraries outside the loader's
    # search path; onnxruntime-gpu only finds them if they are already mapped.
    import ctypes

    roots = site.getsitepackages() + [site.getusersitepackages()]
    for root in roots:
        libs = sorted(glob.glob(os.path.join(root, "nvidia", "*", "lib", "lib*.so*")))
        for _ in range(2):  # second pass resolves inter-library dependencies
            for lib in libs:
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


_preload_cuda_libs()

import gi  # noqa: E402

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import GLib, Gdk, Gtk  # noqa: E402
from pynput import keyboard  # noqa: E402

MODEL = os.environ.get("FLUIDVOICE_MODEL", "nemo-parakeet-tdt-0.6b-v3")
MODEL_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
MODEL_DIR = os.path.expanduser("~/.cache/fluidvoice/parakeet-tdt-0.6b-v3-onnx")
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600  # 100 ms
MIN_SECONDS = 0.3
TERMINAL_CLASSES = {
    "gnome-terminal-server", "kitty", "alacritty", "xterm", "konsole",
    "terminator", "tilix", "wezterm", "foot", "ghostty", "xfce4-terminal",
}


ISO_LEVEL3_SHIFT = 0xFE03  # X11 keysym for AltGr; pynput does not map it to Key.alt_gr


def is_altgr(key) -> bool:
    return key == keyboard.Key.alt_gr or getattr(key, "vk", None) == ISO_LEVEL3_SHIFT


class BluetoothMic:
    """Lets a Bluetooth headset's microphone be used for dictation.

    Bluetooth headphones in A2DP mode (high-quality playback) expose no microphone;
    the mic only exists in the headset (HFP/HSP) profile, which degrades playback.
    The profile is switched to headset while dictating and restored a few seconds
    after the last dictation, so back-to-back dictations don't make playback flap.
    Set FLUIDVOICE_BT_SWITCH=0 to disable (e.g. when another mic is preferred).
    """

    RESTORE_DELAY_S = 8.0
    READY_TIMEOUT_S = 4.0

    def __init__(self) -> None:
        self.enabled = os.environ.get("FLUIDVOICE_BT_SWITCH", "1") != "0"
        self._lock = threading.Lock()
        self._saved: tuple[str, int] | None = None  # (device id, original profile index)
        self._timer: threading.Timer | None = None

    @staticmethod
    def _device_of(target: str) -> tuple[str | None, bool]:
        info = _run("wpctl", "inspect", target)
        dev = re.search(r'device\.id = "(\d+)"', info)
        return (dev.group(1) if dev else None), 'device.api = "bluez5"' in info

    @staticmethod
    def _profiles(dev: str) -> tuple[int, str, list[dict]]:
        params = json.loads(_run("pw-dump", dev))[0]["info"]["params"]
        active = params["Profile"][0]
        return active["index"], active["name"], params["EnumProfile"]

    @staticmethod
    def _set_profile(dev: str, index: int) -> None:
        _run("pw-cli", "s", dev, "Profile", f"{{ index: {index}, save: false }}")

    def acquire(self) -> bool:
        """Make the headset mic the default source; returns False if it never appeared."""
        if not self.enabled:
            return True
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            dev, is_bt = self._device_of("@DEFAULT_AUDIO_SINK@")
            if not (dev and is_bt):
                return True
            index, name, enum = self._profiles(dev)
            if not name.startswith("a2dp"):
                return True  # already in a profile that has a mic
            headset = [p for p in enum if p["name"].startswith("headset-head-unit")]
            headset.sort(key=lambda p: "msbc" not in p["name"])  # wideband first
            if not headset:
                return True
            if self._saved is None:
                self._saved = (dev, index)
            self._set_profile(dev, headset[0]["index"])
            deadline = time.monotonic() + self.READY_TIMEOUT_S
            while time.monotonic() < deadline:
                if self._device_of("@DEFAULT_AUDIO_SOURCE@")[0] == dev:
                    return True
                time.sleep(0.1)
            return False

    def release_later(self) -> None:
        with self._lock:
            if self._saved is None:
                return
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.RESTORE_DELAY_S, self.restore)
            self._timer.daemon = True
            self._timer.start()

    def restore(self) -> None:
        with self._lock:
            if self._saved:
                self._set_profile(*self._saved)
                self._saved = None
            self._timer = None


class Recorder:
    """Streams 16 kHz mono float32 from PipeWire via pw-record."""

    def __init__(self, on_level) -> None:
        self._on_level = on_level
        self._proc: subprocess.Popen | None = None
        self._chunks: list[np.ndarray] = []
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._chunks = []
        self._proc = subprocess.Popen(
            ["pw-record", "--rate", str(SAMPLE_RATE), "--channels", "1", "--format", "f32", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(target=self._pump, args=(self._proc,), daemon=True)
        self._thread.start()

    def _pump(self, proc: subprocess.Popen) -> None:
        nbytes = CHUNK_SAMPLES * 4
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf:
                return
            chunk = np.frombuffer(buf[: len(buf) // 4 * 4], dtype=np.float32)
            self._chunks.append(chunk)
            self._on_level(float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0)

    def snapshot(self) -> np.ndarray:
        chunks = list(self._chunks)  # the reader thread keeps appending
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def stop(self) -> np.ndarray:
        proc, self._proc = self._proc, None
        if proc:
            proc.terminate()
            if self._thread:
                self._thread.join(timeout=1)
            proc.wait()
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)


class Overlay(Gtk.Window):
    """Always-on-top pill at the bottom centre showing the live transcript; never takes focus.

    Built from stock widgets and CSS: custom cairo drawing needs python3-gi-cairo,
    which is not installed everywhere.
    """

    WIDTH = 580
    MAX_CHARS = 110  # what fits on the two visible lines
    CSS = b"""
    .pill { background-color: rgba(20, 20, 26, 0.94); border-radius: 22px; padding: 0 22px; }
    .pill label { color: #f2f2f5; font-size: 15px; }
    .pill label.dim { color: #8d8d99; }
    .dot { background-color: #ff5959; border-radius: 11px; }
    .dot.busy { background-color: #6cb4ff; }
    """

    def __init__(self) -> None:
        super().__init__(type=Gtk.WindowType.POPUP)
        self.set_decorated(False)
        self.set_accept_focus(False)
        self.set_keep_above(True)
        self.set_skip_taskbar_hint(True)
        self.set_app_paintable(True)
        self.set_default_size(self.WIDTH, 76)
        visual = self.get_screen().get_rgba_visual()
        if visual:
            self.set_visual(visual)

        provider = Gtk.CssProvider()
        provider.load_from_data(self.CSS)
        Gtk.StyleContext.add_provider_for_screen(
            self.get_screen(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # The dot swells with the mic level.
        self._dot = Gtk.Box(halign=Gtk.Align.CENTER, valign=Gtk.Align.CENTER)
        self._dot.get_style_context().add_class("dot")
        self._dot.set_size_request(10, 10)
        dot_slot = Gtk.Box()
        dot_slot.set_size_request(22, 22)
        dot_slot.set_center_widget(self._dot)

        self._label = Gtk.Label(xalign=0, yalign=0.5)
        self._label.set_line_wrap(True)
        self._label.set_lines(2)
        self._label.set_ellipsize(3)  # Pango.EllipsizeMode.END
        self._label.set_size_request(self.WIDTH - 110, 44)

        row = Gtk.Box(spacing=16, valign=Gtk.Align.CENTER)
        row.pack_start(dot_slot, False, False, 0)
        row.pack_start(self._label, True, True, 0)
        row.get_style_context().add_class("pill")
        self.add(row)

    def _place(self) -> None:
        display = Gdk.Display.get_default()
        _, x, y = display.get_default_seat().get_pointer().get_position()
        geo = display.get_monitor_at_point(x, y).get_geometry()
        w, h = self.get_size()
        self.move(geo.x + (geo.width - w) // 2, geo.y + geo.height - h - 80)

    def show_state(self, status: str, busy: bool = False) -> None:
        """Show a placeholder message (dimmed); the dot turns blue while busy."""
        self._set_label(status, dim=True)
        ctx = self._dot.get_style_context()
        (ctx.add_class if busy else ctx.remove_class)("busy")
        self._dot.set_size_request(10, 10)
        if not self.get_visible():
            self.show_all()
            self._place()

    def set_transcript(self, text: str) -> None:
        if text:
            if len(text) > self.MAX_CHARS:
                text = "…" + text[-self.MAX_CHARS:].split(" ", 1)[-1]
            self._set_label(text, dim=False)

    def _set_label(self, text: str, dim: bool) -> None:
        self._label.set_text(text)
        ctx = self._label.get_style_context()
        (ctx.add_class if dim else ctx.remove_class)("dim")

    def push_level(self, rms: float) -> None:
        size = 10 + int(min(1.0, rms * 12) * 12)
        self._dot.set_size_request(size, size)


class App:
    LIVE_POLL_S = 0.15
    LIVE_MIN_SECONDS = 0.5

    def __init__(self) -> None:
        self.overlay = Overlay()
        self.recorder = Recorder(lambda rms: GLib.idle_add(self.overlay.push_level, rms))
        self.bluetooth = BluetoothMic()
        self.model = None
        # idle -> starting (mic coming up) -> recording -> processing -> idle
        self.state = "idle"
        self._altgr_down = False
        self._altgr_tainted = False  # another key was pressed during this AltGr hold
        self._generation = 0  # invalidates work that outlived its dictation session
        self._model_lock = threading.Lock()  # the live and final passes share one session

    def load_model(self) -> None:
        import onnx_asr
        import onnxruntime as ort

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        # ONNX Runtime refuses external-data files reached through the symlinks of the
        # default Hugging Face cache, so the model is materialised as plain files.
        from huggingface_hub import snapshot_download

        print(f"loading {MODEL} on {providers[0]} ...", flush=True)
        path = snapshot_download(MODEL_REPO, local_dir=MODEL_DIR)
        self.model = onnx_asr.load_model(MODEL, path, providers=providers)
        self._recognize(np.zeros(SAMPLE_RATE, np.float32))
        print("ready: tap AltGr to start dictating, tap again to paste", flush=True)

    def _recognize(self, audio: np.ndarray) -> str:
        with self._model_lock:
            return str(self.model.recognize(audio, sample_rate=SAMPLE_RATE)).strip()

    def on_press(self, key) -> None:
        if is_altgr(key):
            self._altgr_down = True
            self._altgr_tainted = False
        elif self._altgr_down:
            # AltGr is also a character modifier (e.g. AltGr+q = @): that is typing, not a tap.
            self._altgr_tainted = True
        elif key == keyboard.Key.esc and self.state in ("starting", "recording"):
            GLib.idle_add(self.cancel)

    def on_release(self, key) -> None:
        if is_altgr(key) and self._altgr_down:
            self._altgr_down = False
            if not self._altgr_tainted:
                GLib.idle_add(self.toggle)

    def toggle(self) -> bool:
        if self.state == "idle" and self.model:
            self.begin()
        elif self.state == "recording":
            self.finish()
        elif self.state == "starting":
            self.cancel()
        return False

    def begin(self) -> None:
        self.state = "starting"
        self._generation += 1
        self.overlay.show_state("starting mic...")
        threading.Thread(target=self._prepare_mic, args=(self._generation,), daemon=True).start()

    def _prepare_mic(self, generation: int) -> None:
        if not self.bluetooth.acquire():
            print("headset microphone did not appear; using the default input", file=sys.stderr, flush=True)
        GLib.idle_add(self._mic_ready, generation)

    def _mic_ready(self, generation: int) -> bool:
        if generation != self._generation or self.state != "starting":
            return False  # cancelled while the mic was coming up
        self.state = "recording"
        self.overlay.show_state("Listening...")
        self.recorder.start()
        threading.Thread(target=self._live_loop, args=(generation,), daemon=True).start()
        return False

    def _live_loop(self, generation: int) -> None:
        """Re-transcribes the audio so far, so the overlay follows the speech.

        Parakeet TDT is an offline model; on a GPU a full pass over the buffer is fast
        enough to run continuously and always yields the best transcript of everything said.
        """
        seen = 0
        while self.state == "recording" and generation == self._generation:
            audio = self.recorder.snapshot()
            if audio.size < self.LIVE_MIN_SECONDS * SAMPLE_RATE or audio.size == seen:
                time.sleep(self.LIVE_POLL_S)
                continue
            seen = audio.size
            try:
                text = self._recognize(audio)
            except Exception as exc:
                print(f"live transcription failed: {exc}", file=sys.stderr, flush=True)
                return
            if self.state == "recording" and generation == self._generation:
                GLib.idle_add(self.overlay.set_transcript, text)

    def cancel(self) -> bool:
        if self.state == "recording":
            self.recorder.stop()
        self._generation += 1
        self.state = "idle"
        self.overlay.hide()
        self.bluetooth.release_later()
        return False

    def finish(self) -> None:
        audio = self.recorder.stop()
        self.bluetooth.release_later()
        if audio.size < MIN_SECONDS * SAMPLE_RATE:
            self.cancel()
            return
        self._generation += 1  # stops the live loop
        self.state = "processing"
        self.overlay.show_state("Finishing...", busy=True)
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _transcribe(self, audio: np.ndarray) -> None:
        try:
            text = self._recognize(audio)
            if text:
                paste(text)
        except Exception as exc:  # keep the daemon alive across a bad utterance
            print(f"transcription failed: {exc}", file=sys.stderr, flush=True)
        finally:
            self.state = "idle"
            GLib.idle_add(self.overlay.hide)


def _run(*cmd: str, stdin: str | None = None) -> str:
    res = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    return res.stdout


def paste(text: str) -> None:
    previous = _run("xclip", "-selection", "clipboard", "-o")
    wm_class = _run("xdotool", "getactivewindow", "getwindowclassname").strip().lower()
    combo = "ctrl+shift+v" if wm_class in TERMINAL_CLASSES else "ctrl+v"
    subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
    subprocess.run(["xdotool", "key", "--clearmodifiers", combo], check=True)
    time.sleep(0.15)  # let the target app read the clipboard before restoring it
    subprocess.run(["xclip", "-selection", "clipboard"], input=previous.encode())


def main() -> None:
    app = App()
    atexit.register(app.bluetooth.restore)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    threading.Thread(target=app.load_model, daemon=True).start()
    listener = keyboard.Listener(on_press=app.on_press, on_release=app.on_release)
    listener.start()
    try:
        Gtk.main()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
