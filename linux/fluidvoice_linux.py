#!/usr/bin/env python3
"""Minimal Linux (X11) dictation: hold AltGr, speak, release to paste.

Runs the same Parakeet TDT v3 model family as the macOS app, via ONNX Runtime on CUDA.
"""
from __future__ import annotations

import glob
import os
import site
import subprocess
import sys
import threading
import time

import numpy as np


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

    def stop(self) -> np.ndarray:
        proc, self._proc = self._proc, None
        if proc:
            proc.terminate()
            if self._thread:
                self._thread.join(timeout=1)
            proc.wait()
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)


class Overlay(Gtk.Window):
    """Small always-on-top pill at the bottom centre that never takes focus.

    Built from stock widgets and CSS: custom cairo drawing needs python3-gi-cairo,
    which is not installed everywhere.
    """

    BARS = 20
    CSS = b"""
    .pill { background-color: rgba(20, 20, 26, 0.92); border-radius: 28px; padding: 0 22px; }
    .pill label { color: #9fcfff; font-size: 15px; }
    .bar { background-color: #ff5959; border-radius: 3px; min-width: 5px; }
    """

    def __init__(self) -> None:
        super().__init__(type=Gtk.WindowType.POPUP)
        self.set_decorated(False)
        self.set_accept_focus(False)
        self.set_keep_above(True)
        self.set_skip_taskbar_hint(True)
        self.set_app_paintable(True)
        self.set_default_size(260, 56)
        visual = self.get_screen().get_rgba_visual()
        if visual:
            self.set_visual(visual)

        provider = Gtk.CssProvider()
        provider.load_from_data(self.CSS)
        Gtk.StyleContext.add_provider_for_screen(
            self.get_screen(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self._levels = [0.0] * self.BARS
        self._bars: list[Gtk.Box] = []
        self._bar_row = Gtk.Box(spacing=4, halign=Gtk.Align.CENTER, valign=Gtk.Align.CENTER)
        for _ in range(self.BARS):
            bar = Gtk.Box(valign=Gtk.Align.CENTER)
            bar.get_style_context().add_class("bar")
            bar.set_size_request(5, 4)
            self._bar_row.pack_start(bar, False, False, 0)
            self._bars.append(bar)
        self._label = Gtk.Label()

        self._stack = Gtk.Stack()
        self._stack.add_named(self._bar_row, "recording")
        self._stack.add_named(self._label, "text")
        self._stack.get_style_context().add_class("pill")
        self.add(self._stack)

    def _place(self) -> None:
        display = Gdk.Display.get_default()
        _, x, y = display.get_default_seat().get_pointer().get_position()
        geo = display.get_monitor_at_point(x, y).get_geometry()
        w, h = self.get_size()
        self.move(geo.x + (geo.width - w) // 2, geo.y + geo.height - h - 80)

    def show_state(self, status: str) -> None:
        if status == "recording":
            self._levels = [0.0] * self.BARS
            self._refresh_bars()
            self._stack.set_visible_child_name("recording")
        else:
            self._label.set_text(status)
            self._stack.set_visible_child_name("text")
        if not self.get_visible():
            self.show_all()
            self._place()

    def push_level(self, rms: float) -> None:
        self._levels = self._levels[1:] + [min(1.0, rms * 12)]
        self._refresh_bars()

    def _refresh_bars(self) -> None:
        for bar, level in zip(self._bars, self._levels):
            bar.set_size_request(5, max(4, int(level * 38)))


class App:
    def __init__(self) -> None:
        self.overlay = Overlay()
        self.recorder = Recorder(lambda rms: GLib.idle_add(self.overlay.push_level, rms))
        self.model = None
        self.recording = False
        self.cancelled = False
        self.busy = False

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
        self.model.recognize(np.zeros(SAMPLE_RATE, np.float32), sample_rate=SAMPLE_RATE)
        print("ready: hold AltGr to dictate", flush=True)

    def on_press(self, key) -> None:
        if is_altgr(key):
            if not self.recording and not self.busy and self.model:
                GLib.idle_add(self.begin)
        elif self.recording:
            # AltGr is also a character modifier (e.g. AltGr+q = @): that is typing, not dictation.
            self.cancelled = True

    def on_release(self, key) -> None:
        if is_altgr(key) and self.recording:
            GLib.idle_add(self.finish)

    def begin(self) -> bool:
        self.recording = True
        self.cancelled = False
        self.overlay.show_state("recording")
        self.recorder.start()
        return False

    def finish(self) -> bool:
        self.recording = False
        audio = self.recorder.stop()
        if self.cancelled or audio.size < MIN_SECONDS * SAMPLE_RATE:
            self.overlay.hide()
            return False
        self.busy = True
        self.overlay.show_state("transcribing...")
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()
        return False

    def _transcribe(self, audio: np.ndarray) -> None:
        try:
            text = str(self.model.recognize(audio, sample_rate=SAMPLE_RATE)).strip()
            if text:
                paste(text)
        except Exception as exc:  # keep the daemon alive across a bad utterance
            print(f"transcription failed: {exc}", file=sys.stderr, flush=True)
        finally:
            self.busy = False
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
    threading.Thread(target=app.load_model, daemon=True).start()
    listener = keyboard.Listener(on_press=app.on_press, on_release=app.on_release)
    listener.start()
    try:
        Gtk.main()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
