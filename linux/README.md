# FluidVoice for Linux (X11)

Tap **AltGr** to start: a pill overlay appears and shows a live transcript as
you speak. Tap **AltGr** again to paste the final text into the focused window.
**Esc** cancels. Uses NVIDIA Parakeet TDT 0.6B v3 (the same model family as the
macOS app) through ONNX Runtime on CUDA.

## Requirements
- X11 session, NVIDIA GPU with a recent driver (falls back to CPU otherwise)
- `pw-record` (PipeWire), `xdotool`, `xclip`
- `uv` and the system PyGObject/GTK3 (`python3-gi`, `gir1.2-gtk-3.0`)

## Run
    ./fluidvoice_linux.py

Dependencies are declared in the script header and installed by `uv` on first
run. The first start also downloads the model (~3 GB) to `~/.cache/fluidvoice/`.

Pressing another key while AltGr is held does not count as a tap, so typing
characters such as `@` with AltGr still works.

The live view re-transcribes everything said so far several times a second
(Parakeet TDT is an offline model), which is cheap on a GPU but would be slow
on CPU.

## Bluetooth headsets
Bluetooth headphones in A2DP mode have no microphone. When the default output
is a Bluetooth device, the app switches it to its headset profile while
dictation is active (playback quality drops meanwhile) and restores A2DP 8 seconds
after the last dictation. Set `FLUIDVOICE_BT_SWITCH=0` to disable this.
