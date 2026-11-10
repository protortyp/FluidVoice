# FluidVoice for Linux (X11)

Hold **AltGr**, speak, release: a pill overlay shows the mic level, then the
transcript is pasted into the focused window. Uses NVIDIA Parakeet TDT 0.6B v3
(the same model family as the macOS app) through ONNX Runtime on CUDA.

## Requirements
- X11 session, NVIDIA GPU with a recent driver (falls back to CPU otherwise)
- `pw-record` (PipeWire), `xdotool`, `xclip`
- `uv` and the system PyGObject/GTK3 (`python3-gi`, `gir1.2-gtk-3.0`)

## Run
    ./fluidvoice_linux.py

Dependencies are declared in the script header and installed by `uv` on first
run. The first start also downloads the model (~3 GB) to `~/.cache/fluidvoice/`.

Pressing another key while AltGr is held cancels the dictation, so typing
characters such as `@` with AltGr still works.

## Bluetooth headsets
Bluetooth headphones in A2DP mode have no microphone. When the default output
is a Bluetooth device, the app switches it to its headset profile while
dictating (playback quality drops meanwhile) and restores A2DP 8 seconds after
the last dictation. Set `FLUIDVOICE_BT_SWITCH=0` to disable this.
