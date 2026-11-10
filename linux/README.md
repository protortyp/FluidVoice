# FluidVoice for Linux (X11)

Hold **AltGr**, speak, release: a pill overlay shows the mic level, then the
transcript is pasted into the focused window. Uses NVIDIA Parakeet TDT 0.6B v3
(the same model family as the macOS app) through ONNX Runtime on CUDA.

## Requirements
- X11 session, NVIDIA GPU with a recent driver (falls back to CPU otherwise)
- `pw-record` (PipeWire), `xdotool`, `xclip`
- System Python with PyGObject/GTK3 (`python3-gi`, `gir1.2-gtk-3.0`), `uv`

## Run
    ./setup.sh
    .venv/bin/python fluidvoice_linux.py

The first start downloads the model (~3 GB) to `~/.cache/fluidvoice/`.

Pressing another key while AltGr is held cancels the dictation, so typing
characters such as `@` with AltGr still works.
