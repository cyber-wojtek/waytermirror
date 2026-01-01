# Waytermirror

Real-time Wayland screen mirroring to a terminal using Unicode braille characters, half‑blocks, or ASCII. Includes bidirectional input forwarding, audio streaming (PipeWire), zooming, focus-follow, and optional NVIDIA CUDA acceleration (server-side).

![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux-FCC624?logo=linux&logoColor=black)
![Display](https://img.shields.io/badge/Display-Wayland-1E88E5)

## Table of contents
- [What it is](#what-it-is)
- [How it works (short)](#how-it-works-short)
- [Quickstart](#quickstart)
- [Build & install](#build--install)
- [Runtime requirements & supported compositors](#runtime-requirements--supported-compositors)
- [Usage (server & client)](#usage)
  - [Full server options](#full-server-options)
  - [Full client options](#full-client-options)
- [Keyboard shortcuts (client)](#keyboard-shortcuts-client)
- [Network ports](#network-ports)
- [Performance tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Security & limitations](#security--limitations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## What it is
- A client/server application:
  - Server runs on the Wayland host (captures screen & audio, performs rendering, injects input).
  - Client runs in a terminal (receives ANSI frames, displays them, captures local input, sends it to server).
- Rendering modes: braille, half-blocks, ascii, and hybrid (auto-select per cell).
- Color modes: 16, 256, truecolor (24‑bit).
- Optional CUDA acceleration for server-side rendering (NVIDIA only).

## How it works (short)
- Screen → wlr-screencopy → frame buffer → renderer (CPU/CUDA) → ANSI string → LZ4 → TCP → client terminal
- Input capture on client → forwarded to server → virtual pointer/keyboard (Wayland) on host
- Audio (system → client) and microphone (client → server) via PipeWire

## Quickstart
1. Build (CPU-only)
   ```bash
   git clone https://github.com/cyber-wojtek/waytermirror.git
   cd waytermirror
   make -j$(nproc)
   ```
   This produces `waytermirror_server` and `waytermirror_client`.

2. Run the server on your Wayland desktop:
   ```bash
   ./waytermirror_server
   ```

3. Run the client in your terminal (replace <host>):
   ```bash
   ./waytermirror_client -H <host>
   ```

Tip: run `./waytermirror_client --help` or `./waytermirror_server --help` to see current/compiled-in flags and defaults on your build.

## Build & install

Prerequisites
- Core: gcc/g++, make, wayland, libinput, libudev, pipewire, lz4, rapidjson
- Optional: NVIDIA CUDA toolkit for GPU rendering (nvcc) — see NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit

Arch Linux (packages listed in [PKGBUILD](PKGBUILD))
- Example installable dependencies: wayland, wlroots0.19, wayland-protocols, libinput, libevdev, pipewire, lz4, rapidjson, lua, systemd-libs, glib2

Debian/Ubuntu
```bash
sudo apt install build-essential libwayland-dev libinput-dev libudev-dev libpipewire-0.3-dev liblz4-dev rapidjson-dev libwlroots-dev
# For CUDA: install NVIDIA CUDA toolkit from https://developer.nvidia.com/cuda-toolkit
```

Building (details)
- CPU-only (default):
  ```bash
  make
  ```
- CUDA-enabled (if you have nvcc and CUDA libs):
  ```bash
  make CUDA=true
  ```
- The provided PKGBUILD auto-detects nvcc. You can override with WAYTERMIRROR_CUDA=1 or WAYTERMIRROR_NO_CUDA when building on Arch.

Artifacts
- waytermirror_server
- waytermirror_client

Packaging (Arch)
- Use the included PKGBUILD with `makepkg`:
  ```bash
  git clone https://github.com/cyber-wojtek/waytermirror.git
  cd waytermirror
  makepkg -si
  ```

## Runtime requirements & supported compositors
- Compositor: wlroots-based (Hyprland, Sway, River) or any compositor exposing:
  - wlr-screencopy-unstable-v1 — required for screen capture
  - zwp_virtual_keyboard_v1 (virtual keyboard)
  - zwlr_virtual_pointer_v1 (virtual pointer)
  - wlr-foreign-toplevel-management (optional — focus following)
- Audio: PipeWire (for system audio streaming).
- Input: access to input devices (user must be in the `input` group or run with sufficient privileges to read /dev/input/*).
- NOT SUPPORTED: GNOME or KDE Plasma (they typically do not expose the required wlr protocols).

## Usage

Server
```bash
./waytermirror_server [options]
```

### Full server options
> For the authoritative list use `./waytermirror_server --help`.

| Flag / Short | Long / Name | Description | Default |
|--------------:|-------------|-------------|---------|
| -P <n> | --port <n> | Base TCP port (video base; other services use base+N) | 9999 |
| -F <n> | --capture-fps <n> | Capture framerate from compositor | 30 |
| -C <type> | --compositor <auto\|hyprland\|sway\|kde\|gnome\|generic> | Compositor override | auto |
| -n | --no-video | Disable screen capture / video streaming | off |
| -A | --no-audio | Disable system audio streaming | off |
| -N | --no-input | Disable input injection (do not create virtual devices) | off |
| -m | --no-microphone | Disable microphone reception (client→server mic) | off |

Client
```bash
./waytermirror_client -H <server_ip> [options]
```

### Full client options
> For the authoritative list use `./waytermirror_client --help`.

Connection
| Flag / Short | Long / Name | Description | Default |
|--------------:|-------------|-------------|---------|
| -H <addr> | --host <addr> | Server IP/hostname (required) | *required* |
| -P <n> | --port <n> | Server base port | 9999 |

Video & rendering
| Flag / Short | Long / Name | Description | Default |
|--------------:|-------------|-------------|---------|
| -o <n\|follow> | --output <n\|follow> | Output index or `follow` to track focused window | 0 |
| -F <n> | --fps <n> | Target client FPS / playback framerate | 30 |
| -M <16\|256\|true> | --mode <16\|256\|true> | Color mode (16, 256, truecolor) | 256 |
| -R <type> | --renderer <braille\|blocks\|ascii\|hybrid> | Rendering method | braille |
| -r <cpu\|cuda> | --render-device <cpu\|cuda> | Prefer server-side renderer | cpu |
| -d <0-100> | --detail-level <0-100> | Visual detail (0: fast/smooth, 100: sharp) | 50 |
| -Q <0-100> | --quality <0-100> | Pattern search precision | 50 |
| -S <factor> | --scale <factor> | Scale factor for rendered output | 1.0 |
| -k | --keep-aspect-ratio | Maintain aspect ratio when scaling | off |
| -c | --compress | Enable LZ4 compression | off |
| -L <0-12> | --compression-level <0-12> | LZ4 HC level (0=fast, 12=best) | 0 |
| -n | --no-video | Disable video display | off |

Input (local client input capture / forwarding)
| Flag / Short | Long / Name | Description | Default |
|--------------:|-------------|-------------|---------|
| -N | --no-input | Do not capture/forward local input | off |
| -x | --exclusive-input | Grab input devices exclusively (EVIOCGRAB) | off |
| -C | --center-mouse | Start mouse at screen center when connecting | off |

Audio
| Flag / Short | Long / Name | Description | Default |
|--------------:|-------------|-------------|---------|
| -A | --no-audio | Disable audio playback (server→client) | off |
| -p | --no-microphone | Disable microphone capture (client→server) | off |

Zoom / viewport
| Flag / Short | Long / Name | Description | Default |
|--------------:|-------------|-------------|---------|
| -z | --zoom | Start with zoom enabled | off |
| -Z <1-10> | --zoom-level <1-10> | Magnification | 2.0 |
| -X <px> | --zoom-width <px> | Viewport width (px) | 800 |
| -Y <px> | --zoom-height <px> | Viewport height (px) | 600 |
| -f | --zoom-follow | Follow mouse while zoomed | on |
| -s | --zoom-smooth | Smooth panning while zoomed | on |
| -D <n> | --zoom-speed <n> | Pan speed (px/frame) | 20 |

## Examples
- Basic LAN streaming:
  ```bash
  # Server (desktop)
  ./waytermirror_server -F 60

  # Client (terminal)
  ./waytermirror_client -H 192.168.1.100 -F 60 -M true -R hybrid
  ```

- High quality with CUDA (server built with CUDA support):
  ```bash
  ./waytermirror_client -H 192.168.1.100 -r cuda -d 90 -Q 100 -M true -R braille
  ```

- Low bandwidth:
  ```bash
  ./waytermirror_client -H server.example.com -c -L 12 -F 15 -M 256 -d 30
  ```

- Follow focused window:
  ```bash
  ./waytermirror_client -H 192.168.1.100 -o follow
  ```

- Input-only (no video, control remote desktop):
  ```bash
  ./waytermirror_client -H 192.168.1.100 -n -x
  ```

## Keyboard shortcuts (client)

This table documents the most useful client-side keyboard shortcuts. Shortcuts are designed for local control of the client (zoom, audio, input capture, toggles). By convention the client uses the local-modifier prefix Ctrl+Alt+Shift for client commands so that normal keys are forwarded to the remote session. If you need to send the same Ctrl+Alt+Shift+Key combination to the remote instead of triggering the client shortcut, either use the command-line flags to disable local capturing (e.g. --no-input) or check the client behavior for "double-press" forwarding (see notes below).

Note: Exact bindings may vary by build. Run `./waytermirror_client --help` or inspect waytermirror_client.cpp for the current compiled-in keys.

Default client shortcuts
| Shortcut | Action | Notes |
|---------:|--------|-------|
| Ctrl+Alt+Shift+Q | Quit client / disconnect | Graceful disconnect (sends close to server) |
| Ctrl+Alt+Shift+I | Toggle input forwarding | Enable/disable forwarding of keyboard & mouse to server |
| Ctrl+Alt+Shift+G | Toggle exclusive grab | EVIOCGRAB on local devices (when supported) |
| Ctrl+Alt+Shift+Z | Toggle zoom mode | When zoomed, use arrow keys to pan |
| Ctrl+Alt+Shift++ (or Ctrl+Alt+Shift+=) | Zoom in | Increases zoom level (same as -Z) |
| Ctrl+Alt+Shift+- | Zoom out | Decreases zoom level |
| Ctrl+Alt+Shift+0 | Reset zoom | Reset to 1.0x (or configured -Z default) |
| Ctrl+Alt+Shift+Arrow keys | Pan while zoomed | Left/Right/Up/Down — smooth panning if --zoom-smooth enabled |
| Ctrl+Alt+Shift+PageUp / PageDown | Fast pan while zoomed | Larger vertical pan steps |
| Ctrl+Alt+Shift+F | Toggle focus-follow | Follow focused output/window (if supported by compositor) |
| Ctrl+Alt+Shift+R | Cycle renderer | Cycle: braille → blocks → ascii → hybrid |
| Ctrl+Alt+Shift+C | Cycle color mode | Cycle: 16 → 256 → truecolor |
| Ctrl+Alt+Shift+D | Increase detail | Equivalent to raising --detail-level |
| Ctrl+Alt+Shift+S | Decrease detail | Equivalent to lowering --detail-level |
| Ctrl+Alt+Shift+P | Pause / resume video | Stops rendering updates locally (input still forwarded) |
| Ctrl+Alt+Shift+A | Toggle audio playback | Mute/unmute system audio stream |
| Ctrl+Alt+Shift+M | Toggle microphone capture | Enable/disable mic from client → server |

Quick usage tips
- Zoom panning: when zoomed (Ctrl+Alt+Shift+Z), use arrow keys to pan the viewport. Hold PageUp/PageDown for faster vertical movement.
- Input capture: enable exclusive grab (Ctrl+Alt+Shift+G) only if you trust the client host and want to avoid local desktop interaction while controlling remote.
- Forwarding vs local: if you want to send one of the Ctrl+Alt+Shift+... combos to the remote, either temporarily disable local shortcuts (e.g. Ctrl+Alt+Shift+I to stop input capturing) or use a build/flag to change behavior. Check `waytermirror_client --help` or the source file waytermirror_client.cpp for the exact forwarding rules.
- Customization: keyboard handling is implemented in the client source. To change shortcuts, edit waytermirror_client.cpp and rebuild.

## Network ports
- (base port)         TCP — video frames
- (base port + 1)     TCP — input events
- (base port + 2)     TCP — system audio (server → client)
- (base port + 3)     TCP — configuration/control
- (base port + 4)     TCP — microphone (client → server)

Default base port is 9999 (see -P / --port).

## Performance tuning
- Maximum quality:
  - Server: render_device=cuda, renderer=braille, detail=100, quality=100, color=true
  - Client example: `-r cuda -R braille -d 100 -Q 100 -M true -F 30`
- Smooth video:
  - Use hybrid renderer with moderate detail and higher FPS: `-R hybrid -d 30 -F 60`
- Low bandwidth:
  - Enable compression & reduce color/depth/FPS: `-c -L 9 -M 256 -F 15 -d 30`
- Low latency:
  - Increase capture FPS and lower quality search: `-F 60 -Q 0 -d 50`

## Troubleshooting
- "Failed to initialize libinput"
  ```bash
  sudo usermod -aG input $USER
  # Log out and back in
  ```
- Check compositor support for screencopy:
  ```bash
  wayland-info 2>/dev/null | grep -i screencopy
  ```
  If no screencopy global is present, the compositor doesn't support `wlr-screencopy-unstable-v1`.

- Virtual input not working
  - Server prints messages about virtual devices on startup. Look for messages like:
    ```
    Virtual pointer created: YES
    Virtual keyboard created: YES
    ```
  - If NO, the compositor does not expose required protocols.

- Check PipeWire (audio not working):
  ```bash
  systemctl --user status pipewire
  ```

- CUDA errors / verify GPU:
  ```bash
  nvcc --version
  nvidia-smi
  ```

- Permissions to /dev/input
  - Ensure client (when capturing local input) can read devices or run with privileges. Exclusive grab (-x) uses EVIOCGRAB and requires access.

- Logs & debugging:
  - The server prints status and detection lines to stdout/stderr. Inspect output for messages about compositor detection, screencopy frames, and virtual device creation.
  - For current flags and runtime behavior, prefer `--help` output for the built binaries.

## Security & limitations
- No built-in authentication: protocol uses raw TCP streams. Do not expose the server to untrusted networks. Use an SSH tunnel or VPN for remote connections.
- Requires compositor support for the listed wlroots protocols. GNOME/KDE typically lack required wlr protocols — not supported.
- Input injection requires compositor permission — some compositors may restrict virtual devices.

## Design notes & behavior
- Rendering is performed server-side. The client displays ANSI/escape sequences sent from server — client CPU requirements are minimal.
- Optional CUDA renderer: when built with CUDA, server uses GPU for rendering; otherwise, CPU fallback is used (braille_renderer_stub.cpp).
- Hybrid renderer chooses per-cell between braille and half-blocks based on local variance (detail-level influences decisions).

## Contributing
- Bug reports, feature requests and PRs welcome. Please open issues/PRs on the GitHub project: https://github.com/cyber-wojtek/waytermirror
- See the PKGBUILD for packaging guidance.

## License
- Apache License 2.0 — see [LICENSE](LICENSE)

## Acknowledgments
- wlroots & Wayland protocols — https://github.com/swaywm/wlroots
- PipeWire for audio — https://pipewire.org
- LZ4 for compression — https://github.com/lz4/lz4
- argparse (CLI parsing)

If you need a smaller quick-help snippet, a printable cheat-sheet for shortcuts, or a sample systemd unit for running the server persistently on a particular distribution, tell me your target and I’ll provide a template.
