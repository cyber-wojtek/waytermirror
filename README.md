# Waytermirror

Real-time Wayland screen mirroring to a terminal using Unicode braille characters, half‑blocks, or ASCII. Includes bidirectional input forwarding, audio streaming (PipeWire), zooming, focus-follow, and optional NVIDIA CUDA acceleration (server-side).

<!-- [![Release](https://img.shields.io/github/v/release/cyber-wojtek/waytermirror?label=release)](https://github.com/cyber-wojtek/waytermirror/releases) --->

![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux-FCC624?logo=linux&logoColor=black)
![Display](https://img.shields.io/badge/Display-Wayland-1E88E5)
[![Stars](https://img.shields.io/github/stars/cyber-wojtek/waytermirror?style=)](https://github.com/cyber-wojtek/waytermirror/stargazers)
[![Open Issues](https://img.shields.io/github/issues/cyber-wojtek/waytermirror?color=orange)](https://github.com/cyber-wojtek/waytermirror/issues)
[![PKGBUILD](https://img.shields.io/badge/Packaging-Arch%20PKGBUILD-blue?logo=arch-linux)](https://github.com/cyber-wojtek/waytermirror/blob/main/PKGBUILD)

## Table of contents
- [What it is](#what-it-is)
- [How it works (short)](#how-it-works-short)
- [Quickstart](#quickstart)
- [Build & install](#build--install)
- [Runtime requirements & supported compositors](#runtime-requirements--supported-compositors)
- [Usage (server & client examples)](#usage)
  - [Full server options](#full-server-options)
  - [Full client options](#full-client-options)
- [Network ports](#network-ports)
- [Performance tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Security & limitations](#security--limitations)
- [Contributing](#contributing)
- [License](#license)

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

3. Run the client in the terminal (replace <host>):
   ```bash
   ./waytermirror_client -H <host>
   ```

## Build & install

Prerequisites
- Core: gcc/g++, make, wayland, libinput, libudev, pipewire, lz4, rapidjson
- Optional: NVIDIA CUDA toolkit for GPU rendering (nvcc) — see NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit

Arch Linux (packages listed in [PKGBUILD](https://github.com/cyber-wojtek/waytermirror/blob/main/PKGBUILD))
- Example installable dependencies: wayland, wlroots0.19, wayland-protocols, libinput, libevdev, pipewire, lz4, rapidjson, lua, systemd-libs, glib2

Debian/Ubuntu
```bash
sudo apt install build-essential libwayland-dev libinput-dev libudev-dev libpipewire-0.3-dev liblz4-dev rapidjson-dev libwlroots-dev
# For CUDA: install NVIDIA CUDA toolkit from https://developer.nvidia.com/cuda-toolkit
```

## Building (details)
- CPU-only (default):
  ```bash
  make
  ```
- CUDA-enabled (if you have nvcc and CUDA libs):
  - Option 1: explicit make flag
    ```bash
    make CUDA=true
    ```
  - Option 2: let PKGBUILD detect nvcc or use env override (when using the provided PKGBUILD)
    ```bash
    # PKGBUILD build() supports WAYTERMIRROR_CUDA=1 or WAYTERMIRROR_NO_CUDA
    ```

## Artifacts
- waytermirror_server
- waytermirror_client

## Packaging (Arch)
- Use the included PKGBUILD with `makepkg`:
  ```bash
  git clone https://github.com/cyber-wojtek/waytermirror.git
  cd waytermirror
  makepkg -si
  ```
- The PKGBUILD tries to auto-detect CUDA if `nvcc` is present. You can override with WAYTERMIRROR_CUDA or WAYTERMIRROR_NO_CUDA.

## Runtime requirements & supported compositors
- Compositor: wlroots-based (Hyprland, Sway, River) or any compositor exposing:
  - [wlr-screencopy-unstable-v1](https://github.com/swaywm/wlroots/tree/master/protocols) — required for screen capture
  - [zwp_virtual_keyboard_v1](https://github.com/wayland-project/wayland-protocols) (virtual keyboard)
  - [zwlr_virtual_pointer_v1](https://github.com/swaywm/wlroots/tree/master/protocols) (virtual pointer)
  - wlr-foreign-toplevel-management (optional — focus following)
- Audio: [PipeWire](https://pipewire.org) (for system audio streaming).
- Input: access to input devices (user must be in the `input` group or run with sufficient privileges to read /dev/input/*).
- NOT SUPPORTED: GNOME or KDE Plasma (they typically do not expose the required wlr protocols).

## Usage

Server
```bash
./waytermirror_server [options]
```

### Full server options
> Note: This README consolidates the known server options from the project documentation. For the authoritative, up-to-date list use `./waytermirror_server --help`.

| Flag / Short | Long / Name | Description | Default |
|--------------:|-------------|-------------|---------|
| -P <n> | --port <n> | Base TCP port (video base; other services use base+N) | 9999 |
| -F <n> | --capture-fps <n> | Capture framerate from compositor | 30 |
| -C <type> | --compositor <auto\|hyprland\|sway\|kde\|gnome\|generic> | Compositor override (affects protocol handling/focus) | auto |
| -n | --no-video | Disable screen capture / video streaming | off |
| -A | --no-audio | Disable system audio streaming | off |
| -N | --no-input | Disable input injection (do not create virtual devices) | off |
| -m | --no-microphone | Disable microphone reception (client→server mic) | off |

Common additional server behaviors:
- Server detects compositor type automatically (env XDG_CURRENT_DESKTOP / WAYLAND_DISPLAY heuristics) unless overridden with --compositor.
- Server prints diagnostic lines about virtual device creation and screencopy availability on startup.

Client
```bash
./waytermirror_client -H <server_ip> [options]
```

### Full client options
> Note: This README consolidates the known client options from the project documentation. For the authoritative, up-to-date list use `./waytermirror_client --help`.

Connection
| Flag / Short | Long / Name | Description | Default |
|--------------:|-------------|-------------|---------|
| -H <addr> | --host <addr> | Server IP/hostname (required) | *required* |
| -P <n> | --port <n> | Server base port | 9999 |

Video & rendering
| Flag / Short | Long / Name | Description | Default |
|--------------:|-------------|-------------|---------|
| -o <n\|follow> | --output <n\|follow> | Output index or `follow` to track focused window | 0 |
| -F <n> | --fps <n> | Target client FPS / target playback framerate | 30 |
| -M <16\|256\|true> | --mode <16\|256\|true> | Color mode (16, 256, truecolor) | 256 |
| -R <type> | --renderer <braille\|blocks\|ascii\|hybrid> | Rendering method (hybrid auto-selects per-cell) | braille |
| -r <cpu\|cuda> | --render-device <cpu\|cuda> | Prefer server-side rendering backend (cpu or cuda) | cpu |
| -d <0-100> | --detail-level <0-100> | Visual detail (0: fast/smooth, 100: sharp) | 50 |
| -Q <0-100> | --quality <0-100> | Pattern search precision / quality threshold | 50 |
| -S <factor> | --scale <factor> | Scale factor for rendered output | 1.0 |
| -k | --keep-aspect-ratio | Maintain aspect ratio when scaling | off |
| -c | --compress | Enable LZ4 compression (client↔server frames) | off |
| -L <0-12> | --compression-level <0-12> | LZ4 HC level (0=fast, 12=best ratio) | 0 |
| -n | --no-video | Disable video display (useful for input-only sessions) | off |

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
| -f | --zoom-follow | Follow mouse cursor while zoomed | on |
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

- High quality with CUDA (server must be built with CUDA support):
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

## Network ports
- (base port)         TCP — video frames
- (base port + 1)     TCP — input events
- (base port + 2)     TCP — system audio (server → client)
- (base port + 3)     TCP — configuration/control
- (base port + 4)     TCP — microphone (client → server)

Default base port is 9999 (see -P / --port).

## Performance tuning
- Maximum quality (more GPU/CPU work, higher bandwidth):
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
  If no screencopy global is present, the compositor doesn't support `wlr-screencopy-unstable-v1` — see wlroots protocols: https://github.com/swaywm/wlroots/tree/master/protocols

- Virtual input not working
  - Server will print messages about virtual devices. Look for:
    ```
    Virtual pointer created: YES
    Virtual keyboard created: YES
    ```
  - If NO, the compositor does not expose required protocols.

- Check PipeWire (audio not working):
  ```bash
  systemctl --user status pipewire
  ```
  PipeWire docs: https://pipewire.org

- CUDA errors / verify GPU:
  ```bash
  nvcc --version
  nvidia-smi
  ```
  NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit

- Permissions to /dev/input
  - Ensure client (when capturing local input) can read devices or run with privileges. Exclusive grab (-x) uses EVIOCGRAB and requires access.

- Logs & debugging:
  - The server prints status and detection lines to stdout/stderr. Inspect output for messages about compositor detection, screencopy frames, and virtual device creation.
  - If options differ from this README, prefer `--help` output from the built binaries for the latest flags.

## Security & limitations
- No built-in authentication: the protocol uses raw TCP streams. Avoid exposing the server to untrusted networks. Use an SSH tunnel or VPN if you need to connect over the internet.
- Requires compositor support for the listed wlroots protocols. GNOME/KDE typically lack required wlr protocols — not supported.
- Input injection requires compositor permission — some compositors will restrict virtual devices.

## Design notes & behavior
- Rendering is performed server-side. The client displays ANSI/escape sequences sent from server — client CPU requirements are minimal.
- Optional CUDA renderer: when built with CUDA, server uses GPU for rendering; otherwise, CPU fallback is used (stub in braille_renderer_stub.cpp).
- Hybrid renderer chooses per-cell between braille and half-blocks based on local variance (detail-level influences decisions).

## Contributing
- Bug reports, feature requests and PRs welcome. Please follow repository contribution guidelines (open issues & PRs on the [GitHub project](https://github.com/cyber-wojtek/waytermirror)).
- See the [PKGBUILD](https://github.com/cyber-wojtek/waytermirror/blob/main/PKGBUILD) for a reproducible Arch packaging example.

## License
- Apache License 2.0 — see [LICENSE](LICENSE)

## Acknowledgments
- wlroots & Wayland protocols — https://github.com/swaywm/wlroots
- PipeWire for audio — https://pipewire.org
- LZ4 for compression — https://github.com/lz4/lz4
- argparse (CLI parsing)

If you need a smaller quick-help snippet or a sample systemd unit for running the server persistently, tell me your target distribution and I’ll provide a template.
