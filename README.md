# Waytermirror

Real-time Wayland screen mirroring to a terminal using Unicode braille characters, half‑blocks, or ASCII. Includes bidirectional input forwarding, audio streaming (PipeWire), zooming, focus-follow, and optional NVIDIA CUDA acceleration (server-side).

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux-FCC624?logo=linux&logoColor=black)
![Display](https://img.shields.io/badge/Display-Wayland-1E88E5)
![Releases](https://img.shields.io/github/v/release/cyber-wojtek/waytermirror?label=Releases&sort=semver)
![Stars](https://img.shields.io/github/stars/cyber-wojtek/waytermirror?style=)
![Forks](https://img.shields.io/github/forks/cyber-wojtek/waytermirror?style=)
![Issues](https://img.shields.io/github/issues/cyber-wojtek/waytermirror)
![Contributions welcome](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg)
![Build Status](https://img.shields.io/github/actions/workflow/status/cyber-wojtek/waytermirror/build.yml?branch=main)

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
- Core: [gcc/g++](https://github.com/gcc-mirror/gcc.git), [make](https://git.savannah.gnu.org/git/make.git), [wayland](https://gitlab.com/freedesktop-sdk/mirrors/freedesktop/wayland/wayland.git), [wayland-protocols](https://gitlab.freedesktop.org/wayland/wayland-protocols.git), [libinput](https://gitlab.freedesktop.org/libinput/libinput.git), [pipewire](https://gitlab.freedesktop.org/pipewire/pipewire.git), [lz4](https://github.com/lz4/lz4.git), [rapidjson](https://github.com/Tencent/rapidjson.git), [systemd](https://github.com/systemd/systemd.git), [argparse](https://github.com/p-ranav/argparse.git)
- Optional: NVIDIA CUDA toolkit for GPU rendering (nvcc) — see NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit

Arch Linux-based
```bash
sudo pacman -S base-devel git gcc wayland wayland-protocols libinput pipewire lz4 rapidjson systemd argparse 
# Optional for CUDA:
sudo pacman -S cuda
```

Debian-based
```bash
sudo apt install build-essential git gcc libwayland-dev wayland-protocols libinput-dev libpipewire-0.3-dev liblz4-dev rapidjson-dev libsystemd-dev pkg-config cmake libargparse-dev
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
- The provided PKGBUILD auto-detects nvcc. You can override with WAYTERMIRROR_CUDA=1 or WAYTERMIRROR_NO_CUDA when building with it.

Artifacts
- waytermirror_server
- waytermirror_client

Packaging (Arch)
- **AUR**: Install directly from the [waytermirror-git](https://aur.archlinux.org/packages/waytermirror-git) AUR package using your favorite AUR helper:
  ```bash
  yay -S waytermirror-git
  # or
  paru -S waytermirror-git
  ```

- **Manual build**: Use the included PKGBUILD with `makepkg`:
  ```bash
  git clone https://github.com/cyber-wojtek/waytermirror.git
  cd waytermirror
  makepkg -si
  ```

## Runtime requirements
- Audio: PipeWire (for system audio streaming).
- Input: access to input devices (user must be in the `input` group or run with sufficient privileges to read /dev/input/*).

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
| -B <backend> | --capture-backend <auto\|wlr\|pipewire> | Screen capture backend (see notes below) | auto |
| -I <backend> | --input-backend <auto\|virtual\|uinput> | Input injection backend | auto |
| -n | --no-video | Disable screen capture / video streaming | off |
| -A | --no-audio | Disable system audio streaming | off |
| -N | --no-input | Disable input injection (do not create virtual devices) | off |
| -m | --no-microphone | Disable microphone reception (client→server mic) | off |

**Capture backend notes:**
- `wlr`: Uses wlr-screencopy protocol directly (supports all outputs, compositor must support wlr-screencopy-unstable-v1)
- `pipewire`: Uses PipeWire + xdg-desktop-portal for screen capture (works on more compositors including GNOME/KDE)
  - **Important**: When using PipeWire backend, you'll be prompted to select screens. Select them in their **logical index order** (0, then 1, then 2, etc.) to match the output indices used by the client's `-o` option.
- `auto`: Automatically detects and prefers wlr-screencopy if available, falls back to PipeWire

**Input backend notes:**
- `virtual`: Uses Wayland virtual input protocols (zwlr_virtual_pointer_v1, zwp_virtual_keyboard_v1) — requires compositor support
- `uinput`: Uses Linux uinput (/dev/uinput) — works on any compositor but requires proper permissions
- `auto`: Automatically selects virtual protocols if available, falls back to uinput

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

All client shortcuts use the **Ctrl+Alt+Shift** modifier prefix, so normal keys are forwarded to the remote session. Press **Ctrl+Alt+Shift+H** at any time to display the full shortcut list with current toggle states.

### Session control
| Shortcut | Action | Notes |
|---------:|--------|-------|
| Ctrl+Alt+Shift+Q | Quit / disconnect | Graceful disconnect (sends close to server) |
| Ctrl+Alt+Shift+H | Toggle help | Display all shortcuts and current state |
| Ctrl+Alt+Shift+P | Pause / resume video | Stops rendering updates locally (input still forwarded) |

### Input control
| Shortcut | Action | Notes |
|---------:|--------|-------|
| Ctrl+Alt+Shift+I | Toggle input forwarding | Enable/disable forwarding of keyboard & mouse to server |
| Ctrl+Alt+Shift+G | Toggle exclusive grab | EVIOCGRAB on local devices (when supported) |

### Zoom control
| Shortcut | Action | Notes |
|---------:|--------|-------|
| Ctrl+Alt+Shift+Z | Toggle zoom mode | When zoomed, use arrow keys to pan |
| Ctrl+Alt+Shift++ (or =) | Zoom in | Increases zoom level by 0.5x |
| Ctrl+Alt+Shift+- | Zoom out | Decreases zoom level by 0.5x |
| Ctrl+Alt+Shift+0 | Reset zoom | Reset to 2.0x and center viewport |
| Ctrl+Alt+Shift+N | Toggle zoom follow | Enable/disable zoom following mouse cursor |
| Ctrl+Alt+Shift+Arrow keys | Pan viewport | Left/Right/Up/Down — uses configured pan speed |
| Ctrl+Alt+Shift+PageUp/PageDown | Fast vertical pan | 5× normal pan speed |
| Ctrl+Alt+Shift+Home/End | Fast horizontal pan | 5× normal pan speed |

### Rotation
| Shortcut | Action | Notes |
|---------:|--------|-------|
| Ctrl+Alt+Shift+[ | Rotate left 15° | Counter-clockwise rotation |
| Ctrl+Alt+Shift+] | Rotate right 15° | Clockwise rotation |
| Ctrl+Alt+Shift+\ | Reset rotation | Return to 0° |
| Ctrl+Alt+Shift+T | Rotate 90° CW | Quick 90° clockwise rotation |
| Ctrl+Alt+Shift+Y | Rotate 90° CCW | Quick 90° counter-clockwise rotation |

### Rendering
| Shortcut | Action | Notes |
|---------:|--------|-------|
| Ctrl+Alt+Shift+R | Cycle renderer | braille → blocks → ascii → hybrid |
| Ctrl+Alt+Shift+1 | Braille renderer | Quick switch to braille |
| Ctrl+Alt+Shift+2 | Blocks renderer | Quick switch to half-blocks |
| Ctrl+Alt+Shift+3 | ASCII renderer | Quick switch to ASCII characters |
| Ctrl+Alt+Shift+4 | Hybrid renderer | Quick switch to hybrid (auto per-cell) |
| Ctrl+Alt+Shift+C | Cycle color mode | 16 → 256 → truecolor |
| Ctrl+Alt+Shift+D | Increase detail | +10 detail level |
| Ctrl+Alt+Shift+S | Decrease detail | −10 detail level |
| Ctrl+Alt+Shift+W | Increase quality | +10 quality level |
| Ctrl+Alt+Shift+E | Decrease quality | −10 quality level |
| Ctrl+Alt+Shift+O | Toggle smooth panning | Enable/disable smooth zoom panning |
| Ctrl+Alt+Shift+B | Toggle aspect ratio | Keep/ignore aspect ratio when scaling |
| Ctrl+Alt+Shift+V | Cycle render device | CPU → CUDA (if available) |
| Ctrl+Alt+Shift+U | Toggle compression | Enable/disable LZ4 compression |
| Ctrl+Alt+Shift+L | Cycle compression level | Off → fast LZ4 → HC levels |

### Output & FPS
| Shortcut | Action | Notes |
|---------:|--------|-------|
| Ctrl+Alt+Shift+` | Cycle output | Next output or toggle follow-focus |
| Ctrl+Alt+Shift+J | Increase FPS | +5 FPS |
| Ctrl+Alt+Shift+K | Decrease FPS | −5 FPS (min 0) |
| Ctrl+Alt+Shift+F | Toggle focus-follow | Follow focused output/window |

### Audio
| Shortcut | Action | Notes |
|---------:|--------|-------|
| Ctrl+Alt+Shift+A | Toggle audio | Mute/unmute system audio playback |
| Ctrl+Alt+Shift+M | Toggle microphone | Mute/unmute microphone capture |

Quick usage tips
- Zoom panning: when zoomed (Ctrl+Alt+Shift+Z), use arrow keys to pan the viewport. Hold PageUp/PageDown for faster vertical movement.
- Rotation: use **[** and **]** to rotate in 15° steps, **T**/**Y** for 90° jumps, **\\** to reset. Rotation is handled natively by CUDA when available.
- Quick renderer switch: use **1-4** to instantly select braille/blocks/ascii/hybrid renderers instead of cycling with **R**.
- FPS adjustment: use **J** to increase and **K** to decrease FPS by 5 (range: 1-120).
- Output cycling: press **`** (backtick) to cycle through outputs or toggle follow-focus mode.
- Compression toggle: use **U** to quickly enable/disable LZ4 compression.
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
- Input injection requires elevated privileges or membership in the `input` group to read /dev/input devices unless compositor supports virtual input protocols.

## Design notes & behavior
- Rendering is performed server-side. The client displays ANSI/escape sequences sent from server — client CPU requirements are minimal.
- Optional CUDA renderer: when built with CUDA, server can use GPU for rendering; otherwise, only CPU is available.
- Hybrid renderer chooses per-cell between braille and half-blocks.

## Contributing
- Bug reports, feature requests and PRs welcome.

## License
- MIT License — see [LICENSE](LICENSE) file for details.
