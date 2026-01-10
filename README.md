# Waytermirror

Real-time Wayland screen mirroring to a terminal or native window using Unicode braille characters, half‑blocks, ASCII, sixels, kitty graphics, framebuffer, KMS direct rendering, native GUI window, or hybrid rendering. Includes bidirectional input forwarding, audio streaming (PipeWire), zooming, focus-follow, and optional NVIDIA CUDA acceleration (server-side).

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
- [Waytermirror](#waytermirror)
  - [Table of contents](#table-of-contents)
  - [What it is](#what-it-is)
  - [Demonstration](#demonstration)
  - [How it works (short)](#how-it-works-short)
  - [Quickstart](#quickstart)
  - [Build \& install](#build--install)
    - [Prerequisites](#prerequisites)
    - [Arch Linux-based](#arch-linux-based)
    - [Debian-based](#debian-based)
    - [Building](#building)
    - [Artifacts](#artifacts)
    - [Packaging (Arch)](#packaging-arch)
  - [Runtime requirements](#runtime-requirements)
  - [Usage](#usage)
    - [Server](#server)
      - [Full server options](#full-server-options)
    - [Client](#client)
      - [Full client options](#full-client-options)
  - [Examples](#examples)
  - [Keyboard shortcuts (client)](#keyboard-shortcuts-client)
    - [Session control](#session-control)
    - [Input control](#input-control)
    - [Zoom control](#zoom-control)
    - [Rotation](#rotation)
    - [Rendering](#rendering)
    - [Output \& FPS](#output--fps)
    - [Audio](#audio)
    - [Quick usage tips](#quick-usage-tips)
  - [Network ports](#network-ports)
  - [Performance tuning](#performance-tuning)
    - [Hardware-accelerated pixel rendering (best visual quality)](#hardware-accelerated-pixel-rendering-best-visual-quality)
    - [Unicode rendering (terminal-based)](#unicode-rendering-terminal-based)
    - [Encoder-specific tuning](#encoder-specific-tuning)
  - [Troubleshooting](#troubleshooting)
  - [Security \& limitations](#security--limitations)
  - [Design notes \& behavior](#design-notes--behavior)
    - [Rendering architecture](#rendering-architecture)
    - [Hardware acceleration](#hardware-acceleration)
    - [Compression](#compression)
    - [Hybrid renderer](#hybrid-renderer)
    - [GUI renderer](#gui-renderer)
  - [Contributing](#contributing)
  - [License](#license)

## What it is
- A client/server application:
  - **Server** runs on the Wayland host (captures screen & audio, performs rendering, injects input).
  - **Client** runs locally (receives frames, displays them, captures input, sends it to server).
- **Rendering modes**:
  - **Terminal-based**: braille, half-blocks, ASCII, hybrid (auto-select per cell)
  - **Pixel-based**: sixels, kitty graphics, framebuffer (/dev/fb0), KMS direct rendering, **native Wayland window (GUI)**
- **Color modes**: 16, 256, truecolor (24‑bit).
- **Optional CUDA acceleration** for server-side Unicode rendering (NVIDIA only).
- **Hardware-accelerated H.265/HEVC encoding** for pixel-based renderers with automatic hardware detection (NVENC, QuickSync, VAAPI, AMF, VideoToolbox).
- **Native GUI mode**: Client creates a Wayland window for a native application experience (no terminal required).

## Demonstration
https://github.com/user-attachments/assets/089d1c60-c502-4422-892c-fb83e392149a

## How it works (short)
- **Screen capture**: wlr-screencopy/PipeWire → frame buffer → renderer (CPU/CUDA)
- **Unicode renderers** (braille/blocks/ascii/hybrid): ANSI escape sequences → LZ4 → TCP → client terminal
- **Pixel renderers** (sixel/kitty/framebuffer/KMS/GUI): RGB24 → H.265 encoder (hardware-accelerated) → TCP → client decoder → display
- **Input**: Client libinput capture → TCP → server → virtual pointer/keyboard (Wayland) on host
- **Audio**: System audio (server→client) and microphone (client→server) via PipeWire with optional Opus compression
- **Hardware encoder priority**: NVENC (NVIDIA) → QuickSync (Intel) → VAAPI (Intel/AMD) → AMF (AMD) → VideoToolbox (macOS) → software fallback (x265)
- **Hardware decoder priority**: CUDA (NVIDIA) → VAAPI (Intel/AMD) → QuickSync (Intel) → VDPAU (NVIDIA Legacy) → VideoToolbox (macOS) → multi-threaded software fallback

## Quickstart
1. **Build** (CPU-only)
   ```bash
   git clone https://github.com/cyber-wojtek/waytermirror.git
   cd waytermirror
   make -j$(nproc)
   ```
   This produces `waytermirror_server` and `waytermirror_client`.

2. **Run the server** on your Wayland desktop:
   ```bash
   ./waytermirror_server
   ```

3. **Run the client**:
   - **Terminal mode** (default - works anywhere):
     ```bash
     ./waytermirror_client -H <host>
     ```
   
   - **Native GUI window** (best experience):
     ```bash
     ./waytermirror_client -H <host> -R gui
     ```
   
   - **Framebuffer mode** (TTY/console - requires root):
     ```bash
     sudo ./waytermirror_client -H <host> -R framebuffer
     ```
   
   - **KMS direct rendering** (TTY/console - requires root/DRM permissions):
     ```bash
     sudo ./waytermirror_client -H <host> -R kms
     ```

**Tip**: Run `./waytermirror_client --help` or `./waytermirror_server --help` to see all available flags and defaults.

## Build & install

### Prerequisites
- **Core**: [gcc/g++](https://github.com/gcc-mirror/gcc.git), [make](https://git.savannah.gnu.org/git/make.git), [wayland](https://gitlab.com/freedesktop-sdk/mirrors/freedesktop/wayland/wayland.git), [wayland-protocols](https://gitlab.freedesktop.org/wayland/wayland-protocols.git), [libinput](https://gitlab.freedesktop.org/libinput/libinput.git), [pipewire](https://gitlab.freedesktop.org/pipewire/pipewire.git), [lz4](https://github.com/lz4/lz4.git), [rapidjson](https://github.com/Tencent/rapidjson.git), [systemd](https://github.com/systemd/systemd.git), [argparse](https://github.com/p-ranav/argparse.git), [libsixel](https://github.com/saitoha/libsixel), [libpng](http://www.libpng.org/pub/png/libpng.html), [opus](https://opus-codec.org/), [libudev](https://www.freedesktop.org/software/systemd/man/libudev.html), [libdrm](https://gitlab.freedesktop.org/mesa/drm), [gbm](https://gitlab.freedesktop.org/mesa/mesa)
- **Video encoding**: [FFmpeg](https://ffmpeg.org/) (libavcodec, libavutil, libswscale, libavformat) for H.265/HEVC encoding/decoding with hardware acceleration support
- **Optional**: NVIDIA CUDA toolkit for GPU rendering (nvcc) — see [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

### Arch Linux-based
```bash
sudo pacman -S base-devel git gcc wayland wayland-protocols libinput pipewire lz4 rapidjson systemd argparse libsixel libpng ffmpeg opus libudev0 libdrm mesa
# Optional for CUDA:
sudo pacman -S cuda
```

### Debian-based
```bash
sudo apt install build-essential git gcc libwayland-dev wayland-protocols libinput-dev libpipewire-0.3-dev liblz4-dev rapidjson-dev libsystemd-dev pkg-config cmake libargparse-dev libsixel-dev libpng-dev libavcodec-dev libavutil-dev libswscale-dev libavformat-dev libopus-dev libudev-dev libdrm-dev libgbm-dev
# For CUDA: install NVIDIA CUDA toolkit from https://developer.nvidia.com/cuda-toolkit
```

### Building
- **CPU-only** (default):
  ```bash
  make -j$(nproc)
  ```
- **CUDA-enabled** (if you have nvcc and CUDA libs):
  ```bash
  make CUDA=true -j$(nproc)
  ```
- The provided PKGBUILD auto-detects nvcc. Override with `WAYTERMIRROR_CUDA=1` or `WAYTERMIRROR_NO_CUDA`.

### Artifacts
- `waytermirror_server`
- `waytermirror_client`

### Packaging (Arch)
- **AUR**: Install directly from the [waytermirror-git](https://aur.archlinux.org/packages/waytermirror-git) AUR package:
  ```bash
  yay -S waytermirror-git
  # or
  paru -S waytermirror-git
  ```

- **Manual build**: Use the included PKGBUILD with `makepkg`:
  ```bash
  mkdir -p waytermirror
  cd waytermirror
  wget https://github.com/cyber-wojtek/waytermirror/raw/refs/heads/main/PKGBUILD
  makepkg -si
  ```

## Runtime requirements
- **Audio**: PipeWire (for system audio streaming and microphone capture).
- **Input**: Access to input devices (user must be in the `input` group or run with sufficient privileges to read `/dev/input/*`).
- **GUI mode**: Running Wayland compositor with xdg-shell support.
- **Framebuffer/KMS modes**: Root access or proper permissions for `/dev/fb0` and `/dev/dri/card*`.
- **Hardware video encoding** (server): Supported encoders auto-detected at runtime:
  - NVIDIA: NVENC (most performant, lowest latency)
  - Intel: QuickSync Video or VAAPI
  - AMD: AMF (Windows/Linux) or VAAPI (Linux)
  - Apple: VideoToolbox (macOS)
  - Software fallback: libx265 (very slow, not recommended for real-time)
- **Hardware video decoding** (client): Supported decoders auto-detected at runtime:
  - NVIDIA: CUDA
  - Intel/AMD: VAAPI or QuickSync
  - Software fallback: multi-threaded libavcodec

## Usage

### Server
```bash
./waytermirror_server [options]
```

#### Full server options
> For the authoritative list use `./waytermirror_server --help`.

| Flag / Short | Long / Name                                              | Description                                             | Default |
| ------------ | -------------------------------------------------------- | ------------------------------------------------------- | ------- |
| -P <n>       | --port <n>                                               | Base TCP port (video base; other services use base+N)   | 9999    |
| -F <n>       | --capture-fps <n>                                        | Capture framerate from compositor                       | 30      |
| -R <res>     | --capture-resolution <res>                               | Capture resolution: auto or WxH (e.g., 1920x1080)       | auto    |
| -C <type>    | --compositor <auto\|hyprland\|sway\|kde\|gnome\|generic> | Compositor override                                     | auto    |
| -B <backend> | --capture-backend <auto\|wlr\|pipewire>                  | Screen capture backend                                  | auto    |
| -I <backend> | --input-backend <auto\|virtual\|uinput>                  | Input injection backend                                 | auto    |
| -n           | --no-video                                               | Disable screen capture / video streaming                | off     |
| -A           | --no-audio                                               | Disable system audio streaming                          | off     |
| -N           | --no-input                                               | Disable input injection (do not create virtual devices) | off     |
| -m           | --no-microphone                                          | Disable microphone reception (client→server mic)        | off     |

**Capture backend notes:**
- `wlr`: Uses wlr-screencopy protocol directly (compositor must support wlr-screencopy-unstable-v1)
- `pipewire`: Uses PipeWire + xdg-desktop-portal for screen capture (works on more compositors including GNOME/KDE)
  - **Important**: When using PipeWire backend, select screens in their **logical index order** (0, then 1, then 2, etc.) to match the output indices used by the client's `-o` option.
- `auto`: Automatically detects and prefers wlr-screencopy if available, falls back to PipeWire

**Input backend notes:**
- `virtual`: Uses Wayland virtual input protocols (zwlr_virtual_pointer_v1, zwp_virtual_keyboard_v1) — requires compositor support
- `uinput`: Uses Linux uinput (`/dev/uinput`) — works on any compositor but requires proper permissions
- `auto`: Automatically selects virtual protocols if available, falls back to uinput

### Client
```bash
./waytermirror_client -H <server_ip> [options]
```

#### Full client options
> For the authoritative list use `./waytermirror_client --help`.

**Connection**
| Flag / Short | Long / Name   | Description                   | Default    |
| ------------ | ------------- | ----------------------------- | ---------- |
| -H <addr>    | --host <addr> | Server IP/hostname (required) | *required* |
| -P <n>       | --port <n>    | Server base port              | 9999       |

**Video & rendering**
| Flag / Short       | Long / Name                                                                      | Description                                          | Default |
| ------------------ | -------------------------------------------------------------------------------- | ---------------------------------------------------- | ------- |
| -o <n\|follow>     | --output <n\|follow>                                                             | Output index or `follow` to track focused window     | 0       |
| -F <n>             | --fps <n>                                                                        | Target client FPS / playback framerate               | 30      |
| -M <16\|256\|true> | --mode <16\|256\|true>                                                           | Color mode (16, 256, truecolor)                      | 256     |
| -R <type>          | --renderer <braille\|blocks\|ascii\|sixel\|kitty\|framebuffer\|kms\|gui\|hybrid> | Rendering method                                     | braille |
| -K <res>           | --receive-resolution <res>                                                       | Client-side decode resolution: native or WxH         | native  |
| -r <cpu\|cuda>     | --render-device <cpu\|cuda>                                                      | Prefer server-side renderer (for Unicode modes)      | cpu     |
| -d <0-100>         | --detail-level <0-100>                                                           | Visual detail (0: fast/smooth, 100: sharp)           | 50      |
| -Q <0-100>         | --quality <0-100>                                                                | H.265 encoding quality (0: fast/low, 100: slow/best) | 50      |
| -S <factor>        | --scale <factor>                                                                 | Scale factor for rendered output                     | 1.0     |
| -k                 | --keep-aspect-ratio                                                              | Maintain aspect ratio when scaling                   | off     |
| -c                 | --compress                                                                       | Enable LZ4 compression (for Unicode renderers)       | off     |
| -L <0-12>          | --compression-level <0-12>                                                       | LZ4 HC level (0=fast, 12=best)                       | 0       |
| -T <degrees>       | --rotation <degrees>                                                             | Rotation angle (0-360 degrees)                       | 0       |
| -n                 | --no-video                                                                       | Disable video display                                | off     |

**Renderer descriptions:**
- **braille**: Unicode braille patterns (2×4 pixels per cell, works in any terminal)
- **blocks**: Half-block characters (1×2 pixels per cell, better color support)
- **ascii**: ASCII art style rendering (1×1 pixel per cell)
- **hybrid**: Automatically selects braille or blocks per cell for optimal quality
- **sixel**: Sixel graphics protocol (pixel-perfect, requires sixel-compatible terminal)
- **kitty**: Kitty graphics protocol (pixel-perfect, requires kitty or compatible terminal)
- **framebuffer**: Direct Linux framebuffer (`/dev/fb0`) rendering (requires root, works on TTY)
- **kms**: Direct KMS/DRM rendering (requires root/DRM permissions, best TTY performance)
- **gui**: Native Wayland window (no terminal needed, resizable, best user experience)

**Quality setting notes:**
- For **pixel renderers** (sixel/kitty/framebuffer/kms/gui): Controls H.265 encoding quality
  - 0-40: Fast encoding, lower bitrate, acceptable quality
  - 50-70: Balanced quality/performance (recommended)
  - 80-100: Maximum quality, slower encoding, higher bitrate
- For **Unicode renderers** (braille/blocks/ascii/hybrid): Controls pattern search precision

**Input (local client input capture / forwarding)**
| Flag / Short | Long / Name       | Description                                  | Default |
| ------------ | ----------------- | -------------------------------------------- | ------- |
| -N           | --no-input        | Do not capture/forward local input           | off     |
| -x           | --exclusive-input | Grab input devices exclusively (EVIOCGRAB)   | off     |
| -C           | --center-mouse    | Start mouse at screen center when connecting | off     |

**Audio**
| Flag / Short | Long / Name                                      | Description                                | Default |
| ------------ | ------------------------------------------------ | ------------------------------------------ | ------- |
| -A           | --no-audio                                       | Disable audio playback (server→client)     | off     |
| -a           | --audio-compress                                 | Enable audio Opus compression              | off     |
| -u <Hz>      | --audio-sample-rate <Hz>                         | Audio opus sample rate                     | 48000   |
| -v <n>       | --audio-channels <n>                             | Audio opus channels                        | 2       |
| -b <kbps>    | --audio-bitrate <kbps>                           | Audio opus bitrate                         | 64      |
| -y <0-10>    | --audio-complexity <0-10>                        | Opus audio complexity                      | 5       |
| -w <type>    | --audio-application <voip\|audio\|lowdelay>      | Opus audio application mode                | audio   |
| -p           | --no-microphone                                  | Disable microphone capture (client→server) | off     |
| -m           | --microphone-compress                            | Enable microphone Opus compression         | off     |
| -U <Hz>      | --microphone-sample-rate <Hz>                    | Microphone opus sample rate                | 48000   |
| -V <n>       | --microphone-channels <n>                        | Microphone opus channels                   | 2       |
| -B <kbps>    | --microphone-bitrate <kbps>                      | Microphone opus bitrate                    | 64      |
| -Y <0-10>    | --microphone-complexity <0-10>                   | Microphone opus complexity                 | 5       |
| -W <type>    | --microphone-application <voip\|audio\|lowdelay> | Microphone opus application mode           | voip    |

**Zoom / viewport**
| Flag / Short | Long / Name         | Description                 | Default |
| ------------ | ------------------- | --------------------------- | ------- |
| -z           | --zoom              | Start with zoom enabled     | off     |
| -Z <1-10>    | --zoom-level <1-10> | Magnification               | 2.0     |
| -X <px>      | --zoom-width <px>   | Viewport width (px)         | 800     |
| -Y <px>      | --zoom-height <px>  | Viewport height (px)        | 600     |
| -f           | --zoom-follow       | Follow mouse while zoomed   | on      |
| -s           | --zoom-smooth       | Smooth panning while zoomed | on      |
| -D <n>       | --zoom-speed <n>    | Pan speed (px/frame)        | 20      |

## Examples

**Basic LAN streaming (terminal)**
```bash
# Server (desktop)
./waytermirror_server -F 60

# Client (terminal)
./waytermirror_client -H 192.168.1.100 -F 60 -M true -R hybrid
```

**Native GUI window (best experience for graphical applications)**
```bash
./waytermirror_client -H 192.168.1.100 -R gui -Q 75
# Creates a native Wayland window, resizable, hardware-accelerated
# No terminal required, works like a native application
```

**High quality sixel with hardware encoding**
```bash
./waytermirror_client -H 192.168.1.100 -R sixel -Q 85 -M true
# Server auto-detects best hardware encoder (NVENC/QuickSync/VAAPI/AMF)
```

**Framebuffer output (TTY/console)**
```bash
sudo ./waytermirror_client -H 192.168.1.100 -R framebuffer -Q 70
# Mirrors to /dev/fb0, hardware-accelerated H.265 encoding
```

**KMS direct rendering (best TTY performance)**
```bash
sudo ./waytermirror_client -H 192.168.1.100 -R kms -Q 80
# Direct KMS/DRM rendering with hardware acceleration
```

**Low bandwidth (Unicode with compression)**
```bash
./waytermirror_client -H server.example.com -c -L 12 -F 15 -M 256 -d 30 -R braille
```

**Follow focused window**
```bash
./waytermirror_client -H 192.168.1.100 -o follow
```

**Input-only (control remote desktop without video)**
```bash
./waytermirror_client -H 192.168.1.100 -n -x
```

**4K GUI window with audio**
```bash
./waytermirror_client -H 192.168.1.100 -R gui -K 3840x2160 -Q 85 -a -b 128
```

**Zoom mode with GUI**
```bash
./waytermirror_client -H 192.168.1.100 -R gui -z -Z 3.0 -X 1920 -Y 1080
```

## Keyboard shortcuts (client)

All client shortcuts use the **Ctrl+Alt+Shift** modifier prefix, so normal keys are forwarded to the remote session. Press **Ctrl+Alt+Shift+H** at any time to display the full shortcut list with current toggle states.

### Session control
| Shortcut         | Action               | Notes                                                   |
| ---------------- | -------------------- | ------------------------------------------------------- |
| Ctrl+Alt+Shift+Q | Quit / disconnect    | Graceful disconnect (sends close to server)             |
| Ctrl+Alt+Shift+H | Toggle help          | Display all shortcuts and current state                 |
| Ctrl+Alt+Shift+P | Pause / resume video | Stops rendering updates locally (input still forwarded) |

### Input control
| Shortcut         | Action                  | Notes                                                   |
| ---------------- | ----------------------- | ------------------------------------------------------- |
| Ctrl+Alt+Shift+I | Toggle input forwarding | Enable/disable forwarding of keyboard & mouse to server |
| Ctrl+Alt+Shift+G | Toggle exclusive grab   | EVIOCGRAB on local devices (when supported)             |

### Zoom control
| Shortcut                       | Action              | Notes                                          |
| ------------------------------ | ------------------- | ---------------------------------------------- |
| Ctrl+Alt+Shift+Z               | Toggle zoom mode    | When zoomed, use arrow keys to pan             |
| Ctrl+Alt+Shift++ (or =)        | Zoom in             | Increases zoom level by 0.125×                   |
| Ctrl+Alt+Shift+-               | Zoom out            | Decreases zoom level by 0.125×                   |
| Ctrl+Alt+Shift+0               | Reset zoom          | Reset to 2.0× and center viewport              |
| Ctrl+Alt+Shift+N               | Toggle zoom follow  | Enable/disable zoom following mouse cursor     |
| Ctrl+Alt+Shift+Arrow keys      | Pan viewport        | Left/Right/Up/Down — uses configured pan speed |
| Ctrl+Alt+Shift+PageUp/PageDown | Fast vertical pan   | 5× normal pan speed                            |
| Ctrl+Alt+Shift+Home/End        | Fast horizontal pan | 5× normal pan speed                            |

### Rotation
| Shortcut         | Action          | Notes                                |
| ---------------- | --------------- | ------------------------------------ |
| Ctrl+Alt+Shift+[ | Rotate left 5°  | Counter-clockwise rotation           |
| Ctrl+Alt+Shift+] | Rotate right 5° | Clockwise rotation                   |
| Ctrl+Alt+Shift+\ | Reset rotation  | Return to 0°                         |
| Ctrl+Alt+Shift+T | Rotate 90° CW   | Quick 90° clockwise rotation         |
| Ctrl+Alt+Shift+Y | Rotate 90° CCW  | Quick 90° counter-clockwise rotation |

### Rendering
| Shortcut         | Action                  | Notes                                                                       |
| ---------------- | ----------------------- | --------------------------------------------------------------------------- |
| Ctrl+Alt+Shift+R | Cycle renderer          | braille → blocks → ascii → hybrid → sixel → kitty → framebuffer → kms → gui |
| Ctrl+Alt+Shift+C | Cycle color mode        | 16 → 256 → truecolor                                                        |
| Ctrl+Alt+Shift+D | Increase detail         | +10 detail level (Unicode) or quality (H.265)                               |
| Ctrl+Alt+Shift+S | Decrease detail         | −10 detail level (Unicode) or quality (H.265)                               |
| Ctrl+Alt+Shift+W | Increase quality        | +10 quality level                                                           |
| Ctrl+Alt+Shift+E | Decrease quality        | −10 quality level                                                           |
| Ctrl+Alt+Shift+O | Toggle smooth panning   | Enable/disable smooth zoom panning                                          |
| Ctrl+Alt+Shift+B | Toggle aspect ratio     | Keep/ignore aspect ratio when scaling                                       |
| Ctrl+Alt+Shift+V | Cycle render device     | CPU → CUDA (for Unicode modes)                                              |
| Ctrl+Alt+Shift+U | Toggle compression      | Enable/disable LZ4 compression (Unicode modes)                              |
| Ctrl+Alt+Shift+L | Cycle compression level | Off → fast LZ4 → HC levels (Unicode modes)                                  |

### Output & FPS
| Shortcut         | Action              | Notes                              |
| ---------------- | ------------------- | ---------------------------------- |
| Ctrl+Alt+Shift+` | Cycle output        | Next output or toggle follow-focus |
| Ctrl+Alt+Shift+J | Increase FPS        | +5 FPS                             |
| Ctrl+Alt+Shift+K | Decrease FPS        | −5 FPS (min 0)                     |
| Ctrl+Alt+Shift+F | Toggle focus-follow | Follow focused output/window       |

### Audio
| Shortcut         | Action                       | Notes                             |
| ---------------- | ---------------------------- | --------------------------------- |
| Ctrl+Alt+Shift+A | Toggle audio                 | Mute/unmute system audio playback |
| Ctrl+Alt+Shift+M | Toggle microphone            | Mute/unmute microphone capture    |
| Ctrl+Alt+Shift+5 | Cycle audio compression      | Off → Opus                        |
| Ctrl+Alt+Shift+6 | Cycle microphone compression | Off → Opus                        |

### Quick usage tips
- **GUI mode**: Ctrl+Alt+Shift+R cycles to GUI renderer, creating a native resizable Wayland window
- **Zoom panning**: When zoomed (Ctrl+Alt+Shift+Z), use arrow keys to pan the viewport
- **Rotation**: Use **[** and **]** to rotate in 5° steps, **T**/**Y** for 90° jumps, **\\** to reset
- **Renderer cycling**: Press **R** repeatedly to cycle through all 9 renderers
- **Pixel modes** (sixel/kitty/framebuffer/kms/gui): Use **hardware-accelerated H.265 encoding** automatically
- **FPS adjustment**: Use **J** to increase and **K** to decrease FPS by 5
- **Audio compression**: Press **5** and **6** to toggle audio and microphone Opus compression

## Network ports
- **(base port)** — TCP — video frames
- **(base port + 1)** — TCP — input events
- **(base port + 2)** — TCP — system audio (server → client)
- **(base port + 3)** — TCP — configuration/control
- **(base port + 4)** — TCP — microphone (client → server)

Default base port is **9999** (see `-P` / `--port`).

## Performance tuning

### Hardware-accelerated pixel rendering (best visual quality)

**Native GUI window (recommended)**
```bash
-R gui -Q 70
```
- Creates native Wayland window (no terminal needed)
- Hardware-accelerated H.265 encoding/decoding
- Resizable, works like any native application
- Best user experience for desktop mirroring

**Sixel mode**
```bash
-R sixel -Q 70
```
- Pixel-perfect graphics in sixel-compatible terminals
- Hardware H.265 encoding

**Kitty mode**
```bash
-R kitty -Q 70
```
- Kitty graphics protocol with H.265 compression
- Requires Kitty or compatible terminal

**Framebuffer mode**
```bash
sudo -R framebuffer -Q 80
```
- Direct framebuffer rendering
- Works on physical TTYs
- Requires root access

**KMS mode** (best TTY performance)
```bash
sudo -R kms -Q 85
```
- Direct KMS/DRM rendering
- Zero-copy rendering path
- Best performance on TTY
- Requires root/DRM permissions

**Quality ranges:**
- 50-70: Balanced performance (recommended)
- 80-100: Maximum quality (higher bitrate)

**Hardware encoder auto-detection priority:**
1. NVENC (NVIDIA) — lowest latency, best performance
2. QuickSync (Intel) — excellent performance
3. VAAPI (Intel/AMD) — good performance
4. AMF (AMD) — good performance
5. Software (x265) — fallback, very slow (not recommended)

### Unicode rendering (terminal-based)

**Maximum quality**
```bash
-r cuda -R braille -d 100 -Q 100 -M true
```

**Smooth video**
```bash
-R hybrid -d 30 -F 60
```

**Low bandwidth**
```bash
-c -L 9 -M 256 -F 15 -d 30
```

**Low latency**
```bash
-F 60 -Q 0 -d 50
```

### Encoder-specific tuning
- **NVENC (NVIDIA)**: Ultra-low latency, CBR mode, no B-frames
- **QuickSync (Intel)**: Low delay mode, minimal look-ahead
- **VAAPI (Intel/AMD)**: Quality-based encoding
- **Software fallback**: Not recommended for real-time (install hardware drivers)

## Troubleshooting

**"Failed to initialize libinput"**
```bash
sudo usermod -aG input $USER
# Log out and back in
```

**PipeWire issues (audio not working)**
```bash
systemctl --user status pipewire
systemctl --user restart pipewire
```

**CUDA errors / GPU verification**
```bash
nvcc --version
nvidia-smi
```

**H.265 encoding issues**
- Check server logs for encoder selection messages
- **NVENC not available**: Ensure NVIDIA drivers are installed
- **QuickSync not available**: Install Intel media drivers (`intel-media-driver` on Arch)
- **Software encoder very slow**: This is expected — install hardware acceleration

**Sixel/Kitty/GUI rendering artifacts**
- Increase quality: `-Q 80` or higher
- Check encoder logs for errors
- Try different renderer: `-R gui` for native window

**GUI window not responding**
- Ensure Wayland compositor is running
- Check that xdg-shell protocol is available
- Look for "GUI" messages in client logs
- Try another renderer: `-R sixel` or `-R kms`

**GUI window won't resize**
- This is a known issue being fixed
- Restart client to reset window size
- Use `-K WxH` to set specific resolution

**Permissions to /dev/input**
```bash
# Check group membership
groups $USER

# Add to input group if missing
sudo usermod -aG input $USER
```

**Framebuffer/KMS permission denied**
```bash
# Framebuffer
sudo chmod 666 /dev/fb0  # Temporary
# Or run with sudo

# KMS/DRM
sudo usermod -aG video $USER  # Permanent
# Log out and back in
```

**Logs & debugging**
- Server prints encoder selection: Look for "HEVC ENCODER" in logs
- Client prints decoder selection: Look for "HEVC DECODER" in logs
- Compositor detection: Check server startup messages
- Network issues: Verify firewall allows ports 9999-10003

## Security & limitations

**Security considerations**
- **No built-in authentication**: Uses raw TCP streams
- **Do not expose to untrusted networks**: Use SSH tunnel or VPN for remote access
- **Example SSH tunnel**:
  ```bash
  ssh -L 9999:localhost:9999 -L 10000:localhost:10000 -L 10001:localhost:10001 \
      -L 10002:localhost:10002 -L 10003:localhost:10003 user@server
  # Then connect client to localhost
  ```

**Limitations**
- Input injection requires elevated privileges or `input` group membership
- Framebuffer/KMS modes require root access or proper permissions
- H.265 hardware encoding: Software fallback (libx265) is very slow
- GUI mode: Currently has some window resize handling issues (being fixed)
- PipeWire backend: Must manually select screens in correct order

**Recommended setup**
- Run on trusted local network or VPN
- Use hardware acceleration when possible
- Keep software up to date
- Monitor system resources during operation

## Design notes & behavior

### Rendering architecture

**Unicode modes** (braille/blocks/ascii/hybrid)
- Rendering performed server-side
- Client displays ANSI escape sequences
- Optional LZ4 compression
- CUDA acceleration available (server-side)

**Pixel modes** (sixel/kitty/framebuffer/kms/gui)
- Server: RGB24 → H.265 encoding (hardware-accelerated)
- Network: Compressed H.265 stream
- Client: H.265 decoding (hardware-accelerated) → display
- Much more efficient than LZ4 for pixel data

### Hardware acceleration

**Server (encoding)**
- Automatic encoder detection prioritizes best available
- NVENC > QuickSync > VAAPI > AMF > software
- Logs show selected encoder at startup

**Client (decoding)**
- Automatic decoder detection: CUDA > VAAPI > QuickSync > VDPAU > software
- Multi-threaded software fallback when hardware unavailable
- Sequence number tracking prevents frame drops

**Unicode rendering** (optional)
- CUDA acceleration for pattern search (server-side)
- Only when built with `CUDA=true`
- Significantly faster braille/blocks rendering

### Compression

**Unicode modes**
- Optional LZ4 compression of ANSI strings
- Levels 0-12 (0=fast LZ4, 1-12=HC)
- Toggle with Ctrl+Alt+Shift+U

**Pixel modes**
- H.265 video compression (much more efficient)
- Hardware-accelerated for real-time performance
- Quality controlled by `-Q` flag

**Audio**
- Optional Opus compression (both directions)
- Configurable bitrate/complexity
- Toggle with Ctrl+Alt+Shift+5 (playback) or 6 (microphone)

### Hybrid renderer
- Chooses per-cell between braille and half-blocks
- Adaptive Unicode rendering for best quality
- CPU or CUDA accelerated

### GUI renderer
- Native Wayland window using xdg-shell protocol
- Hardware-accelerated H.265 decoding
- Resizable window (some issues being fixed)
- Best user experience for desktop mirroring
- No terminal emulator required

## Contributing
- Bug reports, feature requests, and PRs welcome
- Join discussions in Issues

## License
MIT License — see [LICENSE](LICENSE) file for details.

---

**Project Status**: Active development