# Waytermirror

Real-time Wayland screen mirroring to a terminal using Unicode braille characters, half‑blocks, ASCII, sixels, kitty graphics, framebuffer, or hybrid rendering. Includes bidirectional input forwarding, audio streaming (PipeWire), zooming, focus-follow, and optional NVIDIA CUDA acceleration (server-side).

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
  - [Runtime requirements](#runtime-requirements)
  - [Usage](#usage)
    - [Full server options](#full-server-options)
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
  - [Network ports](#network-ports)
  - [Performance tuning](#performance-tuning)
  - [Troubleshooting](#troubleshooting)
  - [Security \& limitations](#security--limitations)
  - [Design notes \& behavior](#design-notes--behavior)
  - [Contributing](#contributing)
  - [License](#license)

## What it is
- A client/server application:
  - Server runs on the Wayland host (captures screen & audio, performs rendering, injects input).
  - Client runs in a terminal (receives ANSI/Sixel frames, displays them, captures local input, sends it to server).
- Rendering modes: braille, half-blocks, ASCII, sixels, kitty graphics, framebuffer (/dev/fb0), KMS direct rendering, and hybrid (auto-select per cell). Framebuffer mode allows direct writing to the Linux framebuffer device, enabling output to physical TTYs or virtual consoles without a terminal emulator.
- Color modes: 16, 256, truecolor (24‑bit).
- Optional CUDA acceleration for server-side rendering (NVIDIA only).
- Sixel graphics support for direct pixel rendering on compatible terminals.
- **Hardware-accelerated H.265/HEVC encoding** for pixel-based renderers (sixel, kitty, framebuffer, KMS) with automatic hardware detection (NVENC, QuickSync, VAAPI, AMF, VideoToolbox).

## Demonstration
https://github.com/user-attachments/assets/089d1c60-c502-4422-892c-fb83e392149a

## How it works (short)
- Screen → wlr-screencopy/PipeWire → frame buffer → renderer (CPU/CUDA) → ANSI/Sixel/H.265 → LZ4 → TCP → client terminal
- For pixel renderers (sixel/kitty/framebuffer/KMS): RGB24 → H.265 encoder (hardware-accelerated) → client decoder → display
- Input capture on client → forwarded to server → virtual pointer/keyboard (Wayland) on host
- Audio (system → client) and microphone (client → server) via PipeWire with optional Opus compression
- Automatic hardware encoder detection prioritizes: NVENC (NVIDIA) → QuickSync (Intel) → VAAPI (Intel/AMD) → AMF (AMD) → VideoToolbox (macOS) → software fallback (x265)

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
   - To use framebuffer mode (e.g., on a TTY/console), run the client as root or with permissions to access /dev/fb0, and specify the framebuffer renderer:
     ```bash
     sudo ./waytermirror_client -H <host> -R framebuffer
     ```
   - This will write frames directly to the framebuffer device, bypassing terminal emulators. Useful for full-screen mirroring on physical consoles.
   ```bash
   ./waytermirror_client -H <host>
   ```

Tip: run `./waytermirror_client --help` or `./waytermirror_server --help` to see current/compiled-in flags and defaults on your build.

## Build & install

Prerequisites
- Core: [gcc/g++](https://github.com/gcc-mirror/gcc.git), [make](https://git.savannah.gnu.org/git/make.git), [wayland](https://gitlab.com/freedesktop-sdk/mirrors/freedesktop/wayland/wayland.git), [wayland-protocols](https://gitlab.freedesktop.org/wayland/wayland-protocols.git), [libinput](https://gitlab.freedesktop.org/libinput/libinput.git), [pipewire](https://gitlab.freedesktop.org/pipewire/pipewire.git), [lz4](https://github.com/lz4/lz4.git), [rapidjson](https://github.com/Tencent/rapidjson.git), [systemd](https://github.com/systemd/systemd.git), [argparse](https://github.com/p-ranav/argparse.git), [libsixel](https://github.com/saitoha/libsixel), [libpng](http://www.libpng.org/pub/png/libpng.html)
- **Video encoding**: [FFmpeg](https://ffmpeg.org/) (libavcodec, libavutil, libswscale) for H.265/HEVC encoding with hardware acceleration support
- Optional: NVIDIA CUDA toolkit for GPU rendering (nvcc) – see NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit

Arch Linux-based
```bash
sudo pacman -S base-devel git gcc wayland wayland-protocols libinput pipewire lz4 rapidjson systemd argparse libsixel libpng ffmpeg
# Optional for CUDA:
sudo pacman -S cuda
```

Debian-based
```bash
sudo apt install build-essential git gcc libwayland-dev wayland-protocols libinput-dev libpipewire-0.3-dev liblz4-dev rapidjson-dev libsystemd-dev pkg-config cmake libargparse-dev libsixel-dev libpng-dev libavcodec-dev libavutil-dev libswscale-dev libavformat-dev
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
  mkdir -p waytermirror
  cd waytermirror
  wget https://github.com/cyber-wojtek/waytermirror/raw/refs/heads/main/PKGBUILD
  makepkg -si
  ```

## Runtime requirements
- Audio: PipeWire (for system audio streaming).
- Input: access to input devices (user must be in the `input` group or run with sufficient privileges to read /dev/input/*).
- **Hardware video encoding**: Supported hardware encoders will be auto-detected at runtime:
  - NVIDIA: NVENC (most performant, lowest latency)
  - Intel: QuickSync Video or VAAPI
  - AMD: AMF (Windows/Linux) or VAAPI (Linux)
  - Apple: VideoToolbox (macOS)
  - Software fallback: libx265 (very slow, not recommended for real-time)

## Usage

Server
```bash
./waytermirror_server [options]
```

### Full server options
> For the authoritative list use `./waytermirror_server --help`.

| Flag / Short | Long / Name                                              | Description                                             | Default |
| -----------: | -------------------------------------------------------- | ------------------------------------------------------- | ------- |
|       -P <n> | --port <n>                                               | Base TCP port (video base; other services use base+N)   | 9999    |
|       -F <n> | --capture-fps <n>                                        | Capture framerate from compositor                       | 30      |
|     -R <res> | --capture-resolution <res>                               | Capture resolution: auto or WxH (e.g., 1920x1080)       | auto    |
|    -C <type> | --compositor <auto\|hyprland\|sway\|kde\|gnome\|generic> | Compositor override                                     | auto    |
| -B <backend> | --capture-backend <auto\|wlr\|pipewire>                  | Screen capture backend (see notes below)                | auto    |
| -I <backend> | --input-backend <auto\|virtual\|uinput>                  | Input injection backend                                 | auto    |
|           -n | --no-video                                               | Disable screen capture / video streaming                | off     |
|           -A | --no-audio                                               | Disable system audio streaming                          | off     |
|           -N | --no-input                                               | Disable input injection (do not create virtual devices) | off     |
|           -m | --no-microphone                                          | Disable microphone reception (client→server mic)        | off     |

**Capture backend notes:**
- `wlr`: Uses wlr-screencopy protocol directly (compositor must support wlr-screencopy-unstable-v1)
- `pipewire`: Uses PipeWire + xdg-desktop-portal for screen capture (works on more compositors including GNOME/KDE)
  - **Important**: When using PipeWire backend, you'll be prompted to select screens. Select them in their **logical index order** (0, then 1, then 2, etc.) to match the output indices used by the client's `-o` option.
- `auto`: Automatically detects and prefers wlr-screencopy if available, falls back to PipeWire

**Input backend notes:**
- `virtual`: Uses Wayland virtual input protocols (zwlr_virtual_pointer_v1, zwp_virtual_keyboard_v1) – requires compositor support
- `uinput`: Uses Linux uinput (/dev/uinput) – works on any compositor but requires proper permissions
- `auto`: Automatically selects virtual protocols if available, falls back to uinput

Client
```bash
./waytermirror_client -H <server_ip> [options]
```

### Full client options
> For the authoritative list use `./waytermirror_client --help`.

Connection
| Flag / Short | Long / Name   | Description                   | Default    |
| -----------: | ------------- | ----------------------------- | ---------- |
|    -H <addr> | --host <addr> | Server IP/hostname (required) | *required* |
|       -P <n> | --port <n>    | Server base port              | 9999       |

Video & rendering
|       Flag / Short | Long / Name                                                                       | Description                                          | Default |
| -----------------: | --------------------------------------------------------------------------------- | ---------------------------------------------------- | ------- |
|     -o <n\|follow> | --output <n\|follow>                                                              | Output index or `follow` to track focused window     | 0       |
|             -F <n> | --fps <n>                                                                         | Target client FPS / playback framerate               | 30      |
| -M <16\|256\|true> | --mode <16\|256\|true>                                                            | Color mode (16, 256, truecolor)                      | 256     |
|          -R <type> | --renderer <braille\|blocks\|ascii\|sixel\|kitty\|framebuffer\|kms\|hybrid>       | Rendering method                                     | braille |
|           -K <res> | --receive-resolution <res>                                                        | Client-side decode resolution: native or WxH         | native  |
|        framebuffer | Directly writes to /dev/fb0 (Linux framebuffer). Requires root or fb permissions. |
|                kms | Direct KMS/DRM rendering (auto-detects display, requires DRM permissions)         |
|     -r <cpu\|cuda> | --render-device <cpu\|cuda>                                                       | Prefer server-side renderer (for Unicode modes)      | cpu     |
|         -d <0-100> | --detail-level <0-100>                                                            | Visual detail (0: fast/smooth, 100: sharp)           | 50      |
|         -Q <0-100> | --quality <0-100>                                                                 | H.265 encoding quality (0: fast/low, 100: slow/best) | 50      |
|        -S <factor> | --scale <factor>                                                                  | Scale factor for rendered output                     | 1.0     |
|                 -k | --keep-aspect-ratio                                                               | Maintain aspect ratio when scaling                   | off     |
|                 -c | --compress                                                                        | Enable LZ4 compression (for Unicode renderers)       | off     |
|          -L <0-12> | --compression-level <0-12>                                                        | LZ4 HC level (0=fast, 12=best)                       | 0       |
|                 -n | --no-video                                                                        | Disable video display                                | off     |

**Quality setting notes:**
- For **pixel renderers** (sixel/kitty/framebuffer/kms): Controls H.265 encoding quality
  - 0-40: Fast encoding, lower bitrate, acceptable quality
  - 50-70: Balanced quality/performance (recommended)
  - 80-100: Maximum quality, slower encoding, higher bitrate
- For **Unicode renderers** (braille/blocks/ascii/hybrid): Controls pattern search precision
- Hardware encoders automatically optimize encoding based on quality setting

Input (local client input capture / forwarding)
| Flag / Short | Long / Name       | Description                                  | Default |
| -----------: | ----------------- | -------------------------------------------- | ------- |
|           -N | --no-input        | Do not capture/forward local input           | off     |
|           -x | --exclusive-input | Grab input devices exclusively (EVIOCGRAB)   | off     |
|           -C | --center-mouse    | Start mouse at screen center when connecting | off     |

Audio
| Flag / Short | Long / Name     | Description                                | Default |
| -----------: | --------------- | ------------------------------------------ | ------- |
|           -A | --no-audio      | Disable audio playback (server→client)     | off     |
|           -p | --no-microphone | Disable microphone capture (client→server) | off     |

Zoom / viewport
| Flag / Short | Long / Name         | Description                 | Default |
| -----------: | ------------------- | --------------------------- | ------- |
|           -z | --zoom              | Start with zoom enabled     | off     |
|    -Z <1-10> | --zoom-level <1-10> | Magnification               | 2.0     |
|      -X <px> | --zoom-width <px>   | Viewport width (px)         | 800     |
|      -Y <px> | --zoom-height <px>  | Viewport height (px)        | 600     |
|           -f | --zoom-follow       | Follow mouse while zoomed   | on      |
|           -s | --zoom-smooth       | Smooth panning while zoomed | on      |
|       -D <n> | --zoom-speed <n>    | Pan speed (px/frame)        | 20      |

## Examples
- Basic LAN streaming:
  ```bash
  # Server (desktop)
  ./waytermirror_server -F 60

  # Client (terminal)
  ./waytermirror_client -H 192.168.1.100 -F 60 -M true -R hybrid
  ```

- High quality sixel with hardware encoding:
  ```bash
  ./waytermirror_client -H 192.168.1.100 -R sixel -Q 85 -M true
  # Server auto-detects best hardware encoder (NVENC/QuickSync/VAAPI/AMF)
  ```

- Framebuffer output (on a TTY/console):
  ```bash
  sudo ./waytermirror_client -H 192.168.1.100 -R framebuffer -Q 70
  ```
  This will mirror the remote screen directly to the local framebuffer device (/dev/fb0). Hardware-accelerated H.265 encoding ensures smooth performance.

- KMS direct rendering (best performance on TTY):
  ```bash
  sudo ./waytermirror_client -H 192.168.1.100 -R kms -Q 80
  ```
  Direct KMS/DRM rendering with hardware acceleration for maximum performance on physical displays.

- Low bandwidth with software fallback:
  ```bash
  ./waytermirror_client -H server.example.com -c -L 12 -F 15 -M 256 -d 30 -R braille
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
|         Shortcut | Action               | Notes                                                   |
| ---------------: | -------------------- | ------------------------------------------------------- |
| Ctrl+Alt+Shift+Q | Quit / disconnect    | Graceful disconnect (sends close to server)             |
| Ctrl+Alt+Shift+H | Toggle help          | Display all shortcuts and current state                 |
| Ctrl+Alt+Shift+P | Pause / resume video | Stops rendering updates locally (input still forwarded) |

### Input control
|         Shortcut | Action                  | Notes                                                   |
| ---------------: | ----------------------- | ------------------------------------------------------- |
| Ctrl+Alt+Shift+I | Toggle input forwarding | Enable/disable forwarding of keyboard & mouse to server |
| Ctrl+Alt+Shift+G | Toggle exclusive grab   | EVIOCGRAB on local devices (when supported)             |

### Zoom control
|                       Shortcut | Action              | Notes                                          |
| -----------------------------: | ------------------- | ---------------------------------------------- |
|               Ctrl+Alt+Shift+Z | Toggle zoom mode    | When zoomed, use arrow keys to pan             |
|        Ctrl+Alt+Shift++ (or =) | Zoom in             | Increases zoom level by 0.5x                   |
|               Ctrl+Alt+Shift+- | Zoom out            | Decreases zoom level by 0.5x                   |
|               Ctrl+Alt+Shift+0 | Reset zoom          | Reset to 2.0x and center viewport              |
|               Ctrl+Alt+Shift+N | Toggle zoom follow  | Enable/disable zoom following mouse cursor     |
|      Ctrl+Alt+Shift+Arrow keys | Pan viewport        | Left/Right/Up/Down – uses configured pan speed |
| Ctrl+Alt+Shift+PageUp/PageDown | Fast vertical pan   | 5× normal pan speed                            |
|        Ctrl+Alt+Shift+Home/End | Fast horizontal pan | 5× normal pan speed                            |

### Rotation
|         Shortcut | Action           | Notes                                |
| ---------------: | ---------------- | ------------------------------------ |
| Ctrl+Alt+Shift+[ | Rotate left 5°  | Counter-clockwise rotation           |
| Ctrl+Alt+Shift+] | Rotate right 5° | Clockwise rotation                   |
| Ctrl+Alt+Shift+\ | Reset rotation   | Return to 0°                         |
| Ctrl+Alt+Shift+T | Rotate 90° CW    | Quick 90° clockwise rotation         |
| Ctrl+Alt+Shift+Y | Rotate 90° CCW   | Quick 90° counter-clockwise rotation |

### Rendering
|         Shortcut | Action                  | Notes                                                                 |
| ---------------: | ----------------------- | --------------------------------------------------------------------- |
| Ctrl+Alt+Shift+R | Cycle renderer          | braille → blocks → ascii → hybrid → sixel → kitty → framebuffer → kms |
| Ctrl+Alt+Shift+C | Cycle color mode        | 16 → 256 → truecolor                                                  |
| Ctrl+Alt+Shift+D | Increase detail         | +10 detail level (Unicode) or quality (H.265)                         |
| Ctrl+Alt+Shift+S | Decrease detail         | −10 detail level (Unicode) or quality (H.265)                         |
| Ctrl+Alt+Shift+W | Increase quality        | +10 quality level                                                     |
| Ctrl+Alt+Shift+E | Decrease quality        | −10 quality level                                                     |
| Ctrl+Alt+Shift+O | Toggle smooth panning   | Enable/disable smooth zoom panning                                    |
| Ctrl+Alt+Shift+B | Toggle aspect ratio     | Keep/ignore aspect ratio when scaling                                 |
| Ctrl+Alt+Shift+V | Cycle render device     | CPU → CUDA (for Unicode modes)                                        |
| Ctrl+Alt+Shift+U | Toggle compression      | Enable/disable LZ4 compression (Unicode modes)                        |
| Ctrl+Alt+Shift+L | Cycle compression level | Off → fast LZ4 → HC levels (Unicode modes)                            |

### Output & FPS
|         Shortcut | Action              | Notes                              |
| ---------------: | ------------------- | ---------------------------------- |
| Ctrl+Alt+Shift+` | Cycle output        | Next output or toggle follow-focus |
| Ctrl+Alt+Shift+J | Increase FPS        | +5 FPS                             |
| Ctrl+Alt+Shift+K | Decrease FPS        | −5 FPS (min 0)                     |
| Ctrl+Alt+Shift+F | Toggle focus-follow | Follow focused output/window       |

### Audio
|         Shortcut | Action                       | Notes                             |
| ---------------: | ---------------------------- | --------------------------------- |
| Ctrl+Alt+Shift+A | Toggle audio                 | Mute/unmute system audio playback |
| Ctrl+Alt+Shift+M | Toggle microphone            | Mute/unmute microphone capture    |
| Ctrl+Alt+Shift+5 | Cycle audio compression      | Off → Opus                        |
| Ctrl+Alt+Shift+6 | Cycle microphone compression | Off → Opus                        |

Quick usage tips
- Zoom panning: when zoomed (Ctrl+Alt+Shift+Z), use arrow keys to pan the viewport. Hold PageUp/PageDown for faster vertical movement.
- Rotation: use **[** and **]** to rotate in 15° steps, **T**/**Y** for 90° jumps, **\\** to reset. Rotation is handled natively by CUDA when available.
- Renderer cycling: use **R** to cycle through all renderers in sequence.
- **Sixel/Kitty/Framebuffer/KMS rendering**: These modes use **hardware-accelerated H.265 encoding** automatically. The server detects available encoders (NVENC, QuickSync, VAAPI, AMF) and uses the best one. Quality setting controls encoding bitrate/quality tradeoff.
- FPS adjustment: use **J** to increase and **K** to decrease FPS by 5 (range: 1-120).
- Output cycling: press **`** (backtick) to cycle through outputs or toggle follow-focus mode.
- Compression toggle: use **U** to quickly enable/disable LZ4 compression (for Unicode modes only).
- Audio/mic compression: use **5** and **6** to cycle audio and microphone compression (off → Opus).

## Network ports
- (base port)         TCP – video frames
- (base port + 1)     TCP – input events
- (base port + 2)     TCP – system audio (server → client)
- (base port + 3)     TCP – configuration/control
- (base port + 4)     TCP – microphone (client → server)

Default base port is 9999 (see -P / --port).

## Performance tuning
- **Hardware-accelerated pixel rendering** (best visual quality):
  - **Sixel mode**: `-R sixel -Q 70` - Uses H.265 hardware encoding for pixel-perfect graphics
  - **Kitty mode**: `-R kitty -Q 70` - Kitty graphics protocol with H.265 compression
  - **Framebuffer mode**: `-R framebuffer -Q 80` - Direct framebuffer with hardware encoding
  - **KMS mode**: `-R kms -Q 85` - Direct KMS/DRM with zero-copy rendering (best performance)
  - Quality range: 50-70 for balanced performance, 80-100 for maximum quality
  - Server automatically detects and uses best available encoder:
    - NVENC (NVIDIA) - lowest latency, best performance
    - QuickSync (Intel) - excellent performance
    - VAAPI (Intel/AMD) - good performance
    - AMF (AMD) - good performance
    - Software (x265) - fallback, very slow

- **Unicode rendering** (terminal-based):
  - Maximum quality: `render_device=cuda, renderer=braille, detail=100, quality=100, color=true`
  - Smooth video: `-R hybrid -d 30 -F 60`
  - Low bandwidth: `-c -L 9 -M 256 -F 15 -d 30`
  - Low latency: `-F 60 -Q 0 -d 50`

- **Encoder-specific tuning**:
  - NVENC (NVIDIA): Ultra-low latency, CBR mode, no B-frames
  - QuickSync (Intel): Low delay mode, minimal look-ahead
  - VAAPI (Intel/AMD): Quality-based encoding
  - Software fallback: Not recommended for real-time (use hardware when possible)

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

- **H.265 encoding issues**:
  - Check available encoders: Server logs will show which encoder was selected
  - NVENC not available: Ensure NVIDIA drivers are installed
  - QuickSync not available: Ensure Intel media drivers are installed (`intel-media-driver` on Arch)
  - Software encoder very slow: This is expected - install hardware acceleration drivers

- **Sixel/Kitty rendering artifacts**:
  - Increase quality: `-Q 80` or higher
  - Check encoder logs for errors
  - Try different renderer: `-R kms` for direct rendering

- Permissions to /dev/input:
  - Ensure client (when capturing local input) can read devices or run with privileges. Exclusive grab (-x) uses EVIOCGRAB and requires access.

- Logs & debugging:
  - The server prints status and detection lines to stdout/stderr. Inspect output for messages about compositor detection, screencopy frames, encoder selection, and virtual device creation.
  - Look for "HEVC DECODER" and "HEVC ENCODER" sections in logs for codec details

## Security & limitations
- No built-in authentication: protocol uses raw TCP streams. Do not expose the server to untrusted networks. Use an SSH tunnel or VPN for remote connections.
- Input injection requires elevated privileges or membership in the `input` group to read /dev/input devices unless compositor supports virtual input protocols.
- **H.265 encoding**: Hardware encoders are prioritized for performance. Software fallback (libx265) is very slow and not recommended for real-time use.

## Design notes & behavior
- **Rendering architecture**:
  - **Unicode modes** (braille/blocks/ascii/hybrid): Rendering performed server-side, client displays ANSI escape sequences
  - **Pixel modes** (sixel/kitty/framebuffer/kms): RGB24 → H.265 encoding (server) → H.265 decoding (client) → display
- **Hardware acceleration**:
  - Server: Automatic encoder detection (NVENC > QuickSync > VAAPI > AMF > software)
  - Client: Hardware H.265 decoding with multi-threaded software fallback
  - Both support CUDA rendering for Unicode modes when built with CUDA
- **Compression**:
  - Unicode modes: Optional LZ4 compression of ANSI strings
  - Pixel modes: H.265 video compression (much more efficient than LZ4)
  - Audio: Optional Opus compression for reduced bandwidth
- Hybrid renderer chooses per-cell between braille and half-blocks for adaptive Unicode rendering.

## Contributing
- Bug reports, feature requests and PRs welcome.

## License
- MIT License – see [LICENSE](LICENSE) file for details.