# Waytermirror

Real-time Wayland screen mirroring to a terminal using Unicode braille characters, half-blocks, or ASCII. Includes full bidirectional input forwarding, audio streaming, and optional CUDA acceleration.

![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)

## How It Works

Waytermirror is a client-server application:

1. **Server** runs on the Wayland host — captures screen via `wlr-screencopy`, captures system audio via PipeWire, and injects input events via virtual keyboard/pointer protocols
2. **Client** runs in any terminal — receives frames, **renders them server-side** to ANSI escape sequences, displays in terminal, captures local input via libinput, and forwards it to server

The server does all rendering work. The client just displays pre-rendered ANSI output and captures input. This means:
- Rendering quality/performance depends on server hardware
- Client can be a low-power device (just needs a terminal)
- CUDA acceleration happens on the server

### Rendering Pipeline

```
Screen → wlr-screencopy → Frame buffer → Renderer → ANSI string → LZ4 → TCP → Client terminal
                                            ↓
                              ┌─────────────┴─────────────┐
                              │  braille (2×4 subpixel)   │
                              │  blocks (▀ half-blocks)   │
                              │  ascii (character ramp)   │
                              │  hybrid (auto-switch)     │
                              └───────────────────────────┘
```

### Hybrid Renderer

The `hybrid` renderer (default: `--renderer hybrid`) automatically chooses per-cell:
- **Braille** for areas with edges/detail (detected via luminance variance)
- **Half-blocks** for flat/smooth areas (faster, cleaner)

This gives the best balance of quality and performance.

## Features

- **4 renderers**: braille, blocks, ascii, hybrid
- **3 color modes**: 16, 256, or 24-bit truecolor
- **CUDA acceleration**: GPU rendering on server (NVIDIA only)
- **Bidirectional audio**: System audio → client, microphone → server
- **Full input**: Keyboard + mouse via libinput with optional exclusive grab
- **Zoom**: Magnification with mouse-follow
- **Compression**: LZ4 + delta encoding for bandwidth efficiency
- **Focus follow**: Mirror the focused window's output (`--output follow`)
- **Multi-monitor**: Select which output to mirror

## Requirements

### Build Dependencies

```bash
# Arch Linux
sudo pacman -S wayland libinput pipewire lz4 rapidjson cuda

# Plus header-only: argparse (from AUR or manual install)
yay -S argparse

# Debian/Ubuntu  
sudo apt install libwayland-dev libinput-dev libpipewire-0.3-dev liblz4-dev rapidjson-dev
# CUDA requires NVIDIA toolkit from nvidia.com
```

### Runtime

- **Compositor**: wlroots-based (Hyprland, Sway, River) or any with:
  - `wlr-screencopy-unstable-v1`
  - `wlr-virtual-pointer-unstable-v1`
  - `virtual-keyboard-unstable-v1`
  - `wlr-foreign-toplevel-management-unstable-v1` (for *proper* focus tracking (that's currently broken as of current wlroots 0.19))
- **Audio**: PipeWire
- **Input**: User must be in `input` group (or run client as root)
- **CUDA**: NVIDIA GPU + driver + CUDA toolkit (optional)

**NOT supported**: GNOME, KDE Plasma (they don't implement wlr protocols)

## Building

```bash
git clone https://github.com/USER/waytermirror.git
cd waytermirror
make -j$(nproc)
```

This produces:
- `waytermirror_server` — run on the machine with the display
- `waytermirror_client` — run in any terminal (local or remote)

## Usage

### Server

```bash
./waytermirror_server [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-P, --port <n>` | Base port | 9999 |
| `-F, --capture-fps <n>` | Capture framerate | 30 |
| `-C, --compositor <type>` | auto, hyprland, sway, kde, gnome, generic | auto |
| `-n, --no-video` | Disable screen capture | off |
| `-A, --no-audio` | Disable system audio | off |
| `-N, --no-input` | Disable input injection | off |
| `-m, --no-microphone` | Disable microphone reception | off |

### Client

```bash
./waytermirror_client -H <server_ip> [options]
```

#### Connection

| Option | Description | Default |
|--------|-------------|---------|
| `-H, --host` | Server IP/hostname | *required* |
| `-P, --port` | Server port | 9999 |

#### Video

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output <n\|follow>` | Output index or `follow` for focus tracking | 0 |
| `-F, --fps <n>` | Target FPS | 30 |
| `-M, --mode <16\|256\|true>` | Color mode | 256 |
| `-R, --renderer <type>` | braille, blocks, ascii, hybrid | braille |
| `-r, --render-device <cpu\|cuda>` | Rendering backend (server-side) | cpu |
| `-d, --detail-level <0-100>` | 0=smooth/fast, 100=sharp/detailed | 50 |
| `-Q, --quality <0-100>` | Threshold search precision | 50 |
| `-S, --scale <factor>` | Scale factor | 1.0 |
| `-k, --keep-aspect-ratio` | Maintain aspect ratio | off |
| `-c, --compress` | Enable LZ4 compression | off |
| `-L, --compression-level <0-12>` | 0=fast, 12=best ratio | 0 |
| `-n, --no-video` | Disable video | off |

#### Input

| Option | Description | Default |
|--------|-------------|---------|
| `-N, --no-input` | Disable input forwarding | off |
| `-x, --exclusive-input` | Grab devices exclusively (EVIOCGRAB) | off |
| `-C, --center-mouse` | Start mouse at screen center | off |

#### Audio

| Option | Description | Default |
|--------|-------------|---------|
| `-A, --no-audio` | Disable audio playback | off |
| `-p, --no-microphone` | Disable microphone capture | off |

#### Zoom

| Option | Description | Default |
|--------|-------------|---------|
| `-z, --zoom` | Start with zoom enabled | off |
| `-Z, --zoom-level <1-10>` | Magnification | 2.0 |
| `-X, --zoom-width <px>` | Viewport width | 800 |
| `-Y, --zoom-height <px>` | Viewport height | 600 |
| `-f, --zoom-follow` | Follow mouse cursor | on |
| `-s, --zoom-smooth` | Smooth panning | on |
| `-D, --zoom-speed <n>` | Pan speed (px/frame) | 20 |

## Examples

**Basic LAN streaming:**
```bash
# Server (on desktop with display)
./waytermirror_server -F 60

# Client (in terminal, same or different machine)
./waytermirror_client -H 192.168.1.100 -F 60 -M true -R hybrid
```

**High quality with CUDA:**
```bash
./waytermirror_client -H 192.168.1.100 -r cuda -d 90 -Q 100 -M true -R braille
```

**Low bandwidth (remote/slow connection):**
```bash
./waytermirror_client -H server.example.com -c -L 12 -F 15 -M 256 -d 30
```

**Follow focused window:**
```bash
./waytermirror_client -H 192.168.1.100 -o follow
```

**Input-only (no video, control remote desktop):**
```bash
./waytermirror_client -H 192.168.1.100 -n -x
```

## Network Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 9999 | TCP | Video frames |
| 10000 | TCP | Input events |
| 10001 | TCP | System audio (server→client) |
| 10002 | TCP | Configuration |
| 10003 | TCP | Microphone (client→server) |

## Keyboard Shortcuts

| Combo | Action |
|-------|--------|
| `Ctrl+Alt+Shift+Delete+X` | Exit client |
| `Ctrl+Alt+Shift+Delete+Z` | Toggle zoom |

## Performance Tuning

| Goal | Settings |
|------|----------|
| Maximum quality | `-r cuda -d 100 -Q 100 -M true -R braille` |
| Smooth video | `-d 30 -R hybrid -F 60` |
| Low bandwidth | `-c -L 9 -d 30 -M 256 -F 15` |
| Minimum latency | `-d 50 -Q 0 -F 60` |

### Detail Level vs Quality

- **Detail level** (`-d`): How much visual detail to preserve. Lower = smoother/faster, higher = sharper edges
- **Quality** (`-Q`): Threshold search precision. Higher = more accurate pattern selection, slower

## Troubleshooting

### "Failed to initialize libinput"
```bash
sudo usermod -aG input $USER
# Log out and back in
```

### No screen capture
Your compositor doesn't support `wlr-screencopy`. Check:
```bash
wayland-info 2>/dev/null | grep -i screencopy
```

### Virtual input not working
Check server output for:
```
Virtual pointer created: YES
Virtual keyboard created: YES
```
If NO, your compositor doesn't support the required protocols.

### CUDA errors
```bash
nvcc --version
nvidia-smi
```

### Audio not working
Ensure PipeWire is running:
```bash
systemctl --user status pipewire
```

## License

Apache License 2.0 — see [LICENSE](LICENSE)

## Acknowledgments

- [wlroots](https://gitlab.freedesktop.org/wlroots/wlroots) — Wayland protocols
- [PipeWire](https://pipewire.org/) — Audio
- [LZ4](https://lz4.github.io/lz4/) — Compression
- [argparse](https://github.com/p-ranav/argparse) — CLI parsing
