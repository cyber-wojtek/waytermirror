# Maintainer: Wojciech Dudek <wojtek.dudek.pl@gmail.com>

pkgname=waytermirror-git
pkgver=r12.798cb81
pkgrel=1
pkgdesc="Mirror Wayland desktop to terminal using braille/block characters"
arch=('x86_64')
url="https://github.com/cyber-wojtek/waytermirror"
license=('Apache-2.0')

depends=(
    # Core runtime
    'glibc'
    'gcc-libs'

    # Wayland / input / compositor stack
    'wayland'
    'libinput'
    'wlroots0.19'
    'wayland-protocols'
    'wlr-protocols'

    # IPC / multimedia
    'pipewire'

    # Compression / data
    'lz4'

    # Device / udev / glib stack
    'systemd-libs'   # libudev
    'glib2'          # glib + gobject + gudev
    'libffi'
    'pcre2'

    # Input helpers
    'libevdev'
    'libwacom'
    'mtdev'

    # Scripting / config
    'lua'

    # JSON
    'rapidjson'

    # CUDA (Need to include it, there is not conditional dependency support)
    'cuda'
)

makedepends=(
    'git'
    'gcc'
    'make'
    'pkgconf'
)

optdepends=(
    'cuda: GPU-accelerated rendering (libcudart)'
)

provides=('waytermirror')
conflicts=('waytermirror')

source=("git+${url}.git")
sha256sums=('SKIP')

pkgver() {
    cd waytermirror
    printf "r%s.%s" \
        "$(git rev-list --count HEAD)" \
        "$(git rev-parse --short HEAD)"
}

build() {
    cd waytermirror

    if [[ -n "$WAYTERMIRROR_CUDA" ]] && command -v nvcc &>/dev/null; then
        echo "==> Building with CUDA support"
        make CUDA=true
    else
        echo "==> Building CPU-only (set WAYTERMIRROR_CUDA=1 to enable CUDA)"
        make
    fi
}

package() {
    cd waytermirror

    install -Dm755 waytermirror_server \
        "$pkgdir/usr/bin/waytermirror_server"
    install -Dm755 waytermirror_client \
        "$pkgdir/usr/bin/waytermirror_client"

    install -Dm644 LICENSE \
        "$pkgdir/usr/share/licenses/$pkgname/LICENSE"

    install -Dm644 README.md \
        "$pkgdir/usr/share/doc/$pkgname/README.md"
}
