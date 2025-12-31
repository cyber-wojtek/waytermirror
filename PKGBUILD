# Maintainer: Wojtek <your-email@example.com>
pkgname=waytermirror-git
pkgver=r11.66bc317
pkgrel=1
pkgdesc="Mirror Wayland desktop to terminal using braille/block characters"
arch=('x86_64')
url="https://github.com/cyber-wojtek/waytermirror"
license=('Apache-2.0')
depends=(
    'wayland'
    'libinput'
    'pipewire'
    'lz4'
    'rapidjson'
    'wlroots0.19'
)
makedepends=(
    'git'
    'gcc'
    'wayland-protocols'
    'wlr-protocols'
)
optdepends=(
    'cuda: GPU-accelerated rendering'
)
provides=('waytermirror')
conflicts=('waytermirror')
source=("git+${url}.git")
sha256sums=('SKIP')

pkgver() {
    cd waytermirror
    printf "r%s.%s" "$(git rev-list --count HEAD)" "$(git rev-parse --short HEAD)"
}

build() {
    cd waytermirror
    
    # Check if CUDA is available and user wants it
    if [[ -n "$WAYTERMIRROR_CUDA" ]] && command -v nvcc &> /dev/null; then
        echo "Building with CUDA support..."
        make CUDA=true
    else
        echo "Building CPU-only (set WAYTERMIRROR_CUDA=1 and install cuda to enable GPU support)"
        make
    fi
}

package() {
    cd waytermirror
    
    # Install binaries
    install -Dm755 waytermirror_server "$pkgdir/usr/bin/waytermirror_server"
    install -Dm755 waytermirror_client "$pkgdir/usr/bin/waytermirror_client"
    
    # Install license
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
    
    # Install documentation
    install -Dm644 README.md "$pkgdir/usr/share/doc/$pkgname/README.md"
}
