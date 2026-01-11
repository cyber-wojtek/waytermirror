project('waytermirror', ['c', 'cpp'],
  version: '0.1.0',
  default_options: [
    'cpp_std=c++23',
    'c_std=c11',
    'warning_level=3',
    'optimization=3',
    'buildtype=release'
  ],
  meson_version: '>=0.63.0'
)

fs = import('fs')

# Options
cuda_opt = get_option('nvidia_cuda')
debug_opt = get_option('buildtype') == 'debug'

# Compiler setup
cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# Common flags
common_flags = []
c_flags = []
cpp_flags = []

if debug_opt
  common_flags += ['-DDEBUG']
endif

# Dependencies
wayland_client = dependency('wayland-client')
threads = dependency('threads')
libinput = dependency('libinput')
libudev = dependency('libudev')
libm = cc.find_library('m', required: true)
lz4 = dependency('liblz4')
opus = dependency('opus')
opusfile = dependency('opusfile')
libsixel = dependency('libsixel')
libdrm = dependency('libdrm')
gbm = dependency('gbm')
libavcodec = dependency('libavcodec')
libavutil = dependency('libavutil')
libswscale = dependency('libswscale')
x265 = dependency('x265')
libpng = dependency('libpng')
gio = dependency('gio-2.0')

# PipeWire (required for audio)
pipewire_dep = dependency('libpipewire-0.3', required: true)
common_flags += ['-DHAVE_PIPEWIRE']
message('PipeWire support: enabled (required)')

# Optional CUDA
cuda_dep = []
cuda_sources = []
if cuda_opt
  cuda_dep = dependency('cuda', required: false)
  if not cuda_dep.found()
    # Fallback to manual detection
    cuda_path = get_option('nvidia_cuda_path')
    if cuda_path == ''
      cuda_path = '/opt/cuda'
    endif
    
    cuda_dep = declare_dependency(
      include_directories: include_directories(cuda_path / 'include'),
      dependencies: cpp.find_library('cudart', dirs: cuda_path / 'lib64', required: true)
    )
  endif
  
  # CUDA renderer
  nvcc = find_program('nvcc', required: true)
  cuda_gen = generator(nvcc,
    output: '@BASENAME@.o',
    arguments: ['-O3', '-c', '@INPUT@', '-o', '@OUTPUT@']
  )
  cuda_sources = cuda_gen.process('braille_renderer_cuda.cu')
  
  common_flags += ['-DHAVE_CUDA']
  message('CUDA support: enabled')
else
  # CPU stub renderer
  cuda_sources = files('braille_renderer_stub.cpp')
  message('CUDA support: disabled')
endif

# Wayland protocols
wayland_scanner = find_program('wayland-scanner', required: true)

# System wayland-protocols (fallback)
wayland_protocols_dir = get_option('wayland_protocols_dir')
if wayland_protocols_dir == ''
  wp = dependency('wayland-protocols', required: false)
  if wp.found()
    wayland_protocols_dir = wp.get_variable('pkgdatadir')
  else
    wayland_protocols_dir = '/usr/share/wayland-protocols'
  endif
endif

# System wlr-protocols (fallback)
wlr_protocols_dir = get_option('wlr_protocols_dir')
if wlr_protocols_dir == ''
  wp = dependency('wlr-protocols', required: false)
  if wp.found()
    wlr_protocols_dir = wp.get_variable('pkgdatadir')
  else
    wlr_protocols_dir = '/usr/share/wlr-protocols'
  endif
endif

protocol_sources = []
protocol_headers = []

# Protocol definitions - matching Makefile exactly
protocols = [
  ['wlr-screencopy-unstable-v1', 'unstable/wlr-screencopy-unstable-v1.xml'],
  ['wlr-virtual-pointer-unstable-v1', 'unstable/wlr-virtual-pointer-unstable-v1.xml'],
  ['virtual-keyboard-unstable-v1', 'unstable/virtual-keyboard-unstable-v1.xml'],
  ['xdg-shell', 'stable/xdg-shell/xdg-shell.xml'],
  ['wlr-foreign-toplevel-management-unstable-v1', 'unstable/wlr-foreign-toplevel-management-unstable-v1.xml']
]

# Process protocols - inline path resolution without functions
foreach p : protocols
  name = p[0]
  relpath = p[1]
  
  # Resolve protocol XML path (prefer local, fallback to system)
  local_xml = join_paths(meson.project_source_root(), 'protocols', relpath)
  system_xml = join_paths(wayland_protocols_dir, relpath)
  system_wlr_xml = join_paths(wlr_protocols_dir, relpath)
  
  xml_file = ''
  if fs.exists(local_xml)
    xml_file = files(local_xml)
  elif fs.exists(system_xml)
    xml_file = files(system_xml)
  elif fs.exists(system_wlr_xml)
    xml_file = files(system_wlr_xml)
  else
    error('Wayland protocol XML not found: ' + relpath)
  endif

  # Generate protocol source
  protocol_sources += custom_target(
    name + '-client-protocol.c',
    input: xml_file,
    output: name + '-client-protocol.c',
    command: [wayland_scanner, 'private-code', '@INPUT@', '@OUTPUT@']
  )

  # Generate protocol header
  protocol_headers += custom_target(
    name + '-client-protocol.h',
    input: xml_file,
    output: name + '-client-protocol.h',
    command: [wayland_scanner, 'client-header', '@INPUT@', '@OUTPUT@']
  )
endforeach

protocol_inc = include_directories('.')

# Backend sources (matching Makefile BACKEND_SOURCES)
backend_sources = files(
  'pipewire_capture.cpp',
  'virtual_input.cpp'
)

# Base dependencies (matching Makefile LIBS_BASE)
base_deps = [
  wayland_client,
  threads,
  libinput,
  libudev,
  libm,
  lz4,
  opus,
  opusfile,
  libsixel,
  libdrm,
  gbm,
  libavcodec,
  libavutil,
  libswscale,
  x265,
  libpng,
  gio,
  pipewire_dep
]

# Server dependencies
server_deps = base_deps + cuda_dep

# Server sources (matching Makefile SERVER_OBJECTS)
server_sources = files('waytermirror_server.cpp') + backend_sources + protocol_sources + cuda_sources

# Server executable
waytermirror_server = executable(
  'waytermirror_server',
  server_sources,
  include_directories: protocol_inc,
  dependencies: server_deps,
  c_args: common_flags + c_flags,
  cpp_args: common_flags + cpp_flags,
  install: true
)

# Client sources (matching Makefile CLIENT_OBJECTS)
client_sources = files('waytermirror_client.cpp') + protocol_sources

# Client executable
waytermirror_client = executable(
  'waytermirror_client',
  client_sources,
  include_directories: protocol_inc,
  dependencies: base_deps,
  c_args: common_flags + c_flags,
  cpp_args: common_flags + cpp_flags,
  install: true
)

# Install udev rules
install_data(
  '99-waytermirror-uinput.rules',
  install_dir: get_option('sysconfdir') / 'udev' / 'rules.d'
)

install_data(
  '99-waytermirror-video.rules',
  install_dir: get_option('sysconfdir') / 'udev' / 'rules.d'
)

# Post-install script
meson.add_install_script('sh', '-c', '''
echo "Installation complete!"
echo "Run 'sudo udevadm control --reload-rules && sudo udevadm trigger' to activate udev rules"
echo "Add your user to the input group: sudo usermod -a -G input $USER"
''')

# Summary
summary({
  'Prefix': get_option('prefix'),
  'Bindir': get_option('bindir'),
  'Sysconfdir': get_option('sysconfdir'),
  'PipeWire': true,
  'CUDA': cuda_opt,
  'Build type': get_option('buildtype'),
}, section: 'Configuration')

summary({
  'C compiler': cc.get_id() + ' ' + cc.version(),
  'C++ compiler': cpp.get_id() + ' ' + cpp.version(),
}, section: 'Compilers')
