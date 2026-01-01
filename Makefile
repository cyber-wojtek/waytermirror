CC = gcc
CXX = g++
CXXFLAGS = -std=c++23 -Wall -O3
CFLAGS = -Wall -O3

# Include directories
INCLUDES = -I/usr/include/wayland-client \
           -I/usr/include/libinput \
           -I/usr/include/libudev \
           -I/usr/include/pipewire-0.3 \
           -I/usr/include/spa-0.2 \
           $(shell pkg-config --cflags gio-2.0)

# Base libraries
LIBS_BASE = -lwayland-client -lm -llz4 -lpthread -linput -ludev $(shell pkg-config --libs gio-2.0)

# PipeWire support (default: enabled)
# To disable: make PIPEWIRE=false
PIPEWIRE ?= true
ifeq ($(PIPEWIRE),true)
LIBS_BASE += -lpipewire-0.3
CXXFLAGS += -DHAVE_PIPEWIRE
endif

# CUDA support: make CUDA=true
ifeq ($(CUDA),true)
LIBS = $(LIBS_BASE) -L/opt/cuda/lib64 -lcudart
CUDA_OBJ = braille_renderer_cuda.o
CXXFLAGS += -DHAVE_CUDA
else
LIBS = $(LIBS_BASE)
CUDA_OBJ = braille_renderer_stub.o
endif

# Wayland protocol files
PROTOCOLS_C = wlr-screencopy-unstable-v1-client-protocol.c \
              wlr-virtual-pointer-unstable-v1-client-protocol.c \
              virtual-keyboard-unstable-v1-client-protocol.c \
              xdg-shell-protocol.c \
              wlr-foreign-toplevel-management-unstable-v1-client-protocol.c

PROTOCOLS_O = $(PROTOCOLS_C:.c=.o)

# New backend source files
BACKEND_SOURCES = pipewire_capture.cpp virtual_input.cpp
BACKEND_OBJECTS = $(BACKEND_SOURCES:.cpp=.o)

# Server object files
SERVER_OBJECTS = waytermirror_server.o $(PROTOCOLS_O) $(CUDA_OBJ) $(BACKEND_OBJECTS)

# Client object files  
CLIENT_OBJECTS = waytermirror_client.o $(PROTOCOLS_O)

# Default target
all: waytermirror_server waytermirror_client

# CUDA renderer
braille_renderer_cuda.o: braille_renderer_cuda.cu
	nvcc -O3 -c braille_renderer_cuda.cu -o braille_renderer_cuda.o

# CPU stub renderer
braille_renderer_stub.o: braille_renderer_stub.cpp
	$(CXX) $(CXXFLAGS) -c braille_renderer_stub.cpp -o braille_renderer_stub.o

# Server executable
waytermirror_server: $(SERVER_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Server main file
waytermirror_server.o: waytermirror_server.cpp pipewire_capture.h virtual_input.h
	$(CXX) $(CXXFLAGS) -c waytermirror_server.cpp $(INCLUDES)

# Backend object files
pipewire_capture.o: pipewire_capture.cpp pipewire_capture.h
	$(CXX) $(CXXFLAGS) -c pipewire_capture.cpp $(INCLUDES)

virtual_input.o: virtual_input.cpp virtual_input.h
	$(CXX) $(CXXFLAGS) -c virtual_input.cpp $(INCLUDES)

# Client executable
waytermirror_client: $(CLIENT_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Client main file
waytermirror_client.o: waytermirror_client.cpp
	$(CXX) $(CXXFLAGS) -c waytermirror_client.cpp $(INCLUDES)

# Protocol files
%.o: %.c
	$(CC) $(CFLAGS) -c $< $(INCLUDES)

# Install target
install: waytermirror_server waytermirror_client
	install -D -m 755 waytermirror_server $(DESTDIR)/usr/local/bin/waytermirror_server
	install -D -m 755 waytermirror_client $(DESTDIR)/usr/local/bin/waytermirror_client
	install -D -m 644 99-waytermirror-uinput.rules $(DESTDIR)/etc/udev/rules.d/99-waytermirror-uinput.rules
	@echo "Installation complete!"
	@echo "Run 'sudo udevadm control --reload-rules && sudo udevadm trigger' to activate udev rules"
	@echo "Add your user to the input group: sudo usermod -a -G input $$USER"

# Uninstall target
uninstall:
	rm -f $(DESTDIR)/usr/local/bin/waytermirror_server
	rm -f $(DESTDIR)/usr/local/bin/waytermirror_client
	rm -f $(DESTDIR)/etc/udev/rules.d/99-waytermirror-uinput.rules

# Clean build artifacts
clean:
	rm -f waytermirror_server waytermirror_client *.o

# Clean everything including backup files
distclean: clean
	rm -f *~ *.bak

# Help target
help:
	@echo "Waytermirror Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all (default)    - Build both server and client"
	@echo "  waytermirror_server - Build server only"
	@echo "  waytermirror_client - Build client only"
	@echo "  install          - Install binaries and udev rules"
	@echo "  uninstall        - Remove installed files"
	@echo "  clean            - Remove build artifacts"
	@echo "  distclean        - Remove all generated files"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  CUDA=true        - Enable CUDA acceleration (default: false)"
	@echo "  PIPEWIRE=false   - Disable PipeWire support (default: true)"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build with PipeWire, without CUDA"
	@echo "  make CUDA=true          # Build with CUDA acceleration"
	@echo "  make PIPEWIRE=false     # Build without PipeWire (wlr-only)"
	@echo "  make install            # Install to /usr/local/bin"
	@echo "  make install DESTDIR=/tmp/pkg  # Install to custom root"

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@command -v $(CC) >/dev/null 2>&1 || { echo "Error: gcc not found"; exit 1; }
	@command -v $(CXX) >/dev/null 2>&1 || { echo "Error: g++ not found"; exit 1; }
	@pkg-config --exists wayland-client || { echo "Error: wayland-client not found"; exit 1; }
	@pkg-config --exists libinput || { echo "Error: libinput not found"; exit 1; }
ifeq ($(PIPEWIRE),true)
	@pkg-config --exists libpipewire-0.3 || { echo "Warning: PipeWire not found, use PIPEWIRE=false to disable"; }
endif
ifeq ($(CUDA),true)
	@command -v nvcc >/dev/null 2>&1 || { echo "Error: CUDA nvcc not found"; exit 1; }
endif
	@echo "All dependencies OK!"

# Debug build
debug: CXXFLAGS = -std=c++23 -Wall -g -O0 -DDEBUG
debug: CFLAGS = -Wall -g -O0 -DDEBUG
debug: clean all

# Show build configuration
config:
	@echo "Build Configuration:"
	@echo "  CXX: $(CXX)"
	@echo "  CXXFLAGS: $(CXXFLAGS)"
	@echo "  CUDA: $(if $(filter true,$(CUDA)),enabled,disabled)"
	@echo "  PIPEWIRE: $(if $(filter true,$(PIPEWIRE)),enabled,disabled)"
	@echo "  LIBS: $(LIBS)"

.PHONY: all clean distclean install uninstall help check-deps debug config
