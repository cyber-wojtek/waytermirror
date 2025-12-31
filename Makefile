CC = gcc
CXX = g++
CXXFLAGS = -std=c++23 -Wall -O3
CFLAGS = -Wall -O3
INCLUDES = -I/usr/include/wayland-client -I/usr/include/libinput -I/usr/include/libudev -I/usr/include/pipewire-0.3 -I/usr/include/spa-0.2
LIBS_BASE = -lwayland-client -lm -llz4 -lpthread -linput -ludev -lpipewire-0.3

# CUDA support: make CUDA=true
ifeq ($(CUDA),true)
    LIBS = $(LIBS_BASE) -L/opt/cuda/lib64 -lcudart
    CUDA_OBJ = braille_renderer_cuda.o
else
    LIBS = $(LIBS_BASE)
    CUDA_OBJ = braille_renderer_stub.o
endif

PROTOCOLS_C = wlr-screencopy-unstable-v1-client-protocol.c \
              wlr-virtual-pointer-unstable-v1-client-protocol.c \
              virtual-keyboard-unstable-v1-client-protocol.c \
			  xdg-shell-protocol.c \
			  wlr-foreign-toplevel-management-unstable-v1-client-protocol.c

PROTOCOLS_O = $(PROTOCOLS_C:.c=.o)

all: waytermirror_server waytermirror_client

braille_renderer_cuda.o: braille_renderer_cuda.cu
	nvcc -O3 -c braille_renderer_cuda.cu -o braille_renderer_cuda.o

braille_renderer_stub.o: braille_renderer_stub.cpp
	$(CXX) $(CXXFLAGS) -c braille_renderer_stub.cpp -o braille_renderer_stub.o

waytermirror_server: waytermirror_server.o $(PROTOCOLS_O) $(CUDA_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) ${INCLUDES}

waytermirror_server.o: waytermirror_server.cpp
	$(CXX) $(CXXFLAGS) -c waytermirror_server.cpp ${INCLUDES}

waytermirror_client: waytermirror_client.o $(PROTOCOLS_O)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) ${INCLUDES}
waytermirror_client.o: waytermirror_client.cpp
	$(CXX) $(CXXFLAGS) -c waytermirror_client.cpp ${INCLUDES}

%.o: %.c
	$(CC) $(CFLAGS) -c $< ${INCLUDES}

clean:
	rm -f waytermirror_server waytermirror_client *.o

.PHONY: all clean
