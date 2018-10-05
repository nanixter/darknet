# LiveStreamDarknet Makefile.
GPU=1

CXX = g++
DARKNET_HEADER_PATH = ../include/
CPPFLAGS += `pkg-config libavformat libavcodec` -g -O4
CXXFLAGS += -std=c++11 -I $(DARKNET_HEADER_PATH) -I ./NvCodec
LDFLAGS += -L/usr/local/lib `pkg-config --libs libavformat libavcodec opencv`\
           -Wl,--as-needed -ldl -lpthread\
           -L../ -Wl,--as-needed -ldarknet

ifeq ($(GPU), 1)
CXXFLAGS+= -DGPU -I/usr/local/cuda/include/
LDFLAGS+= -L/usr/local/cuda/lib64 -Wl,--as-needed -lcuda -lcudart -lcublas -lcurand -lcudnn
endif

all: sync

certs:
	./gen_cert.sh

sync: client server

server: main.o
	$(CXX) $^ -I $(DARKNET_HEADER_PATH) $(LDFLAGS) -o $@
clean:
	rm -f *.o server
clean_certs:
	rm -f *.csr *.key *.crt
