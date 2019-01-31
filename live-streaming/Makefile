# LiveStreamDarknet Makefile.
GPU=1

CXX = clang++
NVCC = nvcc
DARKNET_HEADER_PATH = ../include/
CPPFLAGS += `pkg-config libavformat libavcodec` -g -O3
CXXFLAGS += -std=c++11 -I $(DARKNET_HEADER_PATH) -Invpipe/src/NvCodec/Utils -Iutils
LDFLAGS += -L/usr/local/lib `pkg-config --libs libavformat libavcodec` -lNvPipe \
			-Wl,--as-needed -ldl -lpthread \
			-L../ -Wl,--as-needed -ldarknet

ifeq ($(GPU), 1)
CXXFLAGS+= -DGPU -I/usr/local/cuda/include/
LDFLAGS+= -L/usr/local/cuda/lib64 -Wl,--as-needed -lcuda -lcudart -lcublas -lcurand -lcudnn -lnppisu -lnppicom -lnppig -lnppc -lnppidei -lnvToolsExt
endif

all: server

server: main.o cudaYUV-NV12.o cudaResize.o cudaRGB.o cudaOverlay.o
	$(CXX) $^ -I $(DARKNET_HEADER_PATH) $(LDFLAGS) -o $@

cudaYUV-NV12.o: utils/cudaYUV-NV12.cu
	nvcc -g -std=c++11 -c -arch=sm_60 $^ -o $@

cudaResize.o: utils/cudaResize.cu
	nvcc -g -std=c++11 -c -arch=sm_60 $^ -o $@

cudaRGB.o: utils/cudaRGB.cu
	nvcc -g -std=c++11 -c -arch=sm_60 $^ -o $@

cudaOverlay.o: utils/cudaOverlay.cu
	nvcc -g -std=c++11 -c -arch=sm_60 $^ -o $@

clean:
	rm -f *.o server
