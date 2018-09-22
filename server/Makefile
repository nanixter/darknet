# Darknet Makefile. Adapted from example Makefile provided by grpc authors.
GPU=1

CXX = g++
DARKNET_HEADER_PATH = ../include/
CPPFLAGS += `pkg-config --cflags grpc` -g -O4
CXXFLAGS += -std=c++11 -I $(DARKNET_HEADER_PATH)
LDFLAGS += -L/usr/local/lib `pkg-config --libs grpc++ grpc opencv`\
           -Wl,--no-as-needed -lgrpc++_reflection -Wl,--as-needed -ldl -lpthread\
           -L../ -Wl,--as-needed -ldarknet
FLATC = flatc

ifeq ($(GPU), 1)
CXXFLAGS+= -DGPU -I/usr/local/cuda/include/
LDFLAGS+= -L/usr/local/cuda/lib64 -Wl,--as-needed -lcuda -lcudart -lcublas -lcurand -lcudnn
endif

all: sync async

certs:
	./gen_cert.sh

sync: client server

async: async_client async_server

client: darknetserver.grpc.fb.o client.o
	$(CXX) $^ $(LDFLAGS) -o $@

server: darknetserver.grpc.fb.o server.o
	$(CXX) $^ -I $(DARKNET_HEADER_PATH) $(LDFLAGS) -o $@

async_client: darknetserver.grpc.fb.o async_client.o
	$(CXX) $^ $(LDFLAGS) -o $@

async_server: darknetserver.grpc.fb.o async_server.o
	$(CXX) $^ -I $(DARKNET_HEADER_PATH) $(LDFLAGS) -o $@

.PRECIOUS: %.grpc.fb.cc darknetserver_generated.h
%.grpc.fb.cc: %.fbs
	$(FLATC) --grpc --cpp $<

clean:
	rm -f *.o *.fb.cc *.fb.h darknetserver_generated.h client server async_client async_server
clean_certs:
	rm -f *.csr *.key *.crt
