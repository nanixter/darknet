#include <iostream>
#include <memory>
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdio>

#include <grpcpp/grpcpp.h>
#include <grpc/support/log.h>
#include <thread>
#include <sys/time.h>

#include "darknetserver.grpc.fb.h"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;
using darknetServer::DetectedObjects;
using darknetServer::DetectedObject;
using darknetServer::KeyFrame;
using darknetServer::bbox;
using darknetServer::ImageDetection;

#include "darknet_wrapper_fb.h"

using DarknetWrapper::WorkRequest;
// using DarknetWrapper::DetectionQueue;
using DarknetWrapper::Detector;


class ServiceImpl final : public ImageDetection::Service {
 public:
	~ServiceImpl() {
		detector.Shutdown();
	}

	// TODO: Catch terminate signals and actually do clean up.
	explicit ServiceImpl(int argc, char** argv) {
		detector.Init(argc, argv);
	}

	Status RequestDetection(::grpc::ServerContext* context,
						const flatbuffers::grpc::Message<KeyFrame>* requestMessage,
						flatbuffers::grpc::Message<DetectedObjects>* responseMessage) {
		// Start timer
		probe_time_start2(&ts_server);

		// This structure was mostly created for the Async version, but we use it too...
		WorkRequest work;
		work.done = false;
		work.tag = this;
		work.frame = requestMessage->GetRoot();
		work.messageBuilder = &messageBuilder;
		work.responseMessage = responseMessage;

		// The actual processing.
		detector.doDetection(work);

		GPR_ASSERT(work.done == true);

		std::cout << work.tag << "Server took " << probe_time_end2(&ts_server) << " milliseconds"<< std::endl;
		return Status::OK;
	}

 private:
	struct timestamp ts_server;

	// Darknet detector
	Detector detector;
	flatbuffers::grpc::MessageBuilder messageBuilder;
};

int main(int argc, char** argv) {

	if(argc < 4){
		fprintf(stderr, "usage: %s <datacfg> <cfg> <weights>\n", argv[0]);
		return EXIT_FAILURE;
	}

	ServiceImpl service(argc, argv);
	std::string server_address("128.83.122.71:50051");
	//std::string server_address("localhost:50051");

	ServerBuilder builder;

	// Listen on the given address without any authentication mechanism.
	builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
	builder.SetMaxReceiveMessageSize(INT_MAX);

	// Register "service_" as the instance through which we'll communicate with
	// clients. In this case it corresponds to an *asynchronous* service.
	builder.RegisterService(&service);

	// Finally assemble the server.
	std::unique_ptr<Server> server_ = builder.BuildAndStart();
	std::cout << "Server listening on " << server_address << std::endl;
	// Wait for server to quit
	server_->Wait();

	return EXIT_SUCCESS;
}
