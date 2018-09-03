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

#include "darknetserver.grpc.pb.h"

#include "darknet_wrapper.h"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;
using darknetServer::DetectedObjects;
using darknetServer::KeyFrame;
using darknetServer::ImageDetection;
using DarknetWrapper::WorkRequest;
using DarknetWrapper::DetectionQueue;
using DarknetWrapper::Detector;


class ServiceImpl final : public ImageDetection::Service {
 public:
	~ServiceImpl() {
		detector.Shutdown();
	}

	// TODO: Catch terminate signals and actually do clean up.
	explicit ServiceImpl(int argc, char** argv) {
		// Initialize detector - pass it the request and completion queues
		// requestQueue and completionQueue are meant for asynchronous calls
		// to the detector (not used in synchronous version.)
		// TODO: Refactor detector into base and threaded classes.
		detector.Init(argc, argv, &this->requestQueue, &this->completionQueue);
	}

	Status RequestDetection(ServerContext* context,
		const KeyFrame* frame, DetectedObjects* objects) {
		// Start timer
		probe_time_start2(&ts_server);

		// The actual processing.
		WorkRequest work;
		work.done = false;
		work.tag = this;
		std::memcpy(&(work.frame), frame, sizeof(KeyFrame));

		detector->doDetection(work);

		GPR_ASSERT(work.done == true);
		// GPU processing is done! Time to pass the results back to the client.
		objects = &(work.detectedObjects);

		std::cout << work.tag << "Server took " << probe_time_end2(&ts_server) << " milliseconds"<< std::endl;
		return Status::OK;
	}

 private:
	struct timestamp ts_server;

	// Darknet detector
	Detector detector;
	DetectionQueue requestQueue;
	DetectionQueue completionQueue;
};

int main(int argc, char** argv) {

	if(argc < 4){
		fprintf(stderr, "usage: %s <datacfg> <cfg> <weights>\n", argv[0]);
		return EXIT_FAILURE;
	}

	ServiceImpl service(argc, argv);
	std::string server_address("localhost:50051");

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
