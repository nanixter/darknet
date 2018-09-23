#include <iostream>
#include <fstream>
#include <sstream>
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

#include "darknet_wrapper.h"

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
		work.img = detector.convertImage(requestMessage->GetRoot());
		work.dets = nullptr;
		work.nboxes = 0;
		work.classes = 0;

		// The actual processing.
		detector.doDetection(work);

		// Copy detected objects to the Request
		GPR_ASSERT(work.done == true);
		GPR_ASSERT(work.dets != nullptr);

		std::vector<flatbuffers::Offset<DetectedObject>> objects;
		int numObjects = 0;
		for (int i = 0; i < work.nboxes; i++) {
			if(work.dets[i].objectness == 0) continue;
			bbox box(work.dets[i].bbox.x, work.dets[i].bbox.y, work.dets[i].bbox.w, work.dets[i].bbox.h);
			std::vector<float> prob;
			for (int j = 0; j < work.classes; j++) {
				prob.push_back(work.dets[i].prob[j]);
			}
			auto objectOffset = darknetServer::CreateDetectedObjectDirect(messageBuilder, &box, work.dets[i].classes, work.dets[i].objectness, work.dets[i].sort_class, &prob);
			objects.push_back(objectOffset);
			numObjects++;
		}

		flatbuffers::Offset<DetectedObjects> detectedObjectsOffset = darknetServer::CreateDetectedObjectsDirect(messageBuilder, numObjects, &objects);

		messageBuilder.Finish(detectedObjectsOffset);
		*responseMessage = messageBuilder.ReleaseMessage<DetectedObjects>();
		assert(responseMessage->Verify());

		// Clean up
		free_detections(work.dets, work.nboxes);

		std::cout << work.tag << "Server took " << probe_time_end2(&ts_server) << " milliseconds"<< std::endl;
		return Status::OK;
	}

 private:
	struct timestamp ts_server;
	int classes;
	// Darknet detector
	Detector detector;
	flatbuffers::grpc::MessageBuilder messageBuilder;
};


void readFile(const std::string& filename, std::string& data)
{
	std::ifstream file(filename.c_str(), std::ifstream::in);

	if(file.is_open()) {
		std::stringstream ss;
		ss << file.rdbuf();
		file.close ();

		data = ss.str();
	}
	return;
}

int main(int argc, char** argv) {

	if(argc < 4){
		fprintf(stderr, "usage: %s <datacfg> <cfg> <weights>\n", argv[0]);
		return EXIT_FAILURE;
	}

	ServiceImpl service(argc, argv);
	//std::string server_address("128.83.122.71:50051");
	std::string server_address("zemaitis:50051");

	ServerBuilder builder;
	std::string key;
	std::string cert;
	std::string root;

	readFile("server.crt", cert);
	readFile("server.key", key);
	readFile("ca.crt", root);

	grpc::SslServerCredentialsOptions::PemKeyCertPair keycert = {key, cert};

	grpc::SslServerCredentialsOptions sslOps;
	sslOps.pem_root_certs = root;
	sslOps.pem_key_cert_pairs.push_back (keycert);

	// Listen on the given address with TLS authentication.
	builder.AddListeningPort(server_address, grpc::SslServerCredentials( sslOps ));
	// builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
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
