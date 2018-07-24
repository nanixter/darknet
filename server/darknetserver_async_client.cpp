#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>
#include <grpc/support/log.h>
#include <thread>

#include "darknetserver.grpc.pb.h"
#include "darknet.h"

using grpc::Channel;
using grpc::ClientAsyncReaderWriter;
using grpc::ClientContext;
using grpc::CompletionQueue;
using grpc::Status;
using darknetServer::DetectedObjects;
using darknetServer::KeyFrame;
using darknetServer::ImageDetection;

class ImageDetectionClient {
  public:
    explicit ImageDetectionClient(std::shared_ptr<Channel> channel)
            : stub_(ImageDetection::NewStub(channel)) {}

    // Assembles the client's payload and sends it to the server.
    void SendImage(image *Image) {
        // Data we are sending to the server.
        KeyFrame frame;
        frame.set_width(Image->w);
        frame.set_height(Image->h);
        frame.set_numChannels(Image->c);
        for (int i = 0; i < (Image->h * Image->w * Image->c); i++)
            frame.add_data(Image->data[i]);

        // Call object to store rpc data
        AsyncClientCall* call = new AsyncClientCall;

        // stub_->PrepareAsyncSayHello() creates an RPC object, returning
        // an instance to store in "call" but does not actually start the RPC
        // Because we are using the asynchronous API, we need to hold on to
        // the "call" instance in order to get updates on the ongoing RPC.
        call->async_reader =
            stub_->PrepareAsyncRequestDetection(&call->context, frame, &cq_);

        //Send any associated Metadata
        // call->reader_writer->

        // StartCall initiates the RPC call
        // Tag: the memory address of the call object.
        call->async_reader->StartCall((void *)&call);

        // Request that, upon completion of the RPC, "reply" be updated with the
        // server's response; "status" with the indication of whether the operation
        // was successful. Tag the request with the memory address of the call object.
        call->async_reader->Finish(&call->status, (void*)call);

    }

    // Loop while listening for completed responses.
    // Prints out the response from the server.
    void AsyncCompleteRpc() {
        void* got_tag;
        bool ok = false;

        // Block until the next result is available in the completion queue "cq".
        while (cq_.Next(&got_tag, &ok)) {

            // The tag in this example is the memory location of the call object
            AsyncClientCall* call = static_cast<AsyncClientCall*>(got_tag);

            // Verify that the request was completed successfully. Note that "ok"
            // corresponds solely to the request for updates introduced by Finish().
            GPR_ASSERT(ok);

            std::vector<DetectedObjects> detectedObjects;
            DetectedObjects object;
            while(async_reader.Read(&object, got_tag)) {
                detectedObjects.push_back(object);
            }

            if (call->status.ok()) {
                // print out what we received...
                std::cout   << "Objects detected: " << std::endl;
                for (auto i = detectedObjects.begin(); i < detectedObjects.end(); i++) {
                    std::cout   << i.bbox().x() << ", "
                                << i.bbox().y() << ", "
                                << i.bbox().w() << ", "
                                << i.bbox().h() << ", "
                                << i.classes() << ", ";
                    for (auto j = 0; j < i.prob_size(); j++) {
                        std::cout << i.prob(j);
                    }
                    std::cout << std::endl;
                }
            } else {
                std::cout << "RPC failed" << std::endl;
            }

            // Once we're complete, deallocate the call object.
            delete call;
        }
    }

  private:

    // struct for keeping state and data information
    struct AsyncClientCall {
        // Container for the data we expect from the server.
        DetectedObjects detectedObjects;

        // Context for the client. It could be used to convey extra information to
        // the server and/or tweak certain RPC behaviors.
        ClientContext context;

        // Storage for the status of the RPC upon completion.
        Status status;

        std::unique_ptr<ClientAsyncReader<DetectedObjects>> async_reader;
    };

    // Out of the passed in Channel comes the stub, stored here, our view of the
    // server's exposed services.
    std::unique_ptr<ImageDetection::Stub> stub_;

    // TODO: add deadlines to the tasks.
    // The producer-consumer queue we use to communicate asynchronously with the
    // gRPC runtime.
    CompletionQueue cq_;
};

int main(int argc, char** argv) {

    // Instantiate the client. It requires a channel, out of which the actual RPCs
    // are created. This channel models a connection to an endpoint (in this case,
    // localhost at port 50051). We indicate that the channel isn't authenticated
    // (use of InsecureChannelCredentials()).
    // TODO: Replace with an authenticated channel
    ImageDetectionClient detectionClient(grpc::CreateChannel(
            "localhost:50051", grpc::InsecureChannelCredentials()));

    // Spawn reader thread that loops indefinitely
    std::thread thread_ = std::thread(&ImageDetectionClient::AsyncCompleteRpc, 
        &detectionClient);

    // Do any associated setup (metadata etc.)

    // Open the input video file

    // Decode and obtain KeyFrames

    // The actual RPC call!
    ImageDetection.SendImage(&image);

    std::cout << "Press control-c to quit" << std::endl << std::endl;
    thread_.join();  //blocks forever

    return 0;
}
