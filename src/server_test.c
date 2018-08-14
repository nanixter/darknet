#include "darknet.h"

network *net;
float *predictions;
float *average;
layer l;

int size_network()
{
	int count = 0;
	for(int i = 0; i < net->n; ++i){
		layer l = net->layers[i];
		if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
			count += l.outputs;
		}
	}
	return count;
}
void remember_network()
{
	int count = 0;
	for(int i = 0; i < net->n; ++i){
		layer l = net->layers[i];
		if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
			std::memcpy(predictions + count, net->layers[i].output, sizeof(float) * l.outputs);
			count += l.outputs;
		}
	}
}

detection *average_predictions(int *nboxes, int height, int width)
{
	int i, j;
	int count = 0;
	fill_cpu(numNetworkOutputs, 0, average, 1);
	axpy_cpu(numNetworkOutputs, 1./3, predictions, 1, average, 1);

	for(i = 0; i < net->n; ++i){
		layer l = net->layers[i];
		if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
			std::memcpy(l.output, average + count, sizeof(float) * l.outputs);
			count += l.outputs;
		}
	}
	return get_network_boxes(net, width, height, 0.5, 0.5, 0, 1, nboxes);
		}

void test_server_detection(network *network, float *pred, float *avg, const char *videoFile) {
	printf("TEST\n");
	net = network;
	predictions = pred;
	average = avg;
	l = net->layers[net->n-1];

	char *videoFile = argv[4];
	CvCapture * cap; = cvCreateFileCapture(videoFile);

	image newImage;
	image newImage_letterboxed;
	detection *dets = nullptr;
	int nboxes = 0;

	int status = fill_image_from_stream(cap, newImage);
	while (status != 0) {
		letterbox_image_into(newImage, net->w, net->h, newImage_letterboxed);

		/* Now we finally run the actual network	*/
		network_predict(net, newImage_letterboxed.data);
		remember_network(net);
		dets = average_predictions(&nboxes, newImage.h, newImage.w);

		printf("nboxes = %d\n", nboxes);
		status = fill_image_from_stream(cap, newImage);
	}
}