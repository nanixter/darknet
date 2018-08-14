#include "darknet.h"
#include "image.h"

network *net;
float *predictions;
float *average;
int numNetworkOutputs;

int size_network2()
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
void remember_network2()
{
	int count = 0;
	for(int i = 0; i < net->n; ++i){
		layer l = net->layers[i];
		if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
			memcpy(predictions + count, net->layers[i].output, sizeof(float) * l.outputs);
			count += l.outputs;
		}
	}
}

detection *average_predictions(int *nboxes, int height, int width)
{
	int i;
	int count = 0;
	fill_cpu(numNetworkOutputs, 0, average, 1);
	axpy_cpu(numNetworkOutputs, 1./3, predictions, 1, average, 1);

	for(i = 0; i < net->n; ++i){
		layer l = net->layers[i];
		if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
			memcpy(l.output, average + count, sizeof(float) * l.outputs);
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
	numNetworkOutputs = size_network2();
	CvCapture * cap = cvCreateFileCapture(videoFile);

	image newImage;
	image newImage_letterboxed;
	detection *dets = NULL;
	int nboxes = 0;

	newImage = get_image_from_stream(cap);
	while (newImage.data != 0) {
		letterbox_image_into(newImage, net->w, net->h, newImage_letterboxed);

		/* Now we finally run the actual network	*/
		network_predict(net, newImage_letterboxed.data);
		remember_network2();
		dets = average_predictions(&nboxes, newImage.h, newImage.w);

		printf("nboxes = %d\n", nboxes);
		newImage = get_image_from_stream(cap);
	}
}
