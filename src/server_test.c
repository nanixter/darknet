#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

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
	axpy_cpu(numNetworkOutputs, 1.0, predictions, 1, average, 1);

	for(i = 0; i < net->n; ++i){
		layer l = net->layers[i];
		if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
			memcpy(l.output, average + count, sizeof(float) * l.outputs);
			count += l.outputs;
		}
	}
	return get_network_boxes(net, width, height, 0.5, 0.5, 0, 1, nboxes);
}

void test_server_detection(const char *cfgfile, const char *weightfile, const char *videoFile) {
	printf("TEST\n");
	net = load_network(cfgfile, weightfile, 0);
	set_batch_network(net, 1);
	numNetworkOutputs = size_network2();
	predictions = (float *)malloc(sizeof(float)*numNetworkOutputs);
	average = (float *)malloc(sizeof(float)*numNetworkOutputs);
	CvCapture * cap = cvCreateFileCapture(videoFile);

	image newImage;
	image newImage_letterboxed;
	detection *dets = NULL;
	int nboxes = 0;

	newImage = get_image_from_stream(cap);
	newImage_letterboxed = letterbox_image(newImage, net->w, net->h);
	while (newImage.data != 0) {
		/* Now we finally run the actual network	*/
		network_predict(net, newImage_letterboxed.data);
		remember_network2();
		dets = average_predictions(&nboxes, newImage.h, newImage.w);

		printf("nboxes = %d\n", nboxes);
		newImage = get_image_from_stream(cap);
		letterbox_image_into(newImage, net->w, net->h, newImage_letterboxed);
	}
}
