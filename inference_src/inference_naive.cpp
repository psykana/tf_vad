#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <math.h>
#include <iomanip>

#include "AudioFile/AudioFile.h"

#include "model.h"

#include <fftw3.h>

#include <chrono>

inline float ReLU(float input) {
	if (input > 0) {
		return input;
	}
	else {
		return 0.0;
	}
}

inline float sigmoid(float input) {
	return 1.0f / (1.0f + exp((-1.0f) * input));
}

void fully_connected(float* input, float* output, int layer_size, const float(*weights)[513], const float* biases) {
	float sum;
	for (int idx = 0; idx < layer_size; idx++) {
		sum = 0;
		for (int x = 0; x < layer_size; x++) {
			sum += input[x] * weights[idx][x];
		}
		sum += biases[idx];
		output[idx] = ReLU(sum);
	}
}

void output_layer(float* input, float* output, int layer_size, const float(*weights), const float final_bias) {
	float sum = 0;
	for (int x = 0; x < layer_size; x++) {
		sum += input[x] * weights[x];
	}
	sum += final_bias;
	output[0] = sigmoid(sum);
}

int main() {

	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;

	const std::string path = "SA1.WAV";
	const int N = 1024;
	const int numBins = (N / 2) + 1;
	const int frameSize = 512;
	const int overlap = 256;
	const int step = frameSize - overlap;
	const int channel = 0;

	float* output;
	output = (float*)malloc(sizeof(float) * numBins);
	memset(output, 0, numBins * sizeof(output[0]));

	float* output_simd;
	output_simd = (float*)malloc(sizeof(float) * numBins);
	memset(output_simd, 0, numBins * sizeof(output_simd[0]));

	float* in;
	in = (float*)fftwf_alloc_real(N);
	memset(in, 0, N * sizeof(in[0]));

	float* hann;
	hann = (float*)malloc(sizeof(float) * frameSize);
	for (int i = 0; i < frameSize; i++) {
		hann[i] = 0.5 * (1 - cos(2 * M_PI * i / (frameSize - 1)));
	}

	float* bins;
	bins = (float*)malloc(sizeof(float) * numBins);

	fftwf_complex* out;
	out = (fftwf_complex*)fftwf_alloc_complex(numBins);

	fftwf_plan p;
	p = fftwf_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

	AudioFile<float> audioFile;
	audioFile.load(path);
	audioFile.printSummary();

	int numSamples = audioFile.getNumSamplesPerChannel();

	int frameNum = int(floor(numSamples / overlap)) - 1;

	auto t1 = high_resolution_clock::now();

	for (int i = 0; i < frameNum; i++) {
		int offset = int(i * step);
		memcpy(in, &audioFile.samples[channel][offset], frameSize * sizeof(audioFile.samples[channel][0]));

		for (int j = 0; j < frameSize; j++) {
			in[j] = in[j] * hann[j];
		}

		fftwf_execute(p);

		for (int k = 0; k < numBins; k++) {
			float realVal = out[k][0];
			float imagVal = out[k][1];
			bins[k] = sqrt(realVal * realVal + imagVal * imagVal);
		}

		fully_connected(bins, output_simd, numBins, weights0, biases0);
		fully_connected(output_simd, bins, numBins, weights1, biases1);
		fully_connected(bins, output_simd, numBins, weights2, biases2);
		output_layer(output_simd, bins, numBins, weights3, biases3[0]);
		//std::cout << i << ":  " << bins[0] << std::endl;

	}

	auto t2 = high_resolution_clock::now();

	/* Getting number of milliseconds as an integer. */
	auto ms_int = duration_cast<milliseconds>(t2 - t1);

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = t2 - t1;

	std::cout << ms_int.count() << "ms\n";
	std::cout << ms_double.count()/frameNum << "ms/frame\n";

	fftwf_destroy_plan(p);
	fftwf_free(in);
	fftwf_free(out);
	free(bins);
	return 0;
}
