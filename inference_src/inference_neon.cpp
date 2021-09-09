#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <math.h>
#include <iomanip>

#include "AudioFile.h"

#include "model.h"

#include <fftw3.h>

#include <chrono>

#include "simd_math.h"

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

void fully_connected_simd(float32_t* input, float32_t* output, int layer_size, const float32_t(*weights_arr)[513], const float32_t* biases) {
	nfloat32x4_t sum;
	nfloat32x4_t weights;
	nfloat32x4_t values;
	float32_t sum_res;
	std::div_t divresult;
	divresult = std::div(layer_size, 4);
	for (int idx = 0; idx < layer_size; idx++) {
		sum = reset_vector();
		sum_res = 0;
		for (int j = 0; j < 512; j += 4) {
			weights = load_vector(weights_arr[idx], j);
			values = load_vector(input, j);
			sum = multiply_w_acc(sum, weights, values);
		}
		sum_res = sum_vector(sum);
		sum_res += biases[idx];
		for (int k = (divresult.quot * 4); k < layer_size; k++) { // process leftovers
			sum_res += input[k] * weights_arr[idx][k];
		}
		output[idx] = ReLU(sum_res);
	}
}

float32_t output_layer(float32_t* input, int layer_size, const float32_t* weights_arr, float32_t final_bias) {
	nfloat32x4_t sum;
	nfloat32x4_t weights;
	nfloat32x4_t values;
	float32_t sum_res = 0;
	sum = reset_vector();
	for (int j = 0; j < 512; j += 4) {
		weights = load_vector(weights_arr, j);
		values = load_vector(input, j);
		sum = multiply_w_acc(sum, weights, values);
	}
	sum_res = sum_vector(sum);
	sum_res += input[512] * weights_arr[512];
	sum_res += final_bias;
	return sigmoid(sum_res);
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

	double* in;
	in = (double*)fftw_alloc_real(N);
	memset(in, 0, N * sizeof(in[0]));

	double* hann;
	hann = (double*)malloc(sizeof(double) * frameSize);
	for (int i = 0; i < frameSize; i++) {
		hann[i] = 0.5 * (1 - cos(2 * M_PI * i / (frameSize - 1)));
	}

	float* bins;
	bins = (float*)malloc(sizeof(float) * numBins);

	fftw_complex* out;
	out = (fftw_complex*)fftw_alloc_complex(numBins);

	fftw_plan p;
	p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE); // https://www.fftw.org/fftw3_doc/Planner-Flags.html

	AudioFile<double> audioFile;
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

		fftw_execute(p);

		for (int k = 0; k < numBins; k++) {
			double realVal = out[k][0];
			double imagVal = out[k][1];
			bins[k] = float(realVal * realVal + imagVal * imagVal);
		}

		fully_connected_simd(bins, output_simd, numBins, weights0, biases0);
		fully_connected_simd(output_simd, bins, numBins, weights1, biases1);
		fully_connected_simd(bins, output_simd, numBins, weights2, biases2);
		float32_t res = output_layer(output_simd, numBins, weights3, biases3[0]);
	}

	auto t2 = high_resolution_clock::now();

	auto ms_int = duration_cast<milliseconds>(t2 - t1);

	duration<double, std::milli> ms_double = t2 - t1;

	std::cout << ms_int.count() << "ms\n";
	std::cout << ms_double.count() / frameNum << "ms/frame\n";

	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);
	free(bins);
	return 0;
}
