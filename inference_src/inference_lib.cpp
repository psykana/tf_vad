#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

#include <iostream>
#include <string.h>
#include <math.h>
#include <fftw3.h>

#include "model.h"
#include "simd_math.h"

struct Layer {
	float* weights;
	float* biases;
	int numNeurons;
	int numInput;
	int vectorizedEnd;

	float (*activation)(float);

	float* weightsAt(unsigned row, unsigned column = 0) {
		return (weights + numInput * row + column);
	}
};

struct Mega {
	const int layer_num = LAYERS;
	Layer layers[LAYERS];
	const float* hann;

	float in[1024];
	fftwf_complex out[513];
	float temp1[513];
	float temp2[513];

	fftwf_plan p;
	
	float smoothingThreshold;
};

inline float ReLU(float input) {
	if (input > 0) {
		return input;
	}
	else {
		return 0.0f;
	}
}

inline float sigmoid(float input) {
	return 1.0f / (1.0f + exp((-1.0f) * input));
}

void fully_connected(float* input, float* output, Layer layer) {
	nfloat32x4_t sum;
	nfloat32x4_t weights;
	nfloat32x4_t values;
	float32_t sum_res;
	for (int idx = 0; idx < layer.numNeurons; idx++) {
		sum = reset_vector();
		sum_res = 0;
		for (int j = 0; j < layer.vectorizedEnd; j += 4) {
			weights = load_vector(layer.weightsAt(idx), j);
			values = load_vector(input, j);
			sum = multiply_w_acc(sum, weights, values);
		}
		sum_res = sum_vector(sum);
		sum_res += layer.biases[idx];
		for (int k = layer.vectorizedEnd; k < layer.numInput; k++) {
			sum_res += input[k] * *(layer.weightsAt(idx, k));
		}
		output[idx] = layer.activation(sum_res);
	}
}

int32_t effect_control_get_sizes(size_t* params_bytes, size_t* coeffs_bytes) {
	*coeffs_bytes = sizeof(Mega);
	*params_bytes = sizeof(float);
	return 1;
}

int32_t effect_control_initialize(void* params, void* coeffs, uint32_t sample_rate) {
	Mega* str = (Mega*)coeffs;

	memset(str->in, 0, 1024 * sizeof(str->in[0]));
	str->hann = (float*)hannWindow;

	str->layers[0] = { (float*)weights0, (float*)biases0, weights0_xlen, weights0_ylen, weights0_ylen - weights0_ylen % 4, &ReLU };
	str->layers[1] = { (float*)weights1, (float*)biases1, weights1_xlen, weights1_ylen, weights1_ylen - weights1_ylen % 4, &ReLU };
	str->layers[2] = { (float*)weights2, (float*)biases2, weights2_xlen, weights2_ylen, weights2_ylen - weights2_ylen % 4, &ReLU };
	str->layers[3] = { (float*)weights3, (float*)biases3, weights3_xlen, weights3_ylen, weights3_ylen - weights3_ylen % 4, &sigmoid };
	str->p = fftwf_plan_dft_r2c_1d(N, str->in, str->out, FFTW_ESTIMATE);

	return 1;
}

int32_t effect_set_parameter(void* params, int32_t id, float value) {
	if (id == 0) {
		float* smoothingThreshold = (float*)params;
		*smoothingThreshold = value;
		return 1;
	} else {
		return 0;
	}
}

int32_t effect_update_coeffs(void const* params, void* coeffs) {
	Mega* coeffs_ptr = (Mega*)coeffs;
	float* smoothing = (float*)params;
	coeffs_ptr->smoothingThreshold = *smoothing;
	return 1;
}

int32_t effect_process(void const* coeffs, void* states, void* audio, size_t samples_count, int* result) {

	Mega* coeffs_ptr = (Mega*)coeffs;
	float* audio_ptr = (float*)audio;

	float* in = (float*)coeffs_ptr->in;
	float* hann = (float*)coeffs_ptr->hann;

	memcpy(in, audio_ptr, 512 * sizeof(float));

	for (int j = 0; j < 512; j++) {
		in[j] = in[j] * hann[j];
	}

	fftwf_execute(coeffs_ptr->p);

	for (int i = 0; i < numBins; i++) {
		float realVal = coeffs_ptr->out[i][0];
		float imagVal = coeffs_ptr->out[i][1];
		coeffs_ptr->temp1[i] = float(realVal * realVal + imagVal * imagVal);
	}

	fully_connected(coeffs_ptr->temp1, coeffs_ptr->temp2, coeffs_ptr->layers[0]);
	fully_connected(coeffs_ptr->temp2, coeffs_ptr->temp1, coeffs_ptr->layers[1]);
	fully_connected(coeffs_ptr->temp1, coeffs_ptr->temp2, coeffs_ptr->layers[2]);
	fully_connected(coeffs_ptr->temp2, coeffs_ptr->temp1, coeffs_ptr->layers[3]);

	float modelOutput = coeffs_ptr->temp1[0];
	if (modelOutput > coeffs_ptr->smoothingThreshold) {
		modelOutput = 1;
	}
	else {
		modelOutput = 0;
	}

	*result = modelOutput;

	return 1;
}

