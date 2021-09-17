#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <math.h>
#include <iomanip>

#include "AudioFile/AudioFile.h"

#include <fftw3.h>

#include <chrono>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/optional_debug_tools.h"

#define TFLITE_MINIMAL_CHECK(x)                                  \
	if (!(x)) {                                                  \
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
		exit(1);                                                 \
}

class FlatBufferModel {
	// Build a model based on a file. Return a nullptr in case of failure.
	static std::unique_ptr<FlatBufferModel> BuildFromFile(
			const char* filename,
			tflite::ErrorReporter* error_reporter);

	// Build a model based on a pre-loaded flatbuffer. The caller retains
	// ownership of the buffer and should keep it alive until the returned object
	// is destroyed. Return a nullptr in case of failure.
	static std::unique_ptr<FlatBufferModel> BuildFromBuffer(
			const char* buffer,
			size_t buffer_size,
			tflite::ErrorReporter* error_reporter);
};

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

	float* in;
	in = (float*)fftwf_malloc(sizeof(float) * N);
	memset(in, 0, N * sizeof(in[0]));

	float* hann;
	hann = (float*)malloc(sizeof(float) * frameSize);
	for (int i = 0; i < frameSize; i++) {
		hann[i] = 0.5 * (1 - cos(2 * M_PI * i / (frameSize - 1)));
	}

	float* bins;
	bins = (float*)malloc(sizeof(float) * numBins);

	fftwf_complex* out;
	out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * numBins);

	fftwf_plan p;
	p = fftwf_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

	AudioFile<float> audioFile;
	audioFile.load(path);
	audioFile.printSummary();

	int numSamples = audioFile.getNumSamplesPerChannel();

	int frameNum = int(floor(numSamples / overlap)) - 1;

	// Load the model
	std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("model.tflite");
	TFLITE_MINIMAL_CHECK(model != nullptr);

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder builder(*model, resolver);
	std::unique_ptr<tflite::Interpreter> interpreter;
	builder(&interpreter);
	TFLITE_MINIMAL_CHECK(interpreter != nullptr);

	TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

	float* input_data_ptr = interpreter->typed_input_tensor<float>(0);
	float* output_data_ptr = interpreter->typed_output_tensor<float>(0);

	auto t1 = high_resolution_clock::now();

	for (int i = 0; i < frameNum; i++) {
		int offset = int(i * step);
		memcpy(in, &audioFile.samples[channel][offset], frameSize * sizeof(audioFile.samples[channel][0]));

		for (int j = 0; j < frameSize; j++) {
			in[j] = in[j] * hann[j];
		}

		fftwf_execute(p);

		for (int k = 0; k < (N / 2) + 1; k++) {
			float realVal = out[k][0];
			float imagVal = out[k][1];
			bins[k] = sqrt(realVal * realVal + imagVal * imagVal);
		}

		// Fill `input`.
		float* input_data_ptr = interpreter->typed_input_tensor<float>(0);
		for (int i = 0; i < numBins; i++) {
			*(input_data_ptr) = (float)bins[i];
			//std::cout << i << ": " << *(input_data_ptr) << std::endl;
			input_data_ptr++;
		}

		TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

		//std::cout << *output_data_ptr << std::endl;
	}
	
	auto t2 = high_resolution_clock::now();

	/* Getting number of milliseconds as an integer. */
	auto ms_int = duration_cast<milliseconds>(t2 - t1);

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = t2 - t1;

	std::cout << ms_int.count() << "ms\n";
	std::cout << ms_double.count()/frameNum << "ms/frame\n";

	//txt.close();
	fftwf_destroy_plan(p);
	fftwf_free(in);
	fftwf_free(out);
	free(bins);
	return 0;
}