#include <iostream>

#include "api.h"

#include <math.h>
#include <chrono>

#include "AudioFile/AudioFile.h"

int main() {

	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;
	
	const int overlap = 256;
	const int frameSize = 512;
	const int step = frameSize - overlap;
	const int channel = 0;
	const std::string path = "SA1.WAV";
	
	AudioFile<float> audioFile;
	audioFile.load(path);
	audioFile.printSummary();

	int numSamples = audioFile.getNumSamplesPerChannel();
	int frameNum = int(floor(numSamples / overlap)) - 1;

	size_t params_bytes;
	size_t coeffs_bytes;

	std::cout << "get size: " << effect_control_get_sizes(&params_bytes, &coeffs_bytes) << std::endl;
	
	std::cout << "coeff size: " << coeffs_bytes << " " << "param size: " << params_bytes << std::endl;
	
	void *params = malloc(params_bytes);
	void *coeffs = malloc(coeffs_bytes);
	
	std::cout << "init: " << effect_control_initialize(params, coeffs, 16000) << std::endl;
	
	std::cout << "set param: " << effect_set_parameter(params, 0, 0.7) << std::endl;
	
	std::cout << "update coeffs: " << effect_update_coeffs(params, coeffs) << std::endl;
	
	float audio[512];
	int result;
	
	auto t1 = high_resolution_clock::now();
	
	for (int i = 0; i < frameNum; i++) {
		int offset = int(i * step);
		memcpy(audio, &audioFile.samples[channel][offset], frameSize * sizeof(audioFile.samples[channel][0]));
		effect_process(coeffs, nullptr, (void*)audio, 512, &result);
		//std::cout << i  << ": " << result << std::endl;
	}
	
	auto t2 = high_resolution_clock::now();

	/* Getting number of milliseconds as an integer. */
	auto ms_int = duration_cast<milliseconds>(t2 - t1);

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = t2 - t1;

	std::cout << ms_int.count() << "ms\n";
	std::cout << ms_double.count()/frameNum << "ms/frame\n";
	
	return 0;
}
