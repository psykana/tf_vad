#ifndef __ARM_NEON__
using float32_t = float;
typedef struct nfloat32x4_s {
	float32_t a;
	float32_t b;
	float32_t c;
	float32_t d;
} nfloat32x4_t;
#else
#include <arm_neon.h>
typedef float32x4_t nfloat32x4_t;
#endif

static inline nfloat32x4_t reset_vector() {
	nfloat32x4_t vector;
#ifdef __ARM_NEON__
	vector = vmovq_n_f32(0);
#else
	vector = { 0.0f, 0.0f, 0.0f, 0.0f };
#endif
	return vector;
}

static inline nfloat32x4_t load_vector(const float32_t* input, int index) {
	nfloat32x4_t vector;
#ifdef __ARM_NEON__
	vector = vld1q_f32(input + index);
#else
	vector = { input[index], input[index + 1], input[index + 2], input[index + 3] };
#endif
	return vector;
}

static inline nfloat32x4_t multiply_w_acc(nfloat32x4_t sum, nfloat32x4_t weights, nfloat32x4_t values) {
	nfloat32x4_t vector;
#ifdef __ARM_NEON__
	arr = vfmaq_f32(sum, weights, values);
#else
	vector.a = sum.a + (values.a * weights.a);
	vector.b = sum.b + (values.b * weights.b);
	vector.c = sum.c + (values.c * weights.c);
	vector.d = sum.d + (values.d * weights.d);
#endif
	return vector;
}

static inline float32_t sum_vector(nfloat32x4_t vector) {
	float32_t sum;
#ifdef __ARM_NEON__
	sum = vaddvq_f32(vector);
#else
	sum = vector.a + vector.b + vector.c + vector.d;
#endif
	return sum;
}