#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace sovereigngrad {
namespace simd {

// 1. Vectorized Elementwise Addition: dst[i] = a[i] + b[i]
inline void vec_add(float* dst, const float* a, const float* b, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(dst + i, vr);
    }
#endif
    for (; i < n; ++i) {
        dst[i] = a[i] + b[i];
    }
}

// 2. Vectorized Elementwise Multiplication: dst[i] = a[i] * b[i]
inline void vec_mul(float* dst, const float* a, const float* b, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(dst + i, vr);
    }
#endif
    for (; i < n; ++i) {
        dst[i] = a[i] * b[i];
    }
}

// 3. Vectorized ReLU: dst[i] = max(0, src[i])
inline void vec_relu(float* dst, const float* src, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
    __m256 vzero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 vs = _mm256_loadu_ps(src + i);
        __m256 vr = _mm256_max_ps(vzero, vs);
        _mm256_storeu_ps(dst + i, vr);
    }
#endif
    for (; i < n; ++i) {
        dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
    }
}

// 4. Vectorized Fast GELU (tanh approximation)
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline void vec_gelu(float* dst, const float* src, int64_t n) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;

    int64_t i = 0;
#if defined(__AVX2__)
    __m256 v_half = _mm256_set1_ps(0.5f);
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_sqrt_2_pi = _mm256_set1_ps(sqrt_2_over_pi);
    __m256 v_coeff = _mm256_set1_ps(coeff);

    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        __m256 inner = _mm256_add_ps(x, _mm256_mul_ps(v_coeff, x3));
        __m256 arg = _mm256_mul_ps(v_sqrt_2_pi, inner);

        // Compute tanh per lane with fast scalar fallback / vectorized clamp
        alignas(32) float args[8];
        alignas(32) float tanhs[8];
        _mm256_store_ps(args, arg);
        for (int k = 0; k < 8; ++k) {
            tanhs[k] = std::tanh(args[k]);
        }
        __m256 v_tanh = _mm256_load_ps(tanhs);
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(v_half, x), _mm256_add_ps(v_one, v_tanh));
        _mm256_storeu_ps(dst + i, res);
    }
#endif
    for (; i < n; ++i) {
        float x = src[i];
        float cube = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * cube);
        dst[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

// 5. Vectorized Fast SiLU / Swish: x / (1 + exp(-x))
inline void vec_silu(float* dst, const float* src, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        float x = src[i];
        dst[i] = x / (1.0f + std::exp(-x));
    }
}

// 6. Register-Blocked Micro-Kernel for CPU Matrix Multiplication
// Computes 4x8 block: C[i..i+3, j..j+7] += A[i..i+3, k] * B[k, j..j+7]
inline void gemm_microkernel_4x8(const float* A, const float* B, float* C,
                                int64_t lda, int64_t ldb, int64_t ldc, int64_t K) {
#if defined(__AVX2__)
    __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);

    for (int64_t k = 0; k < K; ++k) {
        __m256 b_row = _mm256_loadu_ps(B + k * ldb);

        __m256 a0 = _mm256_set1_ps(A[0 * lda + k]);
        __m256 a1 = _mm256_set1_ps(A[1 * lda + k]);
        __m256 a2 = _mm256_set1_ps(A[2 * lda + k]);
        __m256 a3 = _mm256_set1_ps(A[3 * lda + k]);

        c0 = _mm256_fmadd_ps(a0, b_row, c0);
        c1 = _mm256_fmadd_ps(a1, b_row, c1);
        c2 = _mm256_fmadd_ps(a2, b_row, c2);
        c3 = _mm256_fmadd_ps(a3, b_row, c3);
    }

    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
#else
    for (int64_t i = 0; i < 4; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float a = A[i * lda + k];
            for (int64_t j = 0; j < 8; ++j) {
                C[i * ldc + j] += a * B[k * ldb + j];
            }
        }
    }
#endif
}

// 7. Register-Blocked Micro-Kernel for CPU Matrix Multiplication (8x8)
// Computes 8x8 block: C[i..i+7, j..j+7] += A[i..i+7, k] * B[k, j..j+7]
inline void gemm_microkernel_8x8(const float* A, const float* B, float* C,
                                int64_t lda, int64_t ldb, int64_t ldc, int64_t K) {
#if defined(__AVX2__)
    __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 c4 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 c5 = _mm256_loadu_ps(C + 5 * ldc);
    __m256 c6 = _mm256_loadu_ps(C + 6 * ldc);
    __m256 c7 = _mm256_loadu_ps(C + 7 * ldc);

    int64_t k = 0;
    // Unroll by 2 to expose more ILP and reduce loop overhead
    for (; k + 1 < K; k += 2) {
        // First K
        __m256 b0 = _mm256_loadu_ps(B + k * ldb);
        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * lda + k]), b0, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * lda + k]), b0, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * lda + k]), b0, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * lda + k]), b0, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(A[4 * lda + k]), b0, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(A[5 * lda + k]), b0, c5);
        c6 = _mm256_fmadd_ps(_mm256_set1_ps(A[6 * lda + k]), b0, c6);
        c7 = _mm256_fmadd_ps(_mm256_set1_ps(A[7 * lda + k]), b0, c7);

        // Second K
        __m256 b1 = _mm256_loadu_ps(B + (k + 1) * ldb);
        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * lda + (k + 1)]), b1, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * lda + (k + 1)]), b1, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * lda + (k + 1)]), b1, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * lda + (k + 1)]), b1, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(A[4 * lda + (k + 1)]), b1, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(A[5 * lda + (k + 1)]), b1, c5);
        c6 = _mm256_fmadd_ps(_mm256_set1_ps(A[6 * lda + (k + 1)]), b1, c6);
        c7 = _mm256_fmadd_ps(_mm256_set1_ps(A[7 * lda + (k + 1)]), b1, c7);
    }
    
    // Tail
    for (; k < K; ++k) {
        __m256 b_row = _mm256_loadu_ps(B + k * ldb);
        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A[0 * lda + k]), b_row, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A[1 * lda + k]), b_row, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(A[2 * lda + k]), b_row, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(A[3 * lda + k]), b_row, c3);
        c4 = _mm256_fmadd_ps(_mm256_set1_ps(A[4 * lda + k]), b_row, c4);
        c5 = _mm256_fmadd_ps(_mm256_set1_ps(A[5 * lda + k]), b_row, c5);
        c6 = _mm256_fmadd_ps(_mm256_set1_ps(A[6 * lda + k]), b_row, c6);
        c7 = _mm256_fmadd_ps(_mm256_set1_ps(A[7 * lda + k]), b_row, c7);
    }

    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
    _mm256_storeu_ps(C + 4 * ldc, c4);
    _mm256_storeu_ps(C + 5 * ldc, c5);
    _mm256_storeu_ps(C + 6 * ldc, c6);
    _mm256_storeu_ps(C + 7 * ldc, c7);
#else
    for (int64_t i = 0; i < 8; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float a = A[i * lda + k];
            for (int64_t j = 0; j < 8; ++j) {
                C[i * ldc + j] += a * B[k * ldb + j];
            }
        }
    }
#endif
}

} // namespace simd
} // namespace sovereigngrad
