// Winograd fast inner product applied to a neural-network linear layer:
//
//     Y = X * W^T + b
//
// X : m x n   (batch size m, in_features n), row-major
// W : p x n   (out_features p, in_features n), row-major   (PyTorch layout)
// b : p
// Y : m x p, row-major
#pragma once

#include <cstddef>

#include "winograd.hpp"  // OpCount

namespace winograd {

// Reference implementation: one dot product per output element.
void linear_naive(const double* X, const double* W, const double* b,
                  double* Y, std::size_t m, std::size_t n, std::size_t p) noexcept;

// Winograd form, self-products of W and b recomputed on every call.
// Suitable for training, where W changes every step.
void linear_winograd(const double* X, const double* W, const double* b,
                     double* Y, std::size_t m, std::size_t n, std::size_t p) noexcept;

// wterm[j] = sum_k W[j,2k]*W[j,2k+1] - b[j]
// Precompute once when W and b are fixed (inference).
void linear_weight_terms(const double* W, const double* b, double* wterm,
                         std::size_t p, std::size_t n) noexcept;

// Same as linear_winograd, but the weight/bias terms are supplied by the
// caller.  Only the X self-products and the pairwise products are timed.
void linear_winograd_cached(const double* X, const double* W,
                            const double* wterm, double* Y,
                            std::size_t m, std::size_t n,
                            std::size_t p) noexcept;

// Analytical operation counts.
OpCount linear_ops_naive(std::size_t m, std::size_t n, std::size_t p) noexcept;
OpCount linear_ops_winograd(std::size_t m, std::size_t n, std::size_t p) noexcept;
OpCount linear_ops_winograd_cached(std::size_t m, std::size_t n,
                                   std::size_t p) noexcept;
OpCount linear_weight_term_ops(std::size_t p, std::size_t n) noexcept;

}  // namespace winograd
