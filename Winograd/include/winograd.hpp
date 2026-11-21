// Winograd fast inner-product algorithm.
//
// Pairwise identity (Winograd):
//   a0*b0 + a1*b1 = (a0 + b1) * (a1 + b0) - a0*a1 - b1*b0
//
// Applied to a length-n dot product it trades multiplications for additions.
// The multiplication saving only materialises when the self-products
// X(a) = sum a_{2k} a_{2k+1} and Y(b) = sum b_{2k} b_{2k+1} are shared by
// many inner products, i.e. in batched products such as C = A*B.
#pragma once

#include <cstddef>
#include <cstdint>

namespace winograd {

struct OpCount {
    std::uint64_t mul = 0;
    std::uint64_t add = 0;
};

// ---------------------------------------------------------------------
// 1-D dot products
// ---------------------------------------------------------------------
double dot_naive(const double* a, const double* b, std::size_t n) noexcept;

// Direct application of the Winograd identity, including recomputation of
// X(a) and Y(b).  Supports odd n by handling the last term directly.
double dot_winograd(const double* a, const double* b, std::size_t n) noexcept;

// dot = sum (a_{2k}+b_{2k+1})(a_{2k+1}+b_{2k}) - x - y (+ last term if n odd)
// with x = X(a) and y = Y(b) supplied by the caller.  Useful for showing the
// amortised cost when X(a) / Y(b) are reused.
double dot_winograd_precomputed(const double* a, const double* b,
                                std::size_t n, double x, double y) noexcept;

// X(a) = sum_{k} a_{2k} a_{2k+1}
double self_product(const double* a, std::size_t n) noexcept;

// ---------------------------------------------------------------------
// Matrix multiplication, row-major storage
// ---------------------------------------------------------------------
// C (m x p) = A (m x n) * B (n x p)
void matmul_naive(const double* A, const double* B, double* C,
                  std::size_t m, std::size_t n, std::size_t p) noexcept;

// Winograd form: X per row of A, Y per column of B, n/2 products per entry.
void matmul_winograd(const double* A, const double* B, double* C,
                     std::size_t m, std::size_t n, std::size_t p) noexcept;

// ---------------------------------------------------------------------
// Analytical operation counts (floating-point multiply / add)
// ---------------------------------------------------------------------
OpCount dot_ops_naive(std::size_t n) noexcept;
OpCount dot_ops_winograd(std::size_t n) noexcept;
OpCount matmul_ops_naive(std::size_t m, std::size_t n, std::size_t p) noexcept;
OpCount matmul_ops_winograd(std::size_t m, std::size_t n, std::size_t p) noexcept;

}  // namespace winograd
