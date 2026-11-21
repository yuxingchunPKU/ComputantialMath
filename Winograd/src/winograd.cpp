#include "winograd.hpp"

#include <vector>

namespace winograd {

// ---------------------------------------------------------------------
// dot products
// ---------------------------------------------------------------------
double dot_naive(const double* a, const double* b, std::size_t n) noexcept {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

double self_product(const double* a, std::size_t n) noexcept {
    double x = 0.0;
    const std::size_t half = n / 2;
    for (std::size_t k = 0; k < half; ++k) {
        x += a[2 * k] * a[2 * k + 1];
    }
    return x;
}

double dot_winograd_precomputed(const double* a, const double* b,
                                std::size_t n, double x, double y) noexcept {
    const std::size_t half = n / 2;
    double s = 0.0;
    for (std::size_t k = 0; k < half; ++k) {
        s += (a[2 * k] + b[2 * k + 1]) * (a[2 * k + 1] + b[2 * k]);
    }
    s -= x + y;
    if (n % 2 != 0) {
        s += a[n - 1] * b[n - 1];
    }
    return s;
}

double dot_winograd(const double* a, const double* b, std::size_t n) noexcept {
    // Both self-products are recomputed here; the amortised version is
    // dot_winograd_precomputed().
    const std::size_t half = n / 2;
    double x = 0.0;
    double y = 0.0;
    for (std::size_t k = 0; k < half; ++k) {
        x += a[2 * k] * a[2 * k + 1];
        y += b[2 * k] * b[2 * k + 1];
    }
    return dot_winograd_precomputed(a, b, n, x, y);
}

// ---------------------------------------------------------------------
// matrix multiplication
// ---------------------------------------------------------------------
void matmul_naive(const double* A, const double* B, double* C,
                  std::size_t m, std::size_t n, std::size_t p) noexcept {
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                s += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = s;
        }
    }
}

void matmul_winograd(const double* A, const double* B, double* C,
                     std::size_t m, std::size_t n, std::size_t p) noexcept {
    const std::size_t half = n / 2;
    const bool odd = (n % 2 != 0);

    // X_i = sum_k A[i,2k] * A[i,2k+1]        (one per row of A)
    // Y_j = sum_k B[2k,j] * B[2k+1,j]        (one per column of B)
    std::vector<double> X(m, 0.0);
    std::vector<double> Y(p, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        double x = 0.0;
        const double* row = A + i * n;
        for (std::size_t k = 0; k < half; ++k) {
            x += row[2 * k] * row[2 * k + 1];
        }
        X[i] = x;
    }
    for (std::size_t j = 0; j < p; ++j) {
        double y = 0.0;
        for (std::size_t k = 0; k < half; ++k) {
            y += B[(2 * k) * p + j] * B[(2 * k + 1) * p + j];
        }
        Y[j] = y;
    }

    for (std::size_t i = 0; i < m; ++i) {
        const double* row = A + i * n;
        double* crow = C + i * p;
        for (std::size_t j = 0; j < p; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < half; ++k) {
                const double a0 = row[2 * k];
                const double a1 = row[2 * k + 1];
                const double b0 = B[(2 * k) * p + j];
                const double b1 = B[(2 * k + 1) * p + j];
                s += (a0 + b1) * (a1 + b0);
            }
            s -= X[i] + Y[j];
            if (odd) {
                s += row[n - 1] * B[(n - 1) * p + j];
            }
            crow[j] = s;
        }
    }
}

// ---------------------------------------------------------------------
// operation counts
// ---------------------------------------------------------------------
OpCount dot_ops_naive(std::size_t n) noexcept {
    OpCount c;
    c.mul = n;
    c.add = n > 0 ? n - 1 : 0;
    return c;
}

OpCount dot_ops_winograd(std::size_t n) noexcept {
    const std::uint64_t half = n / 2;
    const std::uint64_t odd = n % 2;
    OpCount c;
    // X, Y and the pairwise products: three multiplications per pair.
    c.mul = 3 * half + odd;
    // 2 per pair for the two sums, (half-1) each for X, Y and the
    // accumulation, 2 for "s - x - y", 1 for the odd tail.
    c.add = (half == 0 ? 0 : 5 * half - 1) + odd;
    return c;
}

OpCount matmul_ops_naive(std::size_t m, std::size_t n, std::size_t p) noexcept {
    OpCount c;
    c.mul = static_cast<std::uint64_t>(m) * n * p;
    c.add = static_cast<std::uint64_t>(m) * p * (n > 0 ? n - 1 : 0);
    return c;
}

OpCount matmul_ops_winograd(std::size_t m, std::size_t n,
                            std::size_t p) noexcept {
    const std::uint64_t half = n / 2;
    const std::uint64_t odd = n % 2;
    const std::uint64_t M = m;
    const std::uint64_t P = p;
    OpCount c;
    // row/column self-products + pairwise products of the general term
    c.mul = (M + P + M * P) * half + odd * M * P;
    // row/column self-product reductions + per-entry adds:
    //   2*half pair sums + (half-1) accumulation + 2 for "-X-Y"
    const std::uint64_t red = half > 0 ? half - 1 : 0;
    c.add = (M + P) * red + M * P * (3 * half + 1) + odd * M * P;
    return c;
}

}  // namespace winograd
