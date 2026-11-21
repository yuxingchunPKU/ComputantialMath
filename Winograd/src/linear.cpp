#include "linear.hpp"

#include <vector>

namespace winograd {

void linear_naive(const double* X, const double* W, const double* b,
                  double* Y, std::size_t m, std::size_t n,
                  std::size_t p) noexcept {
    for (std::size_t i = 0; i < m; ++i) {
        const double* x = X + i * n;
        double* y = Y + i * p;
        for (std::size_t j = 0; j < p; ++j) {
            const double* w = W + j * n;
            double s = b[j];
            for (std::size_t k = 0; k < n; ++k) {
                s += x[k] * w[k];
            }
            y[j] = s;
        }
    }
}

void linear_weight_terms(const double* W, const double* b, double* wterm,
                         std::size_t p, std::size_t n) noexcept {
    const std::size_t half = n / 2;
    for (std::size_t j = 0; j < p; ++j) {
        const double* w = W + j * n;
        double s = -b[j];
        for (std::size_t k = 0; k < half; ++k) {
            s += w[2 * k] * w[2 * k + 1];
        }
        wterm[j] = s;
    }
}

void linear_winograd_cached(const double* X, const double* W,
                            const double* wterm, double* Y,
                            std::size_t m, std::size_t n,
                            std::size_t p) noexcept {
    const std::size_t half = n / 2;
    const bool odd = (n % 2 != 0);

    for (std::size_t i = 0; i < m; ++i) {
        const double* x = X + i * n;
        double* y = Y + i * p;

        double xs = 0.0;
        for (std::size_t k = 0; k < half; ++k) {
            xs += x[2 * k] * x[2 * k + 1];
        }

        for (std::size_t j = 0; j < p; ++j) {
            const double* w = W + j * n;
            double s = 0.0;
            for (std::size_t k = 0; k < half; ++k) {
                s += (x[2 * k] + w[2 * k + 1]) *
                     (x[2 * k + 1] + w[2 * k]);
            }
            s -= xs + wterm[j];
            if (odd) {
                s += x[n - 1] * w[n - 1];
            }
            y[j] = s;
        }
    }
}

void linear_winograd(const double* X, const double* W, const double* b,
                     double* Y, std::size_t m, std::size_t n,
                     std::size_t p) noexcept {
    std::vector<double> wterm(p);
    linear_weight_terms(W, b, wterm.data(), p, n);
    linear_winograd_cached(X, W, wterm.data(), Y, m, n, p);
}

// ---------------------------------------------------------------------
// operation counts
// ---------------------------------------------------------------------
OpCount linear_ops_naive(std::size_t m, std::size_t n,
                         std::size_t p) noexcept {
    const std::uint64_t M = m;
    const std::uint64_t P = p;
    OpCount c;
    c.mul = M * P * n;
    // n-1 dot-product additions plus one bias addition per entry.
    c.add = M * P * n;
    return c;
}

OpCount linear_weight_term_ops(std::size_t p, std::size_t n) noexcept {
    const std::uint64_t half = n / 2;
    OpCount c;
    c.mul = static_cast<std::uint64_t>(p) * half;
    c.add = static_cast<std::uint64_t>(p) * (half > 0 ? half - 1 : 0);
    return c;
}

OpCount linear_ops_winograd_cached(std::size_t m, std::size_t n,
                                   std::size_t p) noexcept {
    const std::uint64_t half = n / 2;
    const std::uint64_t odd = n % 2;
    const std::uint64_t M = m;
    const std::uint64_t P = p;
    OpCount c;
    // X self-products + pairwise products (+ odd tail).
    c.mul = M * half + M * P * half + odd * M * P;
    // X self-product reduction + per-entry: 2*half pair sums,
    // (half-1) accumulation, 2 for "-xs-wterm", 1 for the odd tail.
    const std::uint64_t red = half > 0 ? half - 1 : 0;
    c.add = M * red + M * P * (3 * half + 1) + odd * M * P;
    return c;
}

OpCount linear_ops_winograd(std::size_t m, std::size_t n,
                            std::size_t p) noexcept {
    const OpCount core = linear_ops_winograd_cached(m, n, p);
    const OpCount wt = linear_weight_term_ops(p, n);
    OpCount c;
    c.mul = core.mul + wt.mul;
    c.add = core.add + wt.add;
    return c;
}

}  // namespace winograd
