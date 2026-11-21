// Benchmark: naive vs Winograd inner product / matrix multiplication.
//
// Build:   make
// Run:     ./bench
//          ./bench --quick     (smaller sizes, faster)

#include "linear.hpp"
#include "winograd.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

volatile double g_sink = 0.0;

template <typename Fn>
double time_it(Fn&& fn, long reps) {
    const auto t0 = Clock::now();
    double acc = 0.0;
    for (long r = 0; r < reps; ++r) {
        acc += fn();
    }
    const auto t1 = Clock::now();
    g_sink += acc;  // keep the result alive
    return std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(reps);
}

// Return the best per-call time over three batches, with the batch size
// calibrated so a batch takes at least `min_time` seconds.
template <typename Fn>
double best_time(Fn&& fn, double min_time = 0.05) {
    double t = time_it(fn, 1);
    if (t < 1.0e-5) {
        long cal = t > 0.0 ? static_cast<long>(1.0e-3 / t) : 10000;
        cal = std::min<long>(std::max<long>(cal, 10), 1000000);
        t = time_it(fn, cal);
    }
    long reps = t > 0.0 ? static_cast<long>(std::llround(min_time / t)) : 1;
    reps = std::min<long>(std::max<long>(reps, 1), 5000000);
    double best = 1.0e300;
    for (int trial = 0; trial < 3; ++trial) {
        best = std::min(best, time_it(fn, reps));
    }
    return best;
}

double gflops(std::uint64_t ops, double seconds) {
    return seconds > 0.0 ? static_cast<double>(ops) / seconds / 1.0e9 : 0.0;
}

struct DotResult {
    std::uint64_t mul;
    std::uint64_t add;
    double seconds;
    double rel_err;
};

DotResult run_dot(const std::vector<double>& a, const std::vector<double>& b,
                  bool use_winograd) {
    const std::size_t n = a.size();
    const winograd::OpCount ops = use_winograd
        ? winograd::dot_ops_winograd(n)
        : winograd::dot_ops_naive(n);
    const double reference = winograd::dot_naive(a.data(), b.data(), n);

    double value = 0.0;
    const double seconds = best_time([&]() {
        value = use_winograd
            ? winograd::dot_winograd(a.data(), b.data(), n)
            : winograd::dot_naive(a.data(), b.data(), n);
        return value;
    });
    const double denom = std::max(1.0, std::abs(reference));
    return {ops.mul, ops.add, seconds, std::abs(value - reference) / denom};
}

void benchmark_dot(const std::vector<std::size_t>& sizes, bool quick) {
    std::printf("\n=== 1-D dot product (double) ===\n");
    std::printf("%10s | %-10s | %12s | %12s | %12s | %10s | %10s\n",
                "n", "algorithm", "mul", "add", "time (us)", "GFLOP/s", "rel.err");
    std::printf("%s\n", std::string(88, '-').c_str());

    std::mt19937_64 rng(20241104ULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (std::size_t n : sizes) {
        std::vector<double> a(n), b(n);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);

        for (bool winograd_mode : {false, true}) {
            const DotResult r = run_dot(a, b, winograd_mode);
            const double us = r.seconds * 1.0e6;
            std::printf("%10zu | %-10s | %12llu | %12llu | %12.3f | %10.3f | %10.2e\n",
                        n, winograd_mode ? "Winograd" : "naive",
                        static_cast<unsigned long long>(r.mul),
                        static_cast<unsigned long long>(r.add),
                        us, gflops(r.mul + r.add, r.seconds), r.rel_err);
        }
        (void)quick;
    }
}

struct MatmulResult {
    std::uint64_t mul;
    std::uint64_t add;
    double seconds;
    double max_abs_err;
};

MatmulResult run_matmul(std::size_t m, std::size_t n, std::size_t p,
                        const std::vector<double>& A,
                        const std::vector<double>& B,
                        bool use_winograd) {
    const winograd::OpCount ops = use_winograd
        ? winograd::matmul_ops_winograd(m, n, p)
        : winograd::matmul_ops_naive(m, n, p);
    std::vector<double> C(m * p, 0.0);
    std::vector<double> R(m * p, 0.0);

    // Reference: naive implementation.
    winograd::matmul_naive(A.data(), B.data(), R.data(), m, n, p);

    const double seconds = best_time([&]() {
        if (use_winograd) {
            winograd::matmul_winograd(A.data(), B.data(), C.data(), m, n, p);
        } else {
            winograd::matmul_naive(A.data(), B.data(), C.data(), m, n, p);
        }
        // Returning a scalar keeps the call observable without summing the
        // entire matrix.
        return C[(m / 2) * p + (p / 2)];
    }, 0.05);

    double max_err = 0.0;
    for (std::size_t i = 0; i < m * p; ++i) {
        max_err = std::max(max_err, std::abs(C[i] - R[i]));
    }
    return {ops.mul, ops.add, seconds, max_err};
}

void benchmark_matmul(const std::vector<std::size_t>& sizes, bool quick) {
    std::printf("\n=== C = A * B (square, row-major, no blocking) ===\n");
    std::printf("%10s | %-10s | %14s | %14s | %12s | %10s | %10s\n",
                "n", "algorithm", "mul", "add", "time (ms)", "GFLOP/s", "max|err|");
    std::printf("%s\n", std::string(96, '-').c_str());

    std::mt19937_64 rng(987654321ULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (std::size_t n : sizes) {
        (void)quick;
        std::vector<double> A(n * n), B(n * n);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);

        for (bool winograd_mode : {false, true}) {
            const MatmulResult r = run_matmul(n, n, n, A, B, winograd_mode);
            const double ms = r.seconds * 1.0e3;
            std::printf("%10zu | %-10s | %14llu | %14llu | %12.3f | %10.3f | %10.2e\n",
                        n, winograd_mode ? "Winograd" : "naive",
                        static_cast<unsigned long long>(r.mul),
                        static_cast<unsigned long long>(r.add),
                        ms, gflops(r.mul + r.add, r.seconds), r.max_abs_err);
        }
    }
}

// ---------------------------------------------------------------------
// neural-network linear layer: Y = X * W^T + b
// ---------------------------------------------------------------------
struct LayerShape {
    std::size_t m;
    std::size_t n;
    std::size_t p;
};

struct LinearResult {
    winograd::OpCount ops;
    double seconds;
    double max_abs_err;
};

// mode: 0 = naive, 1 = Winograd (weight terms recomputed),
//       2 = Winograd with cached weight/bias terms (inference).
LinearResult run_linear(const LayerShape& s, const std::vector<double>& X,
                        const std::vector<double>& W,
                        const std::vector<double>& b, int mode) {
    const std::size_t m = s.m, n = s.n, p = s.p;
    std::vector<double> Y(m * p, 0.0);
    std::vector<double> R(m * p, 0.0);
    winograd::linear_naive(X.data(), W.data(), b.data(), R.data(), m, n, p);

    winograd::OpCount ops;
    std::vector<double> wterm;
    if (mode == 0) {
        ops = winograd::linear_ops_naive(m, n, p);
    } else if (mode == 1) {
        ops = winograd::linear_ops_winograd(m, n, p);
    } else {
        ops = winograd::linear_ops_winograd_cached(m, n, p);
        wterm.resize(p);
        winograd::linear_weight_terms(W.data(), b.data(), wterm.data(), p, n);
    }

    const double seconds = best_time([&]() {
        if (mode == 0) {
            winograd::linear_naive(X.data(), W.data(), b.data(), Y.data(),
                                   m, n, p);
        } else if (mode == 1) {
            winograd::linear_winograd(X.data(), W.data(), b.data(), Y.data(),
                                      m, n, p);
        } else {
            winograd::linear_winograd_cached(X.data(), W.data(), wterm.data(),
                                             Y.data(), m, n, p);
        }
        return Y[(m / 2) * p + p / 2];
    }, 0.05);

    double max_err = 0.0;
    for (std::size_t i = 0; i < m * p; ++i) {
        max_err = std::max(max_err, std::abs(Y[i] - R[i]));
    }
    return {ops, seconds, max_err};
}

void benchmark_linear(const std::vector<LayerShape>& shapes, bool quick) {
    (void)quick;
    std::printf("\n=== linear layer: Y = X * W^T + b ===\n");
    std::printf("(*) Winograd with weight/bias terms precomputed once (inference);\n");
    std::printf("    its precompute cost is excluded, X-dependent work is timed.\n");
    std::printf("%6s %6s %6s | %-14s | %14s | %14s | %10s | %10s | %10s\n",
                "m", "n", "p", "algorithm", "mul", "add", "time (ms)",
                "GFLOP/s", "max|err|");
    std::printf("%s\n", std::string(112, '-').c_str());

    std::mt19937_64 rng(5551234ULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (const LayerShape& s : shapes) {
        std::vector<double> X(s.m * s.n), W(s.p * s.n), b(s.p);
        for (auto& v : X) v = dist(rng);
        for (auto& v : W) v = dist(rng);
        for (auto& v : b) v = dist(rng);

        const char* names[3] = {"naive", "Winograd", "Wino-cached*"};
        for (int mode = 0; mode < 3; ++mode) {
            const LinearResult r = run_linear(s, X, W, b, mode);
            const double ms = r.seconds * 1.0e3;
            std::printf("%6zu %6zu %6zu | %-14s | %14llu | %14llu | %10.3f | %10.3f | %10.2e\n",
                        s.m, s.n, s.p, names[mode],
                        static_cast<unsigned long long>(r.ops.mul),
                        static_cast<unsigned long long>(r.ops.add),
                        ms, gflops(r.ops.mul + r.ops.add, r.seconds),
                        r.max_abs_err);
        }
    }
}

// Correctness checks, including odd dimensions.
void check_correctness() {
    std::printf("\n=== correctness checks ===\n");
    std::mt19937_64 rng(42ULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    double worst_dot = 0.0;
    for (std::size_t n : {1, 2, 3, 7, 16, 17, 100, 101, 1000, 1001}) {
        std::vector<double> a(n), b(n);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);
        const double ref = winograd::dot_naive(a.data(), b.data(), n);
        const double val = winograd::dot_winograd(a.data(), b.data(), n);
        worst_dot = std::max(worst_dot, std::abs(val - ref) / std::max(1.0, std::abs(ref)));
    }
    std::printf("dot:      max relative difference over n in {1,2,3,7,16,17,100,101,1000,1001}: %.3e\n",
                worst_dot);

    double worst_mm = 0.0;
    for (std::size_t n : {1, 2, 3, 7, 16, 17}) {
        const std::size_t m = n + 2, p = n + 1;
        std::vector<double> A(m * n), B(n * p), C(m * p), R(m * p);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        winograd::matmul_naive(A.data(), B.data(), R.data(), m, n, p);
        winograd::matmul_winograd(A.data(), B.data(), C.data(), m, n, p);
        for (std::size_t i = 0; i < m * p; ++i) {
            worst_mm = std::max(worst_mm, std::abs(C[i] - R[i]));
        }
    }
    std::printf("matmul:   max absolute difference for odd/even (m,n,p) up to 19: %.3e\n",
                worst_mm);
}

void check_linear() {
    std::printf("\n=== linear layer correctness ===\n");
    const std::vector<LayerShape> shapes = {
        {1, 1, 1}, {3, 5, 4}, {7, 16, 9}, {5, 17, 6}, {8, 32, 12}, {9, 33, 5}};
    std::mt19937_64 rng(777ULL);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    double worst = 0.0;
    double worst_cached = 0.0;
    for (const LayerShape& s : shapes) {
        std::vector<double> X(s.m * s.n), W(s.p * s.n), b(s.p);
        std::vector<double> Y(s.m * s.p), R(s.m * s.p), Yc(s.m * s.p);
        std::vector<double> wterm(s.p);
        for (auto& v : X) v = dist(rng);
        for (auto& v : W) v = dist(rng);
        for (auto& v : b) v = dist(rng);

        winograd::linear_naive(X.data(), W.data(), b.data(), R.data(),
                               s.m, s.n, s.p);
        winograd::linear_winograd(X.data(), W.data(), b.data(), Y.data(),
                                  s.m, s.n, s.p);
        winograd::linear_weight_terms(W.data(), b.data(), wterm.data(),
                                      s.p, s.n);
        winograd::linear_winograd_cached(X.data(), W.data(), wterm.data(),
                                         Yc.data(), s.m, s.n, s.p);
        for (std::size_t i = 0; i < s.m * s.p; ++i) {
            worst = std::max(worst, std::abs(Y[i] - R[i]));
            worst_cached = std::max(worst_cached, std::abs(Yc[i] - R[i]));
        }
    }
    std::printf("linear:   max absolute difference (Winograd vs naive):        %.3e\n",
                worst);
    std::printf("linear:   max absolute difference (cached Winograd vs naive):  %.3e\n",
                worst_cached);
}

std::vector<LayerShape> parse_layer_shapes(const char* text) {
    std::vector<LayerShape> shapes;
    std::string s(text);
    std::size_t pos = 0;
    while (pos < s.size()) {
        const std::size_t comma = s.find(',', pos);
        const std::string token = s.substr(pos, comma - pos);
        std::size_t m = 0, n = 0, p = 0;
        if (std::sscanf(token.c_str(), "%zu:%zu:%zu", &m, &n, &p) == 3 &&
            m > 0 && n > 0 && p > 0) {
            shapes.push_back({m, n, p});
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return shapes;
}

std::vector<std::size_t> parse_sizes(const char* text) {
    std::vector<std::size_t> sizes;
    std::string s(text);
    std::size_t pos = 0;
    while (pos < s.size()) {
        const std::size_t comma = s.find(',', pos);
        const std::string token = s.substr(pos, comma - pos);
        if (!token.empty()) {
            const long value = std::strtol(token.c_str(), nullptr, 10);
            if (value > 0) {
                sizes.push_back(static_cast<std::size_t>(value));
            }
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return sizes;
}

}  // namespace

int main(int argc, char** argv) {
    bool quick = false;
    std::vector<std::size_t> dot_sizes;
    std::vector<std::size_t> mm_sizes;
    std::vector<LayerShape> fc_shapes;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--quick") == 0) {
            quick = true;
        } else if (std::strcmp(argv[i], "--dot-sizes") == 0 && i + 1 < argc) {
            dot_sizes = parse_sizes(argv[++i]);
        } else if (std::strcmp(argv[i], "--mm-sizes") == 0 && i + 1 < argc) {
            mm_sizes = parse_sizes(argv[++i]);
        } else if (std::strcmp(argv[i], "--fc-shapes") == 0 && i + 1 < argc) {
            fc_shapes = parse_layer_shapes(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0 ||
                   std::strcmp(argv[i], "-h") == 0) {
            std::printf("usage: %s [--quick] [--dot-sizes a,b,c] "
                        "[--mm-sizes a,b,c] [--fc-shapes m:n:p,...]\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
    }

    if (dot_sizes.empty()) {
        dot_sizes = quick ? std::vector<std::size_t>{16, 64, 256, 1024, 4096}
                          : std::vector<std::size_t>{16, 64, 256, 1024, 4096, 16384, 65536};
    }
    if (mm_sizes.empty()) {
        // Non-power-of-two sizes avoid the cache-conflict cliff caused by a
        // row stride of exactly 1/2/4 KB.  Try e.g. --mm-sizes 128,256 to
        // see that effect.
        mm_sizes = quick ? std::vector<std::size_t>{48, 96, 120, 160}
                         : std::vector<std::size_t>{48, 96, 120, 160, 200, 240, 320};
    }
    if (fc_shapes.empty()) {
        fc_shapes = quick
            ? std::vector<LayerShape>{{32, 128, 128}, {64, 256, 256}}
            : std::vector<LayerShape>{{32, 128, 128}, {64, 256, 256},
                                      {128, 512, 512}, {128, 784, 256},
                                      {256, 1024, 1024}};
    }

    std::printf("Winograd inner-product benchmark\n");
    std::printf("Note: a single dot product needs 3n/2 multiplications in Winograd\n");
    std::printf("      form (worse than n); the saving appears only in batched\n");
    std::printf("      products, where X(a) and Y(b) are reused.\n");

    check_correctness();
    check_linear();
    benchmark_dot(dot_sizes, quick);
    benchmark_matmul(mm_sizes, quick);
    benchmark_linear(fc_shapes, quick);

    std::printf("\n(checksum sink: %.6e)\n", g_sink);
    return 0;
}
