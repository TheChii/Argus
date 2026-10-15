#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>

// Forward declaration of functions from fast_features.cpp
extern "C" {
int compute_ticker_technical_cpp(
    const double* close,
    const double* volume,
    int n,
    double* out_ret1,
    double* out_ret5,
    double* out_ret20,
    double* out_ret60,
    double* out_vol5,
    double* out_vol20,
    double* out_vol_zscore20,
    double* out_dist_ma50,
    double* out_rsi14
);

int cross_sectional_rank_cpp(
    const double* const* in_cols,
    double* const* out_cols,
    int num_cols,
    const int* date_offsets,
    int num_dates,
    int total_rows
);
}

int main(int argc, char* argv[]) {
    int n = 2500;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }

    std::vector<double> close(n, 100.0);
    std::vector<double> volume(n, 1000000.0);
    for (int i = 0; i < n; ++i) {
        close[i] = 100.0 + (i % 50) * 1.5;
        volume[i] = 1000000.0 + (i % 20) * 50000.0;
    }

    std::vector<double> r1(n), r5(n), r20(n), r60(n);
    std::vector<double> v5(n), v20(n), vz20(n), d50(n), rsi(n);

    auto t0 = std::chrono::high_resolution_clock::now();
    compute_ticker_technical_cpp(
        close.data(), volume.data(), n,
        r1.data(), r5.data(), r20.data(), r60.data(),
        v5.data(), v20.data(), vz20.data(), d50.data(), rsi.data()
    );
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Computed technical features for N=" << n << " in " << ms << " ms." << std::endl;
    return 0;
}
