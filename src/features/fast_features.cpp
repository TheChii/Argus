#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstring>
#include <omp.h>

#ifdef _WIN32
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#endif

extern "C" {

/**
 * Computes technical features for a single ticker price and volume array.
 *
 * Returns 0 on success, or a negative error code:
 *  -1: NULL pointer passed for any required buffer
 *  -2: Invalid length (n < 0)
 */
EXPORT_SYMBOL int compute_ticker_technical_cpp(
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
) {
    if (!close || !volume || !out_ret1 || !out_ret5 || !out_ret20 || !out_ret60 ||
        !out_vol5 || !out_vol20 || !out_vol_zscore20 || !out_dist_ma50 || !out_rsi14) {
        return -1;
    }
    if (n < 0) {
        return -2;
    }
    if (n == 0) {
        return 0;
    }

    const double nan_val = std::numeric_limits<double>::quiet_NaN();
    const double sqrt_252 = std::sqrt(252.0);

    for (int i = 0; i < n; ++i) {
        out_ret1[i] = (i >= 1 && std::isfinite(close[i]) && std::isfinite(close[i - 1]) && close[i - 1] > 0.0)
            ? (close[i] / close[i - 1] - 1.0) : nan_val;
        out_ret5[i] = (i >= 5 && std::isfinite(close[i]) && std::isfinite(close[i - 5]) && close[i - 5] > 0.0)
            ? (close[i] / close[i - 5] - 1.0) : nan_val;
        out_ret20[i] = (i >= 20 && std::isfinite(close[i]) && std::isfinite(close[i - 20]) && close[i - 20] > 0.0)
            ? (close[i] / close[i - 20] - 1.0) : nan_val;
        out_ret60[i] = (i >= 60 && std::isfinite(close[i]) && std::isfinite(close[i - 60]) && close[i - 60] > 0.0)
            ? (close[i] / close[i - 60] - 1.0) : nan_val;
    }

    for (int i = 0; i < n; ++i) {
        int start5 = std::max(0, i - 4);
        int count5 = 0;
        double sum5 = 0.0;
        for (int j = start5; j <= i; ++j) {
            double v = out_ret1[j];
            if (std::isfinite(v)) {
                sum5 += v;
                count5++;
            }
        }
        if (count5 >= 3) {
            double mean5 = sum5 / count5;
            double sq_sum5 = 0.0;
            for (int j = start5; j <= i; ++j) {
                double v = out_ret1[j];
                if (std::isfinite(v)) {
                    double diff = v - mean5;
                    sq_sum5 += diff * diff;
                }
            }
            // Guard against negative variance from rounding errors
            double s5 = std::sqrt(std::max(0.0, sq_sum5 / (count5 - 1)));
            out_vol5[i] = s5 * sqrt_252;
        } else {
            out_vol5[i] = nan_val;
        }

        int start20 = std::max(0, i - 19);
        int count20 = 0;
        double sum20 = 0.0;
        for (int j = start20; j <= i; ++j) {
            double v = out_ret1[j];
            if (std::isfinite(v)) {
                sum20 += v;
                count20++;
            }
        }
        if (count20 >= 10) {
            double mean20 = sum20 / count20;
            double sq_sum20 = 0.0;
            for (int j = start20; j <= i; ++j) {
                double v = out_ret1[j];
                if (std::isfinite(v)) {
                    double diff = v - mean20;
                    sq_sum20 += diff * diff;
                }
            }
            double s20 = std::sqrt(std::max(0.0, sq_sum20 / (count20 - 1)));
            out_vol20[i] = s20 * sqrt_252;
        } else {
            out_vol20[i] = nan_val;
        }

        int vol_count = 0;
        double vol_sum = 0.0;
        for (int j = start20; j <= i; ++j) {
            double v = volume[j];
            if (std::isfinite(v) && v >= 0.0) {
                vol_sum += v;
                vol_count++;
            }
        }
        if (vol_count >= 10) {
            double vol_mean = vol_sum / vol_count;
            double vol_sq = 0.0;
            for (int j = start20; j <= i; ++j) {
                double v = volume[j];
                if (std::isfinite(v) && v >= 0.0) {
                    double diff = v - vol_mean;
                    vol_sq += diff * diff;
                }
            }
            double vol_std = std::sqrt(std::max(0.0, vol_sq / (vol_count - 1)));
            out_vol_zscore20[i] = (volume[i] - vol_mean) / (vol_std + 1e-8);
        } else {
            out_vol_zscore20[i] = nan_val;
        }

        int start50 = std::max(0, i - 49);
        int ma_count = 0;
        double ma_sum = 0.0;
        for (int j = start50; j <= i; ++j) {
            double p = close[j];
            if (std::isfinite(p)) {
                ma_sum += p;
                ma_count++;
            }
        }
        if (ma_count >= 25) {
            double ma50 = ma_sum / ma_count;
            double denom = (std::abs(ma50) > 1e-8) ? ma50 : 1e-8;
            out_dist_ma50[i] = (close[i] - ma50) / denom;
        } else {
            out_dist_ma50[i] = nan_val;
        }
    }

    const double alpha = 1.0 / 14.0;
    double avg_gain = 0.0;
    double avg_loss = 0.0;
    int valid_obs = 0;

    for (int i = 0; i < n; ++i) {
        if (i == 0 || !std::isfinite(close[i]) || !std::isfinite(close[i - 1])) {
            out_rsi14[i] = nan_val;
            continue;
        }

        double delta = close[i] - close[i - 1];
        double g = (delta > 0.0) ? delta : 0.0;
        double l = (delta < 0.0) ? -delta : 0.0;

        valid_obs++;
        if (valid_obs == 1) {
            avg_gain = g;
            avg_loss = l;
        } else {
            avg_gain = (1.0 - alpha) * avg_gain + alpha * g;
            avg_loss = (1.0 - alpha) * avg_loss + alpha * l;
        }

        if (valid_obs < 14) {
            out_rsi14[i] = nan_val;
        } else {
            if (avg_gain < 1e-7 && avg_loss < 1e-7) {
                out_rsi14[i] = 50.0;
            } else {
                double denom = std::max(0.0, avg_loss) + 1e-8;
                double rs = avg_gain / denom;
                double calc_rsi = 100.0 - (100.0 / (1.0 + rs));
                out_rsi14[i] = std::clamp(calc_rsi, 0.0, 100.0);
            }
        }
    }

    return 0;
}

/**
 * Batch technical feature extractor across multiple tickers in parallel using OpenMP.
 *
 * Returns 0 on success, or a negative error code:
 *  -1: NULL pointer passed
 *  -2: Invalid bounds or corrupt offsets
 */
EXPORT_SYMBOL int compute_panel_technical_cpp(
    const double* close,
    const double* volume,
    const int* ticker_offsets,
    int num_tickers,
    int total_rows,
    double* out_ret1,
    double* out_ret5,
    double* out_ret20,
    double* out_ret60,
    double* out_vol5,
    double* out_vol20,
    double* out_vol_zscore20,
    double* out_dist_ma50,
    double* out_rsi14
) {
    if (!close || !volume || !ticker_offsets || !out_ret1 || !out_ret5 ||
        !out_ret20 || !out_ret60 || !out_vol5 || !out_vol20 ||
        !out_vol_zscore20 || !out_dist_ma50 || !out_rsi14) {
        return -1;
    }
    if (num_tickers < 0 || total_rows < 0) {
        return -2;
    }
    if (num_tickers == 0 || total_rows == 0) {
        return 0;
    }

    if (ticker_offsets[0] != 0 || ticker_offsets[num_tickers] != total_rows) {
        return -2;
    }
    for (int t = 0; t < num_tickers; ++t) {
        if (ticker_offsets[t] < 0 || ticker_offsets[t + 1] < ticker_offsets[t] || ticker_offsets[t + 1] > total_rows) {
            return -2;
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < num_tickers; ++t) {
        int start = ticker_offsets[t];
        int end = ticker_offsets[t + 1];
        int len = end - start;
        if (len <= 0) continue;

        compute_ticker_technical_cpp(
            close + start,
            volume + start,
            len,
            out_ret1 + start,
            out_ret5 + start,
            out_ret20 + start,
            out_ret60 + start,
            out_vol5 + start,
            out_vol20 + start,
            out_vol_zscore20 + start,
            out_dist_ma50 + start,
            out_rsi14 + start
        );
    }

    return 0;
}

struct ValIndex {
    double val;
    int idx;
};

/**
 * Normalizes an array of features cross-sectionally date-by-date into uniform percentile [-0.5, 0.5].
 *
 * Returns 0 on success, or a negative error code:
 *  -1: NULL pointer passed
 *  -2: Invalid bounds or corrupt offsets
 */
EXPORT_SYMBOL int cross_sectional_rank_cpp(
    const double* const* in_cols,
    double* const* out_cols,
    int num_cols,
    const int* date_offsets,
    int num_dates,
    int total_rows
) {
    if (!in_cols || !out_cols || !date_offsets) {
        return -1;
    }
    if (num_cols < 0 || num_dates < 0 || total_rows < 0) {
        return -2;
    }
    if (num_cols == 0 || num_dates == 0 || total_rows == 0) {
        return 0;
    }

    for (int c = 0; c < num_cols; ++c) {
        if (!in_cols[c] || !out_cols[c]) {
            return -1;
        }
    }

    if (date_offsets[0] != 0 || date_offsets[num_dates] != total_rows) {
        return -2;
    }
    for (int d = 0; d < num_dates; ++d) {
        if (date_offsets[d] < 0 || date_offsets[d + 1] < date_offsets[d] || date_offsets[d + 1] > total_rows) {
            return -2;
        }
    }

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int c = 0; c < num_cols; ++c) {
        for (int d = 0; d < num_dates; ++d) {
            int start = date_offsets[d];
            int end = date_offsets[d + 1];
            int group_size = end - start;
            if (group_size <= 0) continue;

            const double* col_in = in_cols[c] + start;
            double* col_out = out_cols[c] + start;

            std::vector<ValIndex> valid_items;
            valid_items.reserve(group_size);

            for (int i = 0; i < group_size; ++i) {
                double val = col_in[i];
                if (std::isfinite(val)) {
                    valid_items.push_back({val, i});
                } else {
                    col_out[i] = 0.0;
                }
            }

            int K = static_cast<int>(valid_items.size());
            if (K <= 1) {
                for (int i = 0; i < group_size; ++i) {
                    col_out[i] = 0.0;
                }
                continue;
            }

            std::sort(valid_items.begin(), valid_items.end(), [](const ValIndex& a, const ValIndex& b) {
                return a.val < b.val;
            });

            int i = 0;
            while (i < K) {
                int j = i;
                while (j < K && valid_items[j].val == valid_items[i].val) {
                    j++;
                }
                double avg_rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
                double normed = (avg_rank - 1.0) / (static_cast<double>(K) - 1.0) - 0.5;

                for (int k = i; k < j; ++k) {
                    col_out[valid_items[k].idx] = normed;
                }
                i = j;
            }
        }
    }

    return 0;
}

} // extern "C"
