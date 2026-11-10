#pragma once
#include <ap_fixed.h>
#include <ap_int.h>

// ============================================================
// Q16.16 fixed-point, TRUNCATION + SATURATION
// ============================================================
typedef ap_fixed<32, 16, AP_TRN, AP_SAT> data_t;

// Đặt MAX_N = kích thước thật cần dùng (16 để test, 192 cho n=180).
// MAX_N ảnh hưởng trực tiếp tới BRAM.
#ifndef MAX_N
#define MAX_N 16
#endif

// ============================================================
// fx_div: bộ chia restoring TUẦN TỰ (1 bit/chu kỳ, 48 chu kỳ).
// Thay cho toán tử '/' của ap_fixed (bộ chia tổ hợp/pipeline rất tốn LUT).
// Kết quả Q16.16, cắt về 0, bão hòa khi tràn / chia 0.
// ============================================================
inline data_t fx_div(data_t a, data_t b)
{
    #pragma HLS INLINE off
    ap_int<32> ra = a.range(31, 0);
    ap_int<32> rb = b.range(31, 0);
    bool neg = (ra < 0) != (rb < 0);

    ap_uint<32> ua = (ra < 0) ? ap_uint<32>(-ap_int<33>(ra)) : ap_uint<32>(ra);
    ap_uint<32> ub = (rb < 0) ? ap_uint<32>(-ap_int<33>(rb)) : ap_uint<32>(rb);

    data_t r;
    if (ub == 0) {
        ap_int<32> sat = neg ? ap_int<32>(0x80000000) : ap_int<32>(0x7FFFFFFF);
        r.range(31, 0) = sat.range(31, 0);
        return r;
    }

    ap_uint<48> num = ((ap_uint<48>)ua) << 16;
    ap_uint<34> rem = 0;
    ap_uint<48> quo = 0;

    div_loop: for (int i = 47; i >= 0; --i) {
        #pragma HLS PIPELINE II=1
        rem = (rem << 1) | ap_uint<34>(num[i]);
        if (rem >= ub) {
            rem -= ub;
            quo[i] = 1;
        }
    }

    ap_uint<48> lim = neg ? ap_uint<48>(0x80000000u) : ap_uint<48>(0x7FFFFFFFu);
    if (quo > lim) quo = lim;
    ap_int<34> sv = neg ? ap_int<34>(-ap_int<35>(quo)) : ap_int<34>(quo);
    r.range(31, 0) = sv.range(31, 0);
    return r;
}

// ============================================================
// fx_sqrt: căn bậc hai nguyên bit-by-bit (24 chu kỳ), Q16.16.
// Thay cho hls::sqrt((float)x) (float sqrt + 2 bộ đổi kiểu rất tốn).
// ============================================================
inline data_t fx_sqrt(data_t x)
{
    #pragma HLS INLINE off
    data_t r = 0;
    if (x <= 0) return r;

    ap_uint<32> rx = x.range(31, 0);
    ap_uint<48> num = ((ap_uint<48>)rx) << 16;   // sqrt(raw * 2^16)
    ap_uint<48> res = 0;
    ap_uint<48> bit = ((ap_uint<48>)1) << 46;

    sqrt_loop: for (int k = 0; k < 24; ++k) {
        #pragma HLS PIPELINE II=1
        ap_uint<48> t = res + bit;
        if (num >= t) {
            num -= t;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    r.range(31, 0) = res.range(31, 0);
    return r;
}

// ---------- LDL^T ----------
void ldlt_decompose_hls(
    data_t A[MAX_N][MAX_N],
    data_t L[MAX_N][MAX_N],
    data_t D[MAX_N],
    int n);

void ldlt_solve_hls(
    data_t L[MAX_N][MAX_N],
    data_t D[MAX_N],
    data_t b[MAX_N],
    data_t x[MAX_N],
    int n);

// ---------- Rank-1 Update (GGMS 1974) ----------
void ldlt_rank1_update_hls(
    data_t L[MAX_N][MAX_N],
    data_t D[MAX_N],
    data_t v[MAX_N],
    data_t sigma,
    int n);

// ---------- Dogleg FULL (3 nhánh) ----------
void dogleg_hls(
    data_t L[MAX_N][MAX_N],
    data_t D[MAX_N],
    data_t q[MAX_N],
    data_t step[MAX_N],
    data_t radius,
    int n);

// ---------- Top function ----------
void delta_trust_top(
    data_t S[MAX_N][MAX_N],
    data_t q[MAX_N],
    data_t step[MAX_N],
    data_t radius,
    int n);