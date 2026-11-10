#include "delta_trust_hls.h"

// ============================================================
// Dogleg FULL — tối ưu tài nguyên
//  * Bỏ apply_S_hls: q^T S q = sum D[i] * (L^T q)[i]^2
//    -> chỉ cần 1 lượt L^T q, không cần lượt L*(D*..) và mảng Sq
//  * Bỏ float sqrt: so sánh bằng bình phương, sqrt bằng fx_sqrt
//  * Nhánh SD: step = radius * q / |q| (alpha triệt tiêu)
//  * Nhánh nội suy: không lưu p_sd/diff, tính lại tại chỗ
//  * step (cổng BRAM) chỉ được GHI 1 lần, tuần tự
// ============================================================
static void write_step(data_t step[MAX_N], data_t src[MAX_N], int n)
{
    wr: for (int i = 0; i < n; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_N
        step[i] = src[i];
    }
}

void dogleg_hls(
    data_t L[MAX_N][MAX_N],
    data_t D[MAX_N],
    data_t q[MAX_N],
    data_t step[MAX_N],
    data_t radius,
    int n)
{
    #pragma HLS INLINE OFF

    data_t pgn[MAX_N];      // bước Gauss-Newton
    data_t r2 = radius * radius;

    // ----- Nhánh 1: Gauss-Newton -----
    ldlt_solve_hls(L, D, q, pgn, n);

    data_t nsq = data_t(0);
    gn_norm: for (int i = 0; i < n; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_N
        nsq += pgn[i] * pgn[i];
    }
    if (nsq <= r2) {
        write_step(step, pgn, n);
        return;
    }

    // ----- Nhánh 2: Steepest Descent -----
    data_t tmp[MAX_N];      // L^T q
    ltq_i: for (int i = 0; i < n; ++i) {
        data_t s = data_t(0);
        ltq_j: for (int j = i; j < n; ++j) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_N
            s += L[j][i] * q[j];
        }
        tmp[i] = s;
    }

    data_t qq  = data_t(0);
    data_t qSq = data_t(0);
    dots: for (int i = 0; i < n; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_N
        data_t t = tmp[i];
        qq  += q[i] * q[i];
        qSq += (D[i] * t) * t;
    }

    data_t nq = fx_sqrt(qq);

    if (qSq <= 0) {   // S không xác định dương: thu GN về biên
        data_t k = fx_div(radius, fx_sqrt(nsq));
        scale_gn: for (int i = 0; i < n; ++i) {
            #pragma HLS PIPELINE II=1
            tmp[i] = k * pgn[i];
        }
        write_step(step, tmp, n);
        return;
    }

    data_t alpha   = fx_div(qq, qSq);
    data_t norm_sd = alpha * nq;

    if (norm_sd >= radius) {
        data_t k = fx_div(radius, nq);     // = radius*alpha/norm_sd /alpha
        scale_sd: for (int i = 0; i < n; ++i) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_N
            tmp[i] = k * q[i];
        }
        write_step(step, tmp, n);
        return;
    }

    // ----- Nhánh 3: interpolation p_sd -> p_gn -----
    data_t a = data_t(0);
    data_t b_coef = data_t(0);
    data_t c = data_t(0);
    coef: for (int i = 0; i < n; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_N
        data_t psd  = alpha * q[i];
        data_t diff = pgn[i] - psd;
        a      += diff * diff;
        b_coef += 2 * psd * diff;
        c      += psd * psd;
    }
    c -= r2;

    if (a == 0) {
        write_step(step, pgn, n);
        return;
    }

    data_t disc = b_coef * b_coef - 4 * a * c;
    if (disc < 0) disc = data_t(0);

    data_t tau = fx_div(-b_coef + fx_sqrt(disc), 2 * a);
    if (tau < 0) tau = data_t(0);
    if (tau > 1) tau = data_t(1);

    mix: for (int i = 0; i < n; ++i) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_N
        data_t psd = alpha * q[i];
        tmp[i] = psd + tau * (pgn[i] - psd);
    }
    write_step(step, tmp, n);
}