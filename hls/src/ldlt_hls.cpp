#include "ldlt_hls.h"

// ============================================================
// LDL^T Decomposition — tối ưu tài nguyên
//  * Làm IN-PLACE trong L (bỏ hẳn mảng Awork MAX_N x MAX_N)
//  * Chỉ cập nhật tam giác dưới (k <= i)  -> giảm ~2x phép tính
//  * L[i][j]*D[j] = A[i][j] nên không cần nhân lần nữa
//    -> chỉ 1 phép nhân / phần tử Schur (trước đây 2)
//  * Pipeline vòng trong cùng (không để HLS unroll vòng ngoài)
// ============================================================
void ldlt_decompose_hls(
    data_t A[MAX_N][MAX_N],
    data_t L[MAX_N][MAX_N],
    data_t D[MAX_N],
    int n
) {
    #pragma HLS INLINE OFF

    data_t lcol[MAX_N];   // L[:,j]
    data_t acol[MAX_N];   // A[:,j] (= L[:,j]*D[j])

    // Copy tam giác dưới của A vào L, tam giác trên = 0
    copy_i: for (int i = 0; i < n; ++i) {
        copy_j: for (int j = 0; j < n; ++j) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_N
            L[i][j] = (j <= i) ? A[i][j] : data_t(0);
        }
    }

    col_loop: for (int j = 0; j < n; ++j) {
        data_t d = L[j][j];
        D[j] = d;

        // cột j của L (1 bộ chia tuần tự dùng chung)
        div_loop: for (int i = j + 1; i < n; ++i) {
            data_t a = L[i][j];
            acol[i] = a;
            lcol[i] = a / d;
        }

        // Schur complement, chỉ tam giác dưới
        update_i: for (int i = j + 1; i < n; ++i) {
            data_t ai = acol[i];
            update_k: for (int k = j + 1; k <= i; ++k) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_N
                L[i][k] -= ai * lcol[k];
            }
        }

        // ghi cột j, đường chéo L = 1
        wb: for (int i = j + 1; i < n; ++i) {
            #pragma HLS PIPELINE II=1
            L[i][j] = lcol[i];
        }
        L[j][j] = data_t(1);
    }
}

// ============================================================
// LDL^T Solve — Forward / Diag / Backward
//  * Pipeline vòng TRONG (bản cũ pipeline vòng ngoài -> HLS unroll
//    vòng trong MAX_N lần = hàng chục bộ nhân)
//  * Dùng bộ chia tuần tự
// ============================================================
void ldlt_solve_hls(
    data_t L[MAX_N][MAX_N],
    data_t D[MAX_N],
    data_t b[MAX_N],
    data_t x[MAX_N],
    int n
) {
    #pragma HLS INLINE OFF

    data_t y[MAX_N];

    forward: for (int i = 0; i < n; ++i) {
        data_t s = b[i];
        fwd_in: for (int j = 0; j < i; ++j) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=0 max=MAX_N
            s -= L[i][j] * y[j];
        }
        y[i] = s;
    }

    diag: for (int i = 0; i < n; ++i) {
        y[i] = y[i] / D[i];
    }

    backward: for (int i = n - 1; i >= 0; --i) {
        data_t s = y[i];
        bwd_in: for (int j = i + 1; j < n; ++j) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=0 max=MAX_N
            s -= L[j][i] * x[j];
        }
        x[i] = s;
    }
}

// ============================================================
// Rank-1 Update — GGMS
// ============================================================
void ldlt_rank1_update_hls(
    data_t L[MAX_N][MAX_N],
    data_t D[MAX_N],
    data_t v[MAX_N],
    data_t sigma,
    int n
) {
    #pragma HLS INLINE OFF
    data_t w[MAX_N];
    copy_v: for (int i = 0; i < n; ++i) {
        #pragma HLS PIPELINE II=1
        w[i] = v[i];
    }

    rank1_loop: for (int j = 0; j < n; ++j) {
        data_t t     = sigma * w[j];
        data_t d_new = D[j] + t * w[j];
        D[j] = d_new;
        data_t alpha = t / d_new;

        update_r: for (int r = j + 1; r < n; ++r) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=0 max=MAX_N
            data_t lr = L[r][j];
            data_t wr = w[r] - alpha * lr;
            w[r] = wr;
            L[r][j] = lr + alpha * wr;
        }
    }
}