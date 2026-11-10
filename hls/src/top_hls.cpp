#include "delta_trust_hls.h"

// ============================================================
// TOP FUNCTION — giao diện GIỮ NGUYÊN (bram + s_axilite)
// nên Block Design hiện tại không cần đổi cổng.
// ============================================================
void delta_trust_top(
    data_t S[MAX_N][MAX_N],
    data_t q[MAX_N],
    data_t step[MAX_N],
    data_t radius,
    int n)
{
    #pragma HLS INTERFACE bram port=S
    #pragma HLS INTERFACE bram port=q
    #pragma HLS INTERFACE bram port=step
    #pragma HLS INTERFACE s_axilite port=radius bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=n bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    // L: 1 mảng duy nhất, ép vào BRAM 2 cổng.
    // KHÔNG partition: bản cũ partition D complete (64x32 FF + mux 64:1)
    // và L cyclic 8 -> tốn LUT/FF/BRAM vô ích vì thiết kế chạy tuần tự.
    data_t L[MAX_N][MAX_N];
    data_t D[MAX_N];
    #pragma HLS BIND_STORAGE variable=L type=ram_2p impl=bram

    ldlt_decompose_hls(S, L, D, n);
    dogleg_hls(L, D, q, step, radius, n);
}