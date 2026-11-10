#ifndef LDLT_HLS_H
#define LDLT_HLS_H

#include "delta_trust_hls.h"

void ldlt_decompose_hls(
    data_t A[MAX_N][MAX_N],
    data_t L[MAX_N][MAX_N],
    data_t D[MAX_N],
    int n
);

void ldlt_solve_hls(
    data_t L[MAX_N][MAX_N],
    data_t D[MAX_N],
    data_t b[MAX_N],
    data_t x[MAX_N],
    int n
);

void ldlt_rank1_update_hls(
    data_t L[MAX_N][MAX_N],
    data_t D[MAX_N],
    data_t v[MAX_N],
    data_t sigma,
    int n
);

#endif