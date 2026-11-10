#include <iostream>
#include <fstream>
#include "stage0/matrix_utils.h"
#include "stage0/ldlt.h"
#include "stage0/ldlt_rank1.h"
#include "stage0/dogleg.h"

using namespace delta_trust;

int main() {
    // Small matrix for HLS testing (n=16 to fit on chip)
    const int n = 16;
    
    // Generate SPD matrix
    auto S = synthetic_schur_matrix(n, 42);
    
    // Save S to file
    std::ofstream fout("hls/data/input.txt");
    fout << n << "\n";
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            fout << S(i, j) << (j == n-1 ? "\n" : " ");
    
    // Compute golden reference
    auto factor = ldlt_decompose(S);
    
    // Save L and D
    std::ofstream gout("hls/data/golden.txt");
    gout << n << "\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            gout << factor.L(i, j) << (j == n-1 ? "\n" : " ");
    }
    for (int i = 0; i < n; ++i)
        gout << factor.D(i) << (i == n-1 ? "\n" : " ");
    
    std::cout << "Generated test data for n=" << n << "\n";
    return 0;
}
