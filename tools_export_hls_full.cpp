#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include "stage0/matrix_utils.h"
#include "stage0/ldlt.h"
#include "stage0/ldlt_rank1.h"
#include "stage0/dogleg.h"

using namespace delta_trust;

int main() {
    const int n = 16;
    auto S = synthetic_schur_matrix(n, 42);

    // ===== 1. Export S =====
    std::ofstream fout("hls/data/input.txt");
    fout << std::scientific << std::setprecision(15);
    fout << n << "\n";
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            fout << S(i, j) << (j == n-1 ? "\n" : " ");

    // ===== 2. Golden L, D =====
    auto factor = ldlt_decompose(S);
    std::ofstream gout("hls/data/golden.txt");
    gout << std::scientific << std::setprecision(15);
    gout << n << "\n";
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            gout << factor.L(i, j) << (j == n-1 ? "\n" : " ");
    for (int i = 0; i < n; ++i)
        gout << factor.D(i) << (i == n-1 ? "\n" : " ");

    // ===== 3. Test vector q cho Dogleg =====
    std::mt19937 rng(123);
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::VectorXd q(n);
    for (int i = 0; i < n; ++i) q(i) = nd(rng);
    std::ofstream qout("hls/data/q.txt");
    qout << std::scientific << std::setprecision(15);
    qout << n << "\n";
    for (int i = 0; i < n; ++i)
        qout << q(i) << (i == n-1 ? "\n" : " ");

    // ===== 4. Golden Dogleg step =====
    auto dl = dogleg_step(factor.L, factor.D, q, 1.0);
    std::ofstream dout("hls/data/dogleg_golden.txt");
    dout << std::scientific << std::setprecision(15);
    dout << n << "\n";
    dout << dl.kind << "\n";
    for (int i = 0; i < n; ++i)
        dout << dl.step(i) << (i == n-1 ? "\n" : " ");

    // ===== 5. Rank-1 update test =====
    Eigen::VectorXd v(n);
    for (int i = 0; i < n; ++i) v(i) = 0.1 * (i + 1);
    double sigma = 0.5;
    std::ofstream vout("hls/data/rank1_v.txt");
    vout << std::scientific << std::setprecision(15);
    vout << n << "\n" << sigma << "\n";
    for (int i = 0; i < n; ++i)
        vout << v(i) << (i == n-1 ? "\n" : " ");

    auto upd = ldlt_rank1_update(factor.L, factor.D, v, sigma);
    std::ofstream uout("hls/data/rank1_golden.txt");
    uout << std::scientific << std::setprecision(15);
    uout << n << "\n" << (upd.ok ? 1 : 0) << "\n";
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            uout << upd.L(i, j) << (j == n-1 ? "\n" : " ");
    for (int i = 0; i < n; ++i)
        uout << upd.D(i) << (i == n-1 ? "\n" : " ");

    std::cout << "Generated: input.txt, golden.txt, q.txt, "
              << "dogleg_golden.txt, rank1_v.txt, rank1_golden.txt\n";
    return 0;
}
