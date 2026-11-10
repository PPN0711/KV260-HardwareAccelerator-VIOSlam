#include "stage0/matrix_utils.h"
#include <random>
namespace delta_trust {
Eigen::MatrixXd synthetic_schur_matrix(int n, unsigned seed, double diag_dominance) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::MatrixXd M(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M(i, j) = nd(rng) * 0.3;
    Eigen::MatrixXd A = M * M.transpose();
    A += Eigen::MatrixXd::Identity(n, n) * (diag_dominance * n);
    return A;
}
Eigen::MatrixXd random_update_columns(int n, int k, unsigned seed, double scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::MatrixXd V(n, k);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j)
            V(i, j) = nd(rng) * scale;
    return V;
}
}
