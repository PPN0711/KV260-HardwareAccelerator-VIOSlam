#pragma once
#include <Eigen/Dense>
namespace delta_trust {
Eigen::MatrixXd synthetic_schur_matrix(int n, unsigned seed, double diag_dominance = 5.0);
Eigen::MatrixXd random_update_columns(int n, int k, unsigned seed, double scale = 0.5);
}
