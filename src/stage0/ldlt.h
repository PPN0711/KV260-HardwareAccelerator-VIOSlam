#pragma once
#include <Eigen/Dense>
namespace delta_trust {
struct LDLT { Eigen::MatrixXd L; Eigen::VectorXd D; };
LDLT ldlt_decompose(const Eigen::MatrixXd& A);
Eigen::VectorXd ldlt_solve(const Eigen::MatrixXd& L, const Eigen::VectorXd& D, const Eigen::VectorXd& rhs);
Eigen::VectorXd apply_S(const Eigen::MatrixXd& L, const Eigen::VectorXd& D, const Eigen::VectorXd& x);
Eigen::MatrixXd reconstruct_S(const Eigen::MatrixXd& L, const Eigen::VectorXd& D);
}
