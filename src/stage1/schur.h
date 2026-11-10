#pragma once
#include <Eigen/Dense>
namespace delta_trust {
struct SchurResult { Eigen::MatrixXd S; Eigen::VectorXd b_reduced; };
SchurResult applySchur(const Eigen::MatrixXd& H, const Eigen::VectorXd& b, int n_pose, int n_point);
}
