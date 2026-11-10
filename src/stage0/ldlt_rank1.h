#pragma once
#include <Eigen/Dense>
namespace delta_trust {
struct UpdateResult {
    Eigen::MatrixXd L;
    Eigen::VectorXd D;
    double min_D;
    bool ok;
    int fail_index;
};
UpdateResult ldlt_rank1_update(const Eigen::MatrixXd& L_in, const Eigen::VectorXd& D_in, const Eigen::VectorXd& v, double sigma, double d_floor = 1e-8);
UpdateResult ldlt_update_rankk(const Eigen::MatrixXd& L_in, const Eigen::VectorXd& D_in, const Eigen::MatrixXd& V, const Eigen::VectorXd& signs, double d_floor = 1e-8);
}
