#pragma once
#include <Eigen/Dense>
#include <string>
namespace delta_trust {
struct DoglegResult {
    Eigen::VectorXd step;
    std::string kind;
};
DoglegResult dogleg_step(const Eigen::MatrixXd& L, const Eigen::VectorXd& D, const Eigen::VectorXd& q, double radius);
}
