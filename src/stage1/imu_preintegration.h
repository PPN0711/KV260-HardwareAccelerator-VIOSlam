#pragma once
#include <Eigen/Dense>
#include "stage1/euroc_reader.h"
namespace delta_trust {
Eigen::Matrix3d skew3(const Eigen::Vector3d& v);
Eigen::Matrix3d expSO3(const Eigen::Vector3d& omega);
Eigen::Vector3d logSO3(const Eigen::Matrix3d& R);

struct ImuPreintegration {
    Eigen::Matrix3d delta_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d delta_v = Eigen::Vector3d::Zero();
    Eigen::Vector3d delta_p = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 9, 9> cov = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 9> cov_inv = Eigen::Matrix<double, 9, 9>::Zero();  // <-- CACHE
    double dt_sum = 0.0;
    int n_measurements = 0;
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, 9.81);
    
    Eigen::Matrix<double, 9, 1> computeResidual(
        const Eigen::Matrix3d& R_i, const Eigen::Vector3d& p_i, const Eigen::Vector3d& v_i,
        const Eigen::Matrix3d& R_j, const Eigen::Vector3d& p_j, const Eigen::Vector3d& v_j) const;
    
    // *** NEW: Analytical Jacobian (9x18) ***
    Eigen::Matrix<double, 9, 18> computeJacobian(
        const Eigen::Matrix3d& R_i, const Eigen::Vector3d& p_i, const Eigen::Vector3d& v_i,
        const Eigen::Matrix3d& R_j, const Eigen::Vector3d& p_j, const Eigen::Vector3d& v_j) const;
};

class ImuPreintegrator {
public:
    ImuPreintegrator(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba, const Eigen::Vector3d& gravity);
    void integrate(const std::vector<ImuMeasurement>& meas);
    const ImuPreintegration& getResult() const { return res_; }
private:
    Eigen::Vector3d bg_, ba_, g_;
    ImuPreintegration res_;
};
}
