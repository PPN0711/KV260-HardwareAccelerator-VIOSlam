#pragma once
#include <Eigen/Dense>
#include "stage1/euroc_reader.h"
namespace delta_trust {
class ReprojectionFactor {
public:
    ReprojectionFactor(const Eigen::Vector3d& P_w, const Eigen::Vector2d& obs, const CameraCalibration& calib);
    Eigen::Vector2d computeResidual(const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& t_cw) const;
    Eigen::Matrix<double, 2, 6> computeJacobianPose(const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& t_cw) const;
    Eigen::Matrix<double, 2, 3> computeJacobianPoint(const Eigen::Matrix3d& R_cw, const Eigen::Vector3d& t_cw) const;
private:
    Eigen::Vector3d P_w_;
    Eigen::Vector2d obs_;
    CameraCalibration calib_;
    Eigen::Vector2d project(const Eigen::Vector3d& P) const;
    Eigen::Vector2d distort(const Eigen::Vector2d& p) const;
};
}
