#include "stage1/reprojection_factor.h"
#include "stage1/imu_preintegration.h"
namespace delta_trust {
ReprojectionFactor::ReprojectionFactor(const Eigen::Vector3d& P, const Eigen::Vector2d& o, const CameraCalibration& c) : P_w_(P), obs_(o), calib_(c) {}
Eigen::Vector2d ReprojectionFactor::project(const Eigen::Vector3d& P) const { return Eigen::Vector2d(P.x()/P.z(), P.y()/P.z()); }
Eigen::Vector2d ReprojectionFactor::distort(const Eigen::Vector2d& p) const {
    double x = p.x(), y = p.y(); double r2 = x*x + y*y, r4 = r2*r2;
    double k1 = calib_.distortion(0), k2 = calib_.distortion(1), p1 = calib_.distortion(2), p2 = calib_.distortion(3);
    double rad = 1.0 + k1*r2 + k2*r4;
    Eigen::Vector2d o;
    o.x() = x*rad + 2*p1*x*y + p2*(r2 + 2*x*x);
    o.y() = y*rad + p1*(r2 + 2*y*y) + 2*p2*x*y;
    return o;
}
Eigen::Vector2d ReprojectionFactor::computeResidual(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) const {
    Eigen::Vector3d P = R * P_w_ + t;
    if (P.z() <= 0.01) return Eigen::Vector2d::Zero();
    Eigen::Vector2d p = distort(project(P));
    Eigen::Vector2d pix;
    pix.x() = p.x() * calib_.focal.x() + calib_.principal.x();
    pix.y() = p.y() * calib_.focal.y() + calib_.principal.y();
    return pix - obs_;
}
Eigen::Matrix<double, 2, 6> ReprojectionFactor::computeJacobianPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) const {
    Eigen::Vector3d P = R * P_w_ + t;
    double X = P.x(), Y = P.y(), Z = P.z(), Z2 = Z*Z;
    Eigen::Matrix<double, 2, 3> dpi;
    dpi << 1.0/Z, 0, -X/Z2, 0, 1.0/Z, -Y/Z2;
    Eigen::Matrix<double, 3, 6> dP;
    dP.block<3,3>(0,0) = -R * skew3(P_w_);
    dP.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
    Eigen::Matrix2d K; K << calib_.focal.x(), 0, 0, calib_.focal.y();
    return K * dpi * dP;
}
Eigen::Matrix<double, 2, 3> ReprojectionFactor::computeJacobianPoint(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) const {
    Eigen::Vector3d P = R * P_w_ + t;
    double X = P.x(), Y = P.y(), Z = P.z(), Z2 = Z*Z;
    Eigen::Matrix<double, 2, 3> dpi;
    dpi << 1.0/Z, 0, -X/Z2, 0, 1.0/Z, -Y/Z2;
    Eigen::Matrix2d K; K << calib_.focal.x(), 0, 0, calib_.focal.y();
    return K * dpi * R;
}
}
