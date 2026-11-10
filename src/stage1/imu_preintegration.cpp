#include "stage1/imu_preintegration.h"
#include <cmath>

namespace delta_trust {

Eigen::Matrix3d skew3(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return S;
}

Eigen::Matrix3d expSO3(const Eigen::Vector3d& omega) {
    double th = omega.norm();
    if (th < 1e-10) return Eigen::Matrix3d::Identity() + skew3(omega);
    return Eigen::AngleAxisd(th, omega / th).toRotationMatrix();
}

Eigen::Vector3d logSO3(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd aa(R);
    return aa.angle() * aa.axis();
}

Eigen::Matrix<double, 9, 1> ImuPreintegration::computeResidual(
    const Eigen::Matrix3d& R_i, const Eigen::Vector3d& p_i, const Eigen::Vector3d& v_i,
    const Eigen::Matrix3d& R_j, const Eigen::Vector3d& p_j, const Eigen::Vector3d& v_j) const {
    Eigen::Matrix<double, 9, 1> r;
    r.segment<3>(0) = logSO3(delta_R.transpose() * R_i.transpose() * R_j);
    r.segment<3>(3) = R_i.transpose() * (v_j - v_i - gravity * dt_sum) - delta_v;
    r.segment<3>(6) = R_i.transpose() * (p_j - p_i - v_i * dt_sum
                     - 0.5 * gravity * dt_sum * dt_sum) - delta_p;
    return r;
}

// Numerical Jacobian — safe, no sign bugs
Eigen::Matrix<double, 9, 18> ImuPreintegration::computeJacobian(
    const Eigen::Matrix3d& R_i, const Eigen::Vector3d& p_i, const Eigen::Vector3d& v_i,
    const Eigen::Matrix3d& R_j, const Eigen::Vector3d& p_j, const Eigen::Vector3d& v_j) const {
    
    Eigen::Matrix<double, 9, 18> J;
    const double eps = 1e-6;
    
    for (int k = 0; k < 18; ++k) {
        int pose = (k < 9) ? 0 : 1;
        int local = (k < 9) ? k : k - 9;
        
        Eigen::Matrix3d R_p = (pose == 0) ? R_i : R_j;
        Eigen::Vector3d p_p = (pose == 0) ? p_i : p_j;
        Eigen::Vector3d v_p = (pose == 0) ? v_i : v_j;
        
        Eigen::Vector3d dtheta = Eigen::Vector3d::Zero();
        Eigen::Vector3d dp = Eigen::Vector3d::Zero();
        Eigen::Vector3d dv = Eigen::Vector3d::Zero();
        
        if (local < 3) dtheta(local) = eps;
        else if (local < 6) dp(local-3) = eps;
        else dv(local-6) = eps;
        
        // Plus
        Eigen::Matrix3d R_p_plus = R_p * expSO3(dtheta);
        Eigen::Matrix3d R_p_minus = R_p * expSO3(-dtheta);
        auto r_plus = computeResidual(
            (pose == 0) ? R_p_plus : R_i,  (pose == 0) ? p_p + dp : p_i,  (pose == 0) ? v_p + dv : v_i,
            (pose == 1) ? R_p_plus : R_j,  (pose == 1) ? p_p + dp : p_j,  (pose == 1) ? v_p + dv : v_j);
        auto r_minus = computeResidual(
            (pose == 0) ? R_p_minus : R_i, (pose == 0) ? p_p - dp : p_i,  (pose == 0) ? v_p - dv : v_i,
            (pose == 1) ? R_p_minus : R_j, (pose == 1) ? p_p - dp : p_j,  (pose == 1) ? v_p - dv : v_j);
        
        J.col(k) = (r_plus - r_minus) / (2.0 * eps);
    }
    
    return J;
}

ImuPreintegrator::ImuPreintegrator(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba, const Eigen::Vector3d& gravity)
    : bg_(bg), ba_(ba), g_(gravity) {
    res_.gravity = gravity;
}

void ImuPreintegrator::integrate(const std::vector<ImuMeasurement>& meas) {
    for (size_t i = 0; i + 1 < meas.size(); ++i) {
        double dt = meas[i+1].timestamp - meas[i].timestamp;
        if (dt <= 0 || dt > 0.1) continue;
        Eigen::Vector3d omega = meas[i].gyro - bg_;
        Eigen::Vector3d accel = meas[i].accel - ba_;
        Eigen::Matrix3d dR = expSO3(omega * dt);
        Eigen::Vector3d a_world = res_.delta_R * accel;
        res_.delta_p += res_.delta_v * dt + 0.5 * a_world * dt * dt;
        res_.delta_v += a_world * dt;
        res_.delta_R = res_.delta_R * dR;
        res_.dt_sum += dt;
        res_.n_measurements++;
    }
    
    // Fixed covariance scale — same for all factors
    const double sigma_imu = 1.0;
    res_.cov = Eigen::Matrix<double, 9, 9>::Identity() * sigma_imu * sigma_imu;
    res_.cov_inv = res_.cov.inverse();
}
}
