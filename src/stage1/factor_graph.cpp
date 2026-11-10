#include "stage1/factor_graph.h"
namespace delta_trust {

void FactorGraph::addImuFactor(int i, int j, const ImuPreintegration& p, double w) {
    imu_factors_.push_back({i, j, p, w});
    imu_edges_.emplace_back(i, j);
    n_poses_ = std::max(n_poses_, std::max(i, j) + 1);
}

void FactorGraph::addReprojFactor(int pi, int pti, const ReprojectionFactor& f, const Eigen::Vector3d& P, double w) {
    reproj_factors_.push_back({pi, pti, f, P, w});
    n_points_ = std::max(n_points_, pti + 1);
    if ((int)points_.size() <= pti) points_.resize(pti + 1, Eigen::Vector3d::Zero());
    points_[pti] = P;
}

// *** NEW: computeCost ***
double FactorGraph::computeCost(const GraphState& state) const {
    double cost = 0.0;
    for (const auto& f : imu_factors_) {
        auto r = f.preint.computeResidual(
            state.R[f.i], state.p[f.i], state.v[f.i],
            state.R[f.j], state.p[f.j], state.v[f.j]);
        cost += 0.5 * f.w * r.dot(f.preint.cov_inv * r);
    }
    for (const auto& f : reproj_factors_) {
        auto r = f.factor.computeResidual(
            state.R[f.pose_idx], state.p[f.pose_idx]);
        cost += 0.5 * f.w * r.dot(r);
    }
    return cost;
}

void FactorGraph::buildHessian(const GraphState& state, Eigen::MatrixXd& H, Eigen::VectorXd& b) const {
    int n = stateDim();
    H = Eigen::MatrixXd::Zero(n, n);
    b = Eigen::VectorXd::Zero(n);
    auto poseOff = [](int idx) { return idx * 9; };

    for (const auto& f : imu_factors_) {
        auto r = f.preint.computeResidual(
            state.R[f.i], state.p[f.i], state.v[f.i],
            state.R[f.j], state.p[f.j], state.v[f.j]);
        auto J = f.preint.computeJacobian(
            state.R[f.i], state.p[f.i], state.v[f.i],
            state.R[f.j], state.p[f.j], state.v[f.j]);

        Eigen::Matrix<double, 9, 9> W = f.w * f.preint.cov_inv;
        Eigen::Matrix<double, 18, 18> H_local = J.transpose() * W * J;
        Eigen::Matrix<double, 18, 1>  b_local = J.transpose() * W * r;

        for (int a = 0; a < 18; ++a) {
            int ga = (a < 9) ? poseOff(f.i) + a : poseOff(f.j) + (a - 9);
            for (int c = 0; c < 18; ++c) {
                int gc = (c < 9) ? poseOff(f.i) + c : poseOff(f.j) + (c - 9);
                H(ga, gc) += H_local(a, c);
            }
            b(ga) -= b_local(a);
        }
    }

    int point_start = n_poses_ * 9;
    for (const auto& f : reproj_factors_) {
        auto r = f.factor.computeResidual(state.R[f.pose_idx], state.p[f.pose_idx]);
        auto Jp = f.factor.computeJacobianPose(state.R[f.pose_idx], state.p[f.pose_idx]);
        auto Jl = f.factor.computeJacobianPoint(state.R[f.pose_idx], state.p[f.pose_idx]);
        int pi = poseOff(f.pose_idx);
        int pti = point_start + f.point_idx * 3;
        Eigen::Matrix2d W = f.w * Eigen::Matrix2d::Identity();
        Eigen::Matrix<double, 6, 6> H_pp = Jp.transpose() * W * Jp;
        Eigen::Matrix<double, 6, 3> H_pl = Jp.transpose() * W * Jl;
        Eigen::Matrix<double, 3, 6> H_lp = Jl.transpose() * W * Jp;
        Eigen::Matrix3d H_ll = Jl.transpose() * W * Jl;
        H.block<6,6>(pi, pi) += H_pp;
        H.block<6,3>(pi, pti) += H_pl;
        H.block<3,6>(pti, pi) += H_lp;
        H.block<3,3>(pti, pti) += H_ll;
        b.segment<6>(pi) -= Jp.transpose() * W * r;
        b.segment<3>(pti) -= Jl.transpose() * W * r;
    }
}

} // namespace delta_trust
