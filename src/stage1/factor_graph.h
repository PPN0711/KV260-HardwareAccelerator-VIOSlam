#pragma once
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include "stage1/imu_preintegration.h"
#include "stage1/reprojection_factor.h"
namespace delta_trust {
struct GraphState {
    std::vector<Eigen::Matrix3d> R;
    std::vector<Eigen::Vector3d> p;
    std::vector<Eigen::Vector3d> v;
};
class FactorGraph {
public:
    void addImuFactor(int i, int j, const ImuPreintegration& preint, double w = 1.0);
    void addReprojFactor(int pose_idx, int point_idx, const ReprojectionFactor& f, const Eigen::Vector3d& P_w, double w = 1.0);
    void buildHessian(const GraphState& state, Eigen::MatrixXd& H, Eigen::VectorXd& b) const;
    int numPoses() const { return n_poses_; }
    int numPoints() const { return n_points_; }
    int stateDim() const { return n_poses_ * 9 + n_points_ * 3; }
    double computeCost(const GraphState& state) const;
    const std::vector<std::tuple<int,int>>& getImuEdges() const { return imu_edges_; }
private:
    int n_poses_ = 0, n_points_ = 0;
    struct ImuFactor { int i, j; ImuPreintegration preint; double w; };
    struct ReprojFactor { int pose_idx, point_idx; ReprojectionFactor factor; Eigen::Vector3d P_w; double w; };
    std::vector<ImuFactor> imu_factors_;
    std::vector<ReprojFactor> reproj_factors_;
    std::vector<std::tuple<int,int>> imu_edges_;
    std::vector<Eigen::Vector3d> points_;
};
}
