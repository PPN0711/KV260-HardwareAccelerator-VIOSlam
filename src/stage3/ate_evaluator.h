#pragma once
#include <Eigen/Dense>
#include <vector>
#include "stage1/euroc_reader.h"

namespace delta_trust {

struct ATEResult {
    double ate_rmse;        // RMSE of position error
    double ate_mean;
    double ate_median;
    double ate_std;
    double ate_max;
    int n_poses;
    Eigen::Matrix4d alignment_transform;   // SE(3) alignment
};

// Compute ATE with Umeyama SE(3) alignment
ATEResult compute_ate(const std::vector<Eigen::Vector3d>& estimated,
                      const std::vector<Eigen::Vector3d>& ground_truth);

// Save trajectory for plotting
void save_trajectory_csv(const std::string& path,
                         const std::vector<Eigen::Vector3d>& estimated,
                         const std::vector<Eigen::Vector3d>& ground_truth);

} // namespace delta_trust
