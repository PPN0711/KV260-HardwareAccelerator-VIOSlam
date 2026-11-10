#pragma once
#include <Eigen/Dense>
#include "stage0/ldlt.h"
#include "stage0/ldlt_rank1.h"

namespace delta_trust {

// ============================================================
// Marginalization: remove the oldest pose from sliding window
// and update LDL^T factor using rank-1 downdate.
// ============================================================

struct MarginalizationResult {
    bool ok;
    int n_removed;         // number of pose blocks removed
    Eigen::MatrixXd S_new; // Reduced Hessian
    Eigen::VectorXd b_new;
    double downdate_norm;  // ||v|| of the downdate vector
};

// Remove pose at index pose_idx from state vector
// Original state: [pose_0 | pose_1 | ... | pose_N-1]
// After marginalization: only N-1 poses remain
MarginalizationResult marginalize_pose(const Eigen::MatrixXd& S,
                                       const Eigen::VectorXd& b,
                                       int n_poses,
                                       int pose_dim,   // 9 for VIO
                                       int pose_to_remove);

// Compare margin vs rebuild — verify rank-1 downdate is correct
struct MarginalizationComparison {
    double rel_err_S;      // ||S_downdate - S_rebuild|| / ||S_rebuild||
    double rel_err_b;
    bool consistent;
};

MarginalizationComparison compare_marginalization(
    const Eigen::MatrixXd& S,
    const Eigen::VectorXd& b,
    int n_poses,
    int pose_dim,
    int pose_to_remove);

} // namespace delta_trust
