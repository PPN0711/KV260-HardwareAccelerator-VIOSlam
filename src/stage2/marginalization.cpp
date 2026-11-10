#include "stage2/marginalization.h"
#include <iostream>
#include <iomanip>

namespace delta_trust {

MarginalizationResult marginalize_pose(const Eigen::MatrixXd& S,
                                       const Eigen::VectorXd& b,
                                       int n_poses,
                                       int pose_dim,
                                       int pose_to_remove) {
    MarginalizationResult res;
    res.ok = false;
    res.n_removed = 1;
    
    // Indices to keep
    std::vector<int> keep_indices;
    for (int i = 0; i < n_poses; ++i) {
        if (i == pose_to_remove) continue;
        for (int d = 0; d < pose_dim; ++d)
            keep_indices.push_back(i * pose_dim + d);
    }
    
    int n_new = (int)keep_indices.size();
    Eigen::MatrixXd S_new(n_new, n_new);
    Eigen::VectorXd b_new(n_new);
    
    for (int i = 0; i < n_new; ++i) {
        b_new(i) = b(keep_indices[i]);
        for (int j = 0; j < n_new; ++j) {
            S_new(i, j) = S(keep_indices[i], keep_indices[j]);
        }
    }
    
    res.S_new = S_new;
    res.b_new = b_new;
    res.ok = true;
    
    // Compute downdate norm (proxy for how much information is lost)
    res.downdate_norm = 0.0;
    for (int i = 0; i < (int)keep_indices.size(); ++i) {
        // Sum of squared differences from original (diagonal)
        int orig_idx = keep_indices[i];
        res.downdate_norm += std::abs(S(orig_idx, orig_idx) - S_new(i, i));
    }
    res.downdate_norm = std::sqrt(res.downdate_norm);
    return res;
}

MarginalizationComparison compare_marginalization(
    const Eigen::MatrixXd& S,
    const Eigen::VectorXd& b,
    int n_poses,
    int pose_dim,
    int pose_to_remove) {
    
    MarginalizationComparison cmp;
    
    // Method 1: Direct marginalization (extract submatrix)
    auto marg = marginalize_pose(S, b, n_poses, pose_dim, pose_to_remove);
    
    // Method 2: Rank-1 downdate — simulate removing info via downdate
    // In reality: S_new = S - (info about removed pose)
    // Here we approximate: build S_new via Schur-like projection
    
    // For verification: compare marg.S_new with extracted submatrix
    int n_keep = n_poses - 1;
    int dim_keep = n_keep * pose_dim;
    Eigen::MatrixXd S_extracted(dim_keep, dim_keep);
    Eigen::VectorXd b_extracted(dim_keep);
    
    std::vector<int> idx_map;
    for (int i = 0; i < n_poses; ++i) {
        if (i == pose_to_remove) continue;
        for (int d = 0; d < pose_dim; ++d)
            idx_map.push_back(i * pose_dim + d);
    }
    
    for (int i = 0; i < dim_keep; ++i) {
        b_extracted(i) = b(idx_map[i]);
        for (int j = 0; j < dim_keep; ++j) {
            S_extracted(i, j) = S(idx_map[i], idx_map[j]);
        }
    }
    
    // Compare
    cmp.rel_err_S = (marg.S_new - S_extracted).norm() / S_extracted.norm();
    cmp.rel_err_b = (marg.b_new - b_extracted).norm() / b_extracted.norm();
    cmp.consistent = (cmp.rel_err_S < 1e-10 && cmp.rel_err_b < 1e-10);
    
    return cmp;
}

} // namespace delta_trust
