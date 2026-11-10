#include "stage3/ate_evaluator.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace delta_trust {

// Umeyama alignment: find R, t, s that align source to target
static Eigen::Matrix4d umeyama_alignment(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst) {
    
    int n = (int)src.size();
    Eigen::Vector3d mean_src = Eigen::Vector3d::Zero();
    Eigen::Vector3d mean_dst = Eigen::Vector3d::Zero();
    
    for (int i = 0; i < n; ++i) {
        mean_src += src[i];
        mean_dst += dst[i];
    }
    mean_src /= n;
    mean_dst /= n;
    
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    double var_src = 0.0;
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d ds = src[i] - mean_src;
        Eigen::Vector3d dd = dst[i] - mean_dst;
        cov += dd * ds.transpose();
        var_src += ds.squaredNorm();
    }
    cov /= n;
    var_src /= n;
    
    // SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, 
        Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    if ((U * V.transpose()).determinant() < 0) D(2,2) = -1;
    
    Eigen::Matrix3d R = U * D * V.transpose();
    double s = 1.0;  // No scale for ATE (trajectory is metric)
    Eigen::Vector3d t = mean_dst - s * R * mean_src;
    
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = s * R;
    T.block<3,1>(0,3) = t;
    return T;
}

ATEResult compute_ate(const std::vector<Eigen::Vector3d>& est,
                     const std::vector<Eigen::Vector3d>& gt) {
    ATEResult res;
    int n = (int)std::min(est.size(), gt.size());
    res.n_poses = n;
    
    if (n == 0) return res;
    
    // Align estimated to GT
    Eigen::Matrix4d T = umeyama_alignment(est, gt);
    res.alignment_transform = T;
    
    // Compute errors after alignment
    std::vector<double> errors;
    errors.reserve(n);
    for (int i = 0; i < n; ++i) {
        Eigen::Vector4d p_homo(est[i].x(), est[i].y(), est[i].z(), 1.0);
        Eigen::Vector4d aligned = T * p_homo;
        Eigen::Vector3d err = aligned.head<3>() - gt[i];
        errors.push_back(err.norm());
    }
    
    // Statistics
    double sum = 0.0, sum_sq = 0.0, max_err = 0.0;
    for (double e : errors) {
        sum += e;
        sum_sq += e * e;
        max_err = std::max(max_err, e);
    }
    
    res.ate_mean = sum / n;
    res.ate_rmse = std::sqrt(sum_sq / n);
    res.ate_max = max_err;
    
    // Median
    std::vector<double> sorted = errors;
    std::sort(sorted.begin(), sorted.end());
    res.ate_median = (n % 2 == 0) ? 
        (sorted[n/2 - 1] + sorted[n/2]) / 2.0 : sorted[n/2];
    
    // Std
    double var = 0.0;
    for (double e : errors) var += (e - res.ate_mean) * (e - res.ate_mean);
    res.ate_std = std::sqrt(var / n);
    
    return res;
}

void save_trajectory_csv(const std::string& path,
                         const std::vector<Eigen::Vector3d>& est,
                         const std::vector<Eigen::Vector3d>& gt) {
    std::ofstream f(path);
    f << "idx,est_x,est_y,est_z,gt_x,gt_y,gt_z,error\n";
    int n = (int)std::min(est.size(), gt.size());
    for (int i = 0; i < n; ++i) {
        f << i << ","
          << est[i].x() << "," << est[i].y() << "," << est[i].z() << ","
          << gt[i].x() << "," << gt[i].y() << "," << gt[i].z() << ","
          << (est[i] - gt[i]).norm() << "\n";
    }
}

} // namespace delta_trust
