#include <iostream>
#include <iomanip>
#include <chrono>
#include "stage1/euroc_reader.h"
#include "stage1/imu_preintegration.h"
#include "stage1/reprojection_factor.h"
#include "stage1/factor_graph.h"
#include "stage1/schur.h"
#include "stage2/sparse_utils.h"
#include "stage2/marginalization.h"
#include "stage0/ldlt.h"
#include "stage0/ldlt_rank1.h"

using namespace delta_trust;
using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <euroc_path>\n";
        return 1;
    }
    
    std::cout << "========================================\n";
    std::cout << "  DeltaTrust-VIO — Stage 2\n";
    std::cout << "  Sparse + AMD + Marginalization\n";
    std::cout << "========================================\n\n";
    
    // === 1. Load EuRoC ===
    EurocReader reader(argv[1]);
    if (!reader.loadAll()) return 1;
    const auto& gt_all = reader.getGroundTruth();
    
    // === 2. Select keyframes ===
    const int N_KF = 20, N_LM = 30;
    int stride = std::max(1, (int)gt_all.size() / N_KF);
    std::vector<GroundTruthPose> kfs;
    for (int i = 0; i < N_KF && i*stride < (int)gt_all.size(); ++i)
        kfs.push_back(gt_all[i*stride]);
    std::cout << "Using " << kfs.size() << " keyframes\n";
    
    // === 3. Build state + graph ===
    GraphState state;
    for (const auto& kf : kfs) {
        state.R.push_back(kf.orientation.toRotationMatrix());
        state.p.push_back(kf.position);
        state.v.push_back(kf.velocity);
    }
    
    FactorGraph graph;
    for (size_t i = 0; i + 1 < kfs.size(); ++i) {
        auto imu_win = reader.getImuBetween(kfs[i].timestamp, kfs[i+1].timestamp);
        ImuPreintegrator integ(kfs[i].bias_gyro, kfs[i].bias_accel,
                               Eigen::Vector3d(0, 0, 9.81));
        integ.integrate(imu_win);
        graph.addImuFactor(i, i+1, integ.getResult());
    }
    
    for (int k = 0; k < N_LM; ++k) {
        Eigen::Vector3d P(3 + 0.5*k, 2 - 0.1*k, 2 + 0.3*k);
        for (size_t i = 0; i < kfs.size(); ++i) {
            Eigen::Matrix3d R_cw = kfs[i].orientation.toRotationMatrix().transpose();
            Eigen::Vector3d t_cw = -R_cw * kfs[i].position;
            Eigen::Vector3d Pc = R_cw * P + t_cw;
            if (Pc.z() < 0.5 || Pc.z() > 20) continue;
            Eigen::Vector2d obs(
                Pc.x()/Pc.z() * reader.getCam0Calib().focal.x() + reader.getCam0Calib().principal.x(),
                Pc.y()/Pc.z() * reader.getCam0Calib().focal.y() + reader.getCam0Calib().principal.y());
            ReprojectionFactor f(P, obs, reader.getCam0Calib());
            graph.addReprojFactor(i, k, f, P, 1.0);
        }
    }
    
    // === 4. Build Hessian DENSE ===
    std::cout << "\n=== 4a. Build Dense Hessian ===\n";
    Eigen::MatrixXd H_dense;
    Eigen::VectorXd b;
    auto t0 = high_resolution_clock::now();
    graph.buildHessian(state, H_dense, b);
    auto t1 = high_resolution_clock::now();
    double t_dense_ms = duration<double, std::milli>(t1 - t0).count();
    std::cout << "  Time: " << std::fixed << std::setprecision(1) 
              << t_dense_ms << " ms\n";
    std::cout << "  Size: " << H_dense.rows() << " x " << H_dense.cols() << "\n";
    
    auto metrics_dense = compute_metrics_dense(H_dense);
    print_metrics(metrics_dense, "Dense H");
    
    // === 5. Convert DENSE → SPARSE ===
    std::cout << "\n=== 5. Convert to Sparse ===\n";
    auto H_sparse = dense_to_sparse(H_dense, 1e-12);
    auto metrics_sparse = compute_metrics(H_sparse);
    print_metrics(metrics_sparse, "Sparse H");
    
    double mem_ratio = (double)metrics_sparse.memory_bytes 
                     / (double)metrics_dense.memory_bytes;
    std::cout << "  Memory reduction: " << std::fixed << std::setprecision(2)
              << (100.0 * (1.0 - mem_ratio)) << "%\n";
    
    // === 6. AMD Reordering ===
    std::cout << "\n=== 6. AMD Ordering ===\n";
    auto amd = amd_reorder(H_sparse);
    
    print_metrics(amd.metrics_before, "Before AMD");
    print_metrics(amd.metrics_after, "After AMD");
    
    if (amd.fill_in_before > 0 && amd.fill_in_after > 0) {
        double fill_reduction = 100.0 * (1.0 - (double)amd.fill_in_after 
                                              / (double)amd.fill_in_before);
        std::cout << "  Fill-in before : " << amd.fill_in_before << "\n";
        std::cout << "  Fill-in after  : " << amd.fill_in_after << "\n";
        std::cout << "  Reduction      : " << std::fixed << std::setprecision(1)
                  << fill_reduction << "%\n";
    }
    
    // === 7. Schur complement on sparse ===
    std::cout << "\n=== 7. Schur Complement (sparse) ===\n";
    SchurResult schur = applySchur(H_dense, b, graph.numPoses(), graph.numPoints());
    std::cout << "  S size: " << schur.S.rows() << " x " << schur.S.cols() << "\n";
    auto S_sparse = dense_to_sparse(schur.S, 1e-12);
    auto metrics_S = compute_metrics(S_sparse);
    print_metrics(metrics_S, "Sparse S");
    
    // === 8. Marginalization test ===
    std::cout << "\n=== 8. Marginalization (remove oldest pose) ===\n";
    int pose_dim = 9;
    int n_poses_after_schur = graph.numPoses();
    
    // Test 1: Remove pose 0
    auto cmp0 = compare_marginalization(schur.S, schur.b_reduced,
                                         n_poses_after_schur, pose_dim, 0);
    std::cout << "  Remove pose 0:\n";
    std::cout << "    rel_err(S) = " << std::scientific 
              << std::setprecision(2) << cmp0.rel_err_S << "\n";
    std::cout << "    rel_err(b) = " << std::scientific 
              << std::setprecision(2) << cmp0.rel_err_b << "\n";
    std::cout << "    Consistent: " << (cmp0.consistent ? "YES" : "NO") << "\n";
    
    // Test 2: Remove middle pose
    auto cmp5 = compare_marginalization(schur.S, schur.b_reduced,
                                         n_poses_after_schur, pose_dim, 5);
    std::cout << "  Remove pose 5:\n";
    std::cout << "    rel_err(S) = " << std::scientific 
              << std::setprecision(2) << cmp5.rel_err_S << "\n";
    std::cout << "    Consistent: " << (cmp5.consistent ? "YES" : "NO") << "\n";
    
    // Test 3: Remove last pose
    auto cmpLast = compare_marginalization(schur.S, schur.b_reduced,
                                            n_poses_after_schur, pose_dim, 
                                            n_poses_after_schur - 1);
    std::cout << "  Remove last pose:\n";
    std::cout << "    rel_err(S) = " << std::scientific 
              << std::setprecision(2) << cmpLast.rel_err_S << "\n";
    std::cout << "    Consistent: " << (cmpLast.consistent ? "YES" : "NO") << "\n";
    
    // === 9. Rank-1 downdate after marginalization ===
    std::cout << "\n=== 9. Rank-1 Downdate Test ===\n";
    auto factor = ldlt_decompose(schur.S);
    std::cout << "  Original S: min(D) = " << std::fixed << std::setprecision(4)
              << factor.D.minCoeff() << "\n";
    
    // Simulate removing a small update via downdate
    Eigen::VectorXd v_test = Eigen::VectorXd::Random(schur.S.rows()) * 0.01;
    auto upd = ldlt_rank1_update(factor.L, factor.D, v_test, -1.0);
    std::cout << "  After downdate: min(D) = " << std::fixed << std::setprecision(4)
              << upd.min_D << "\n";
    std::cout << "  Downdate ok: " << (upd.ok ? "YES" : "NO") << "\n";
    
    // === 10. Summary ===
    std::cout << "\n========================================\n";
    std::cout << "  STAGE 2 SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "  Sparse memory:  " << metrics_sparse.memory_bytes / 1024 
              << " KB (vs " << metrics_dense.memory_bytes / 1024 << " KB dense)\n";
    std::cout << "  Sparsity:       " << std::fixed << std::setprecision(1) 
              << (100.0 * metrics_sparse.sparsity) << "%\n";
    if (amd.fill_in_before > 0 && amd.fill_in_after > 0) {
        std::cout << "  AMD fill-in:    " 
                  << (100.0 * (1.0 - (double)amd.fill_in_after / amd.fill_in_before))
                  << "% reduction\n";
    }
    std::cout << "  Marginalization: " 
              << (cmp0.consistent ? "VERIFIED" : "FAILED") << "\n";
    std::cout << "  Rank-1 downdate: " 
              << (upd.ok ? "OK" : "FAIL") << "\n";
    std::cout << "========================================\n";
    std::cout << "  STAGE 2 COMPLETE\n";
    std::cout << "========================================\n";
    
    return 0;
}
