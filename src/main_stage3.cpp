#include <iostream>
#include <iomanip>
#include <chrono>
#include "src/stage1/euroc_reader.h"
#include "stage1/imu_preintegration.h"
#include "stage1/reprojection_factor.h"
#include "stage1/factor_graph.h"
#include "stage3/optimizer.h"
#include "stage3/ate_evaluator.h"

using namespace delta_trust;
using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <euroc_path>\n";
        return 1;
    }

    std::cout << "========================================\n";
    std::cout << "  DeltaTrust-VIO - Stage 3\n";
    std::cout << "  Optimization + ATE Evaluation\n";
    std::cout << "========================================\n\n";

    auto t_start = high_resolution_clock::now();

    EurocReader reader(argv[1]);
    if (!reader.loadAll()) return 1;
    const auto& gt_all = reader.getGroundTruth();

    const int N_KF = 30;
    const int N_LM = 50;
    int stride = std::max(1, (int)gt_all.size() / N_KF);
    std::vector<GroundTruthPose> kfs;
    for (int i = 0; i < N_KF && i * stride < (int)gt_all.size(); ++i) {
        kfs.push_back(gt_all[i * stride]);
    }
    std::cout << "Using " << kfs.size() << " keyframes\n\n";

    GraphState state;
    Eigen::Vector3d initial_offset(1.0, -0.75, 0.5);
    for (size_t i = 0; i < kfs.size(); ++i) {
        state.R.push_back(kfs[i].orientation.toRotationMatrix());
        state.p.push_back(kfs[i].position + initial_offset);
        state.v.push_back(kfs[i].velocity);
    }

    std::cout << "=== Building Factor Graph ===\n";
    FactorGraph graph;

    for (size_t i = 0; i + 1 < kfs.size(); ++i) {
        auto imu_win = reader.getImuBetween(kfs[i].timestamp, kfs[i+1].timestamp);
        ImuPreintegrator integ(kfs[i].bias_gyro, kfs[i].bias_accel,
                               Eigen::Vector3d(0, 0, 9.81));
        integ.integrate(imu_win);
        graph.addImuFactor(i, i+1, integ.getResult());
    }
    std::cout << "  IMU factors: " << graph.getImuEdges().size() << "\n";

    for (int k = 0; k < N_LM; ++k) {
        Eigen::Vector3d P(3 + 0.5 * k, 2 - 0.1 * k, 2 + 0.3 * k);
        for (size_t i = 0; i < kfs.size(); ++i) {
            Eigen::Matrix3d R_cw = kfs[i].orientation.toRotationMatrix().transpose();
            Eigen::Vector3d t_cw = -R_cw * kfs[i].position;
            Eigen::Vector3d Pc = R_cw * P + t_cw;
            if (Pc.z() < 0.5 || Pc.z() > 20) continue;
            Eigen::Vector2d obs(
                Pc.x() / Pc.z() * reader.getCam0Calib().focal.x() + reader.getCam0Calib().principal.x(),
                Pc.y() / Pc.z() * reader.getCam0Calib().focal.y() + reader.getCam0Calib().principal.y());
            ReprojectionFactor f(P, obs, reader.getCam0Calib());
            graph.addReprojFactor(i, k, f, P, 0.01);
        }
    }
    std::cout << "  Graph: " << graph.numPoses() << " poses, "
              << graph.numPoints() << " points\n\n";

    std::cout << "=== Running Optimization ===\n";
    auto t_opt_start = high_resolution_clock::now();

    VIOOptimizer optimizer(graph, reader.getCam0Calib());
    auto result = optimizer.optimize(state, 100, 1e-6);

    auto t_opt_end = high_resolution_clock::now();
    double opt_ms = duration<double, std::milli>(t_opt_end - t_opt_start).count();

    std::cout << "\n  Optimization time: " << std::fixed << std::setprecision(1)
              << opt_ms << " ms\n";
    std::cout << "  Iterations: " << result.iterations << "\n";
    std::cout << "  Final cost: " << std::scientific << std::setprecision(3)
              << result.final_cost << "\n";
    std::cout << "  Converged: " << (result.converged ? "YES" : "NO") << "\n\n";

    std::cout << "=== Computing ATE ===\n";
    std::vector<Eigen::Vector3d> est_pos;
    std::vector<Eigen::Vector3d> gt_pos;
    for (size_t i = 0; i < result.state.p.size(); ++i) {
        est_pos.push_back(result.state.p[i]);
        gt_pos.push_back(kfs[i].position);
    }

    auto ate = compute_ate(est_pos, gt_pos);

    std::cout << "  Number of poses : " << ate.n_poses << "\n";
    std::cout << "  ATE RMSE        : " << std::fixed << std::setprecision(4)
              << ate.ate_rmse << " m\n";
    std::cout << "  ATE Mean        : " << ate.ate_mean << " m\n";
    std::cout << "  ATE Median      : " << ate.ate_median << " m\n";
    std::cout << "  ATE Std         : " << ate.ate_std << " m\n";
    std::cout << "  ATE Max         : " << ate.ate_max << " m\n";

    double gt_range = 0.0;
    for (size_t i = 1; i < gt_pos.size(); ++i) {
        gt_range += (gt_pos[i] - gt_pos[0]).norm();
    }
    gt_range /= (gt_pos.size() - 1);
    double rel_err = ate.ate_rmse / gt_range * 100.0;
    std::cout << "  Trajectory scale: " << gt_range << " m (avg)\n";
    std::cout << "  Relative error  : " << std::fixed << std::setprecision(2)
              << rel_err << " %\n\n";

    save_trajectory_csv("stage3_trajectory.csv", est_pos, gt_pos);
    std::cout << "  Trajectory saved: stage3_trajectory.csv\n\n";

    auto t_end = high_resolution_clock::now();
    double total_ms = duration<double, std::milli>(t_end - t_start).count();

    std::cout << "========================================\n";
    std::cout << "  STAGE 3 SUMMARY\n";
    std::cout << "========================================\n";
    std::cout << "  Optimization   : " << result.iterations << " iters, "
              << std::fixed << std::setprecision(1) << opt_ms << " ms\n";
    std::cout << "  ATE RMSE       : " << std::setprecision(4) << ate.ate_rmse << " m\n";
    std::cout << "  Relative error : " << std::setprecision(2) << rel_err << " %\n";
    std::cout << "  Converged      : " << (result.converged ? "YES" : "NO") << "\n";
    std::cout << "  Total time     : " << std::setprecision(1) << total_ms << " ms\n";
    std::cout << "========================================\n";
    std::cout << "  STAGE 3 COMPLETE\n";
    std::cout << "========================================\n";

    return 0;
}
