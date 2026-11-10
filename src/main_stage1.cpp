#include <iostream>
#include "stage1/euroc_reader.h"
#include "stage1/imu_preintegration.h"
#include "stage1/reprojection_factor.h"
#include "stage1/factor_graph.h"
#include "stage1/schur.h"
#include "stage0/ldlt.h"
#include "stage0/ldlt_rank1.h"
#include "stage0/dogleg.h"
#include "stage0/matrix_utils.h"
using namespace delta_trust;
int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <euroc_path>\n"; return 1; }
    std::cout << "========================================\n  DeltaTrust-VIO Stage 1\n========================================\n\n";
    EurocReader reader(argv[1]);
    if (!reader.loadAll()) return 1;
    const auto& gt_all = reader.getGroundTruth();
    const int N_KF = 20, N_LM = 30;
    int stride = std::max(1, (int)gt_all.size() / N_KF);
    std::vector<GroundTruthPose> kfs;
    for (int i = 0; i < N_KF && i*stride < (int)gt_all.size(); ++i) kfs.push_back(gt_all[i*stride]);
    std::cout << "Using " << kfs.size() << " keyframes\n";
    GraphState state;
    for (const auto& kf : kfs) {
        state.R.push_back(kf.orientation.toRotationMatrix());
        state.p.push_back(kf.position);
        state.v.push_back(kf.velocity);
    }
    FactorGraph graph;
    for (size_t i = 0; i + 1 < kfs.size(); ++i) {
        auto imu_win = reader.getImuBetween(kfs[i].timestamp, kfs[i+1].timestamp);
        ImuPreintegrator integ(kfs[i].bias_gyro, kfs[i].bias_accel, Eigen::Vector3d(0, 0, 9.81));
        integ.integrate(imu_win);
        graph.addImuFactor(i, i+1, integ.getResult());
    }
    std::cout << "Added " << graph.getImuEdges().size() << " IMU factors\n";
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
    std::cout << "Graph: " << graph.numPoses() << " poses, " << graph.numPoints() << " pts, dim = " << graph.stateDim() << "\n";
    std::cout << "\nBuilding Hessian...\n";
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    graph.buildHessian(state, H, b);
    std::cout << "H size: " << H.rows() << "x" << H.cols() << "\n";
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    std::cout << "H min eig: " << es.eigenvalues().minCoeff() << "\n";
    std::cout << "\nApplying Schur...\n";
    SchurResult schur = applySchur(H, b, graph.numPoses(), graph.numPoints());
    std::cout << "S size: " << schur.S.rows() << "x" << schur.S.cols() << "\n";
    std::cout << "S min eig: " << Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(schur.S).eigenvalues().minCoeff() << "\n";
    std::cout << "\n=== Stage 0 test ===\n";
    auto factor = ldlt_decompose(schur.S);
    std::cout << "LDL^T ok, min(D) = " << factor.D.minCoeff() << "\n";
    auto dl = dogleg_step(factor.L, factor.D, schur.b_reduced, 1.0);
    std::cout << "Dogleg kind: " << dl.kind << ", ||step|| = " << dl.step.norm() << "\n";
    Eigen::MatrixXd V = random_update_columns(schur.S.rows(), 2, 42);
    Eigen::VectorXd signs(2); signs << 1.0, 1.0;
    auto upd = ldlt_update_rankk(factor.L, factor.D, V, signs);
    std::cout << "Rank-1 update ok: " << (upd.ok ? "YES" : "NO") << "\n";
    std::cout << "\n========================================\n  STAGE 1 COMPLETE\n========================================\n";
    return 0;
}
