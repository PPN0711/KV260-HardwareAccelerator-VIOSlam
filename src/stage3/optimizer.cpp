#include "stage3/optimizer.h"
#include "stage0/ldlt.h"
#include "stage0/ldlt_rank1.h"
#include "stage0/dogleg.h"
#include "stage1/imu_preintegration.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace delta_trust {

VIOOptimizer::VIOOptimizer(FactorGraph& g, const CameraCalibration& c)
    : graph_(g), calib_(c) {}

double VIOOptimizer::computeCost(const GraphState& state) {
    return graph_.computeCost(state);
}

void VIOOptimizer::updateState(GraphState& state, const Eigen::VectorXd& delta) {
    int n_poses = (int)state.R.size();
    int pose_dim = 9;
    for (int i = 0; i < n_poses; ++i) {
        Eigen::Vector3d dtheta = delta.segment<3>(i * pose_dim);
        state.R[i] = state.R[i] * expSO3(dtheta);
        state.p[i] += delta.segment<3>(i * pose_dim + 3);
        state.v[i] += delta.segment<3>(i * pose_dim + 6);
    }
}

OptimizeResult VIOOptimizer::optimize(const GraphState& initial,
                                       int max_iters,
                                       double cost_tol) {
    OptimizeResult res;
    res.state = initial;
    res.iterations = 0;
    res.converged = false;
    
    double prev_cost = computeCost(res.state);
    res.cost_history.push_back(prev_cost);
    
    std::cout << "  Iter  |  Cost       |  delta_norm  |  radius   |  status\n";
    std::cout << "  ------+-------------+--------------+-----------+--------\n";
    std::cout << "  0     | " << std::scientific << std::setprecision(3)
              << std::setw(11) << prev_cost << " | "
              << std::setw(12) << "-" << " | "
              << std::setw(9) << "-" << " | init\n";
    
    // *** CONSERVATIVE RADIUS ***
    double radius = 0.2;              // Start small
    const double max_radius = 0.5;    // Cap
    const double min_radius = 1e-6;
    int consecutive_rejects = 0;
    int consecutive_accepts = 0;
    
    for (int iter = 1; iter <= max_iters; ++iter) {
        res.iterations = iter;
        
        Eigen::MatrixXd H;
        Eigen::VectorXd b;
        graph_.buildHessian(res.state, H, b);
        
        SchurResult schur = applySchur(H, b, graph_.numPoses(), graph_.numPoints());
        
        LDLT factor;
        try {
            factor = ldlt_decompose(schur.S);
        } catch (const std::exception&) {
            std::cout << "  LDL^T failed at iter " << iter << "\n";
            break;
        }
        
        DoglegResult dl;
        try {
            dl = dogleg_step(factor.L, factor.D, schur.b_reduced, radius);
        } catch (const std::exception&) {
            std::cout << "  Dogleg failed at iter " << iter << "\n";
            break;
        }
        
        GraphState trial = res.state;
        updateState(trial, dl.step);
        double trial_cost = computeCost(trial);
        
        std::string status;
        
        // *** BLOW-UP DETECTION ***
        if (!std::isfinite(trial_cost) || trial_cost > prev_cost * 10.0) {
            // Severe blow-up: reject immediately, shrink hard
            status = "BLOWUP";
            radius *= 0.2;
            consecutive_rejects++;
            consecutive_accepts = 0;
        }
        else if (trial_cost < prev_cost) {
            // Accept
            res.state = trial;
            double reduction = (prev_cost - trial_cost) / prev_cost;
            prev_cost = trial_cost;
            status = "accept";
            consecutive_rejects = 0;
            consecutive_accepts++;
            
            // *** CONSERVATIVE GROWTH ***
            if (reduction > 0.7 && consecutive_accepts >= 2) {
                radius = std::min(radius * 1.5, max_radius);
                consecutive_accepts = 0;
            }
            // Else: keep radius unchanged
        }
        else {
            // Reject (mild)
            status = "reject";
            radius *= 0.5;
            consecutive_rejects++;
            consecutive_accepts = 0;
        }
        
        res.cost_history.push_back(trial_cost);
        
        std::cout << "  " << std::setw(4) << iter << " | "
                  << std::scientific << std::setprecision(3) << std::setw(11) << trial_cost
                  << " | " << std::setw(12) << dl.step.norm()
                  << " | " << std::fixed << std::setprecision(4) << std::setw(9) << radius
                  << " | " << status << "\n";
        
        // *** TERMINATION ***
        if (radius < min_radius) {
            res.converged = true;
            std::cout << "  Converged: radius < " << min_radius << "\n";
            break;
        }
        if (consecutive_rejects >= 15) {
            std::cout << "  Stopped: 15 consecutive rejects\n";
            break;
        }
        if (dl.step.norm() < cost_tol && status == "accept") {
            res.converged = true;
            std::cout << "  Converged: step norm < " << cost_tol << "\n";
            break;
        }
    }
    
    res.final_cost = prev_cost;
    return res;
}

} // namespace delta_trust
