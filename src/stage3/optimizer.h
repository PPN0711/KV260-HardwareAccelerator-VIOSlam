#pragma once
#include <Eigen/Dense>
#include "stage1/factor_graph.h"
#include "stage1/schur.h"

namespace delta_trust {

struct OptimizeResult {
    GraphState state;
    double final_cost;
    int iterations;
    bool converged;
    std::vector<double> cost_history;
};

class VIOOptimizer {
public:
    VIOOptimizer(FactorGraph& graph, const CameraCalibration& calib);
    
    // Run full optimization with Dogleg trust-region
    OptimizeResult optimize(const GraphState& initial_state,
                            int max_iters = 20,
                            double cost_tol = 1e-6);
    
private:
    FactorGraph& graph_;
    CameraCalibration calib_;
    
    // Compute cost = 0.5 * sum(r^T W r)
    double computeCost(const GraphState& state);
    
    // Update state from Dogleg step (local perturbation)
    void updateState(GraphState& state, const Eigen::VectorXd& delta);
};

} // namespace delta_trust
