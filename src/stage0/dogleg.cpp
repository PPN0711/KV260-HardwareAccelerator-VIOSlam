#include "stage0/dogleg.h"
#include "stage0/ldlt.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
namespace delta_trust {
DoglegResult dogleg_step(const Eigen::MatrixXd& L, const Eigen::VectorXd& D, const Eigen::VectorXd& q, double radius) {
    Eigen::VectorXd p_gn = ldlt_solve(L, D, q);
    const double norm_gn = p_gn.norm();
    if (norm_gn <= radius) return {p_gn, "gauss_newton"};
    Eigen::VectorXd Sq = apply_S(L, D, q);
    const double qq  = q.dot(q);
    const double qSq = q.dot(Sq);
    if (qSq <= 0.0) throw std::runtime_error("q^T S q <= 0");
    const double alpha = qq / qSq;
    Eigen::VectorXd p_sd = alpha * q;
    const double norm_sd = p_sd.norm();
    if (norm_sd >= radius) return {(radius / norm_sd) * p_sd, "steepest_descent"};
    Eigen::VectorXd diff = p_gn - p_sd;
    const double a = diff.dot(diff);
    const double b = 2.0 * p_sd.dot(diff);
    const double c = p_sd.dot(p_sd) - radius * radius;
    const double disc = b * b - 4.0 * a * c;
    double tau = (a > 0.0) ? (-b + std::sqrt(std::max(disc, 0.0))) / (2.0 * a) : 0.0;
    tau = std::clamp(tau, 0.0, 1.0);
    return {p_sd + tau * diff, "dogleg_interp"};
}
}
