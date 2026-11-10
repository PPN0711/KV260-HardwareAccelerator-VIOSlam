#include "stage0/ldlt_rank1.h"
#include <limits>
#include <algorithm>
namespace delta_trust {
UpdateResult ldlt_rank1_update(const Eigen::MatrixXd& L_in, const Eigen::VectorXd& D_in, const Eigen::VectorXd& v, double sigma, double d_floor) {
    const int n = (int)D_in.size();
    Eigen::MatrixXd L = L_in;
    Eigen::VectorXd D = D_in;
    Eigen::VectorXd w = v;
    double b = sigma;
    int fail_index = -1;
    for (int j = 0; j < n; ++j) {
        const double alpha = w(j);
        const double d_old = D(j);
        const double d_new = d_old + b * alpha * alpha;
        double gamma = 0.0, b_next = 0.0;
        if (d_new <= d_floor) {
            fail_index = j;
            D(j) = std::max(d_new, d_floor);
        } else {
            D(j) = d_new;
            gamma = alpha * b / d_new;
            b_next = d_old * b / d_new;
        }
        for (int r = j + 1; r < n; ++r) {
            const double w_r_old = w(r);
            w(r) = w_r_old - alpha * L(r, j);
            L(r, j) = L(r, j) + gamma * w(r);
        }
        b = b_next;
    }
    UpdateResult res;
    res.L = std::move(L);
    res.D = std::move(D);
    res.min_D = res.D.minCoeff();
    res.ok = (fail_index == -1);
    res.fail_index = fail_index;
    return res;
}
UpdateResult ldlt_update_rankk(const Eigen::MatrixXd& L_in, const Eigen::VectorXd& D_in, const Eigen::MatrixXd& V, const Eigen::VectorXd& signs, double d_floor) {
    const int k = (int)V.cols();
    Eigen::MatrixXd L = L_in;
    Eigen::VectorXd D = D_in;
    double worst_min_D = std::numeric_limits<double>::infinity();
    bool any_fail = false;
    int fail_index = -1;
    for (int c = 0; c < k; ++c) {
        UpdateResult r = ldlt_rank1_update(L, D, V.col(c), signs(c), d_floor);
        L = std::move(r.L);
        D = std::move(r.D);
        worst_min_D = std::min(worst_min_D, r.min_D);
        if (!r.ok) { any_fail = true; fail_index = r.fail_index; }
    }
    UpdateResult res;
    res.L = std::move(L);
    res.D = std::move(D);
    res.min_D = worst_min_D;
    res.ok = !any_fail;
    res.fail_index = fail_index;
    return res;
}
}
