#include "stage1/schur.h"
namespace delta_trust {
SchurResult applySchur(const Eigen::MatrixXd& H, const Eigen::VectorXd& b, int n_pose, int n_point) {
    int dp = n_pose * 9, dl = n_point * 3;
    auto H_pp = H.topLeftCorner(dp, dp);
    auto H_pl = H.topRightCorner(dp, dl);
    auto H_lp = H.bottomLeftCorner(dl, dp);
    auto H_ll = H.bottomRightCorner(dl, dl);
    auto b_p = b.head(dp);
    auto b_l = b.tail(dl);
    Eigen::MatrixXd X = H_ll.ldlt().solve(H_lp);
    Eigen::VectorXd y = H_ll.ldlt().solve(b_l);
    SchurResult res;
    res.S = H_pp - H_pl * X;
    res.b_reduced = b_p - H_pl * y;
    return res;
}
}
