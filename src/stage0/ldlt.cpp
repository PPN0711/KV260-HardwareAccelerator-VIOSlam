#include "stage0/ldlt.h"
#include <stdexcept>
#include <string>
namespace delta_trust {
LDLT ldlt_decompose(const Eigen::MatrixXd& A) {
    const int n = (int)A.rows();
    Eigen::MatrixXd L = Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd D = Eigen::VectorXd::Zero(n);
    Eigen::MatrixXd Awork = A;
    for (int j = 0; j < n; ++j) {
        D(j) = Awork(j, j);
        if (D(j) <= 0.0) throw std::runtime_error("Not SPD at " + std::to_string(j));
        for (int i = j + 1; i < n; ++i) L(i, j) = Awork(i, j) / D(j);
        for (int i = j + 1; i < n; ++i)
            for (int k = j + 1; k < n; ++k)
                Awork(i, k) -= L(i, j) * D(j) * L(k, j);
    }
    return {L, D};
}
Eigen::VectorXd ldlt_solve(const Eigen::MatrixXd& L, const Eigen::VectorXd& D, const Eigen::VectorXd& rhs) {
    const int n = (int)D.size();
    Eigen::VectorXd y = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < n; ++i) {
        double s = rhs(i);
        for (int j = 0; j < i; ++j) s -= L(i, j) * y(j);
        y(i) = s;
    }
    Eigen::VectorXd z = y.array() / D.array();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    for (int i = n - 1; i >= 0; --i) {
        double s = z(i);
        for (int j = i + 1; j < n; ++j) s -= L(j, i) * x(j);
        x(i) = s;
    }
    return x;
}
Eigen::VectorXd apply_S(const Eigen::MatrixXd& L, const Eigen::VectorXd& D, const Eigen::VectorXd& x) {
    Eigen::VectorXd y = L.transpose() * x;
    y = D.array() * y.array();
    return L * y;
}
Eigen::MatrixXd reconstruct_S(const Eigen::MatrixXd& L, const Eigen::VectorXd& D) {
    Eigen::MatrixXd LD = L;
    LD = LD.array().colwise() * D.array();
    return LD * L.transpose();
}
}
