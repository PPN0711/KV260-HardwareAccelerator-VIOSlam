#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace delta_trust {

// ============================================================
// Sparse utilities for VIO Hessian
// ============================================================

struct SparseMetrics {
    int rows;
    int cols;
    int nnz;              // non-zero elements
    double density;       // nnz / (rows*cols)
    size_t memory_bytes;  // Estimated memory
    double sparsity;      // 1 - density
};

// Convert dense → sparse with threshold
Eigen::SparseMatrix<double> dense_to_sparse(const Eigen::MatrixXd& H,
                                            double threshold = 1e-12);

// Compute metrics for sparse matrix
SparseMetrics compute_metrics(const Eigen::SparseMatrix<double>& H);

// Compute metrics for dense matrix
SparseMetrics compute_metrics_dense(const Eigen::MatrixXd& H);

// Print metrics
void print_metrics(const SparseMetrics& m, const std::string& label);

// AMD reordering — returns permutation and reordered matrix
struct AMDResult {
    Eigen::SparseMatrix<double> H_reordered;
    Eigen::VectorXi permutation;   // permutation[i] = new index of old i
    Eigen::VectorXi inverse;       // inverse[permutation[i]] = i
    SparseMetrics metrics_before;
    SparseMetrics metrics_after;
    int fill_in_before;
    int fill_in_after;
};

AMDResult amd_reorder(const Eigen::SparseMatrix<double>& H);

// Apply permutation to vector
Eigen::VectorXd permute_vector(const Eigen::VectorXd& v,
                               const Eigen::VectorXi& perm);

// Apply inverse permutation
Eigen::VectorXd unpermute_vector(const Eigen::VectorXd& v,
                                 const Eigen::VectorXi& inv_perm);

// Compute fill-in of LDL^T factorization
int compute_fill_in(const Eigen::SparseMatrix<double>& H);

} // namespace delta_trust
