#include "stage2/sparse_utils.h"
#include <iostream>
#include <iomanip>
#include <Eigen/OrderingMethods>

namespace delta_trust {

Eigen::SparseMatrix<double> dense_to_sparse(const Eigen::MatrixXd& H,
                                            double threshold) {
    const int n = (int)H.rows();
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n * 20);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double val = H(i, j);
            if (std::abs(val) > threshold) {
                triplets.emplace_back(i, j, val);
            }
        }
    }
    
    Eigen::SparseMatrix<double> H_sparse(n, n);
    H_sparse.setFromTriplets(triplets.begin(), triplets.end());
    H_sparse.makeCompressed();
    return H_sparse;
}

SparseMetrics compute_metrics(const Eigen::SparseMatrix<double>& H) {
    SparseMetrics m;
    m.rows = (int)H.rows();
    m.cols = (int)H.cols();
    m.nnz = (int)H.nonZeros();
    m.density = (double)m.nnz / (m.rows * m.cols);
    m.sparsity = 1.0 - m.density;
    m.memory_bytes = (size_t)m.nnz * (8 + 4 + 4);
    return m;
}

SparseMetrics compute_metrics_dense(const Eigen::MatrixXd& H) {
    SparseMetrics m;
    m.rows = (int)H.rows();
    m.cols = (int)H.cols();
    m.nnz = 0;
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            if (std::abs(H(i, j)) > 1e-12) m.nnz++;
    m.density = (double)m.nnz / (m.rows * m.cols);
    m.sparsity = 1.0 - m.density;
    m.memory_bytes = (size_t)m.rows * m.cols * 8;
    return m;
}

void print_metrics(const SparseMetrics& m, const std::string& label) {
    std::cout << "  " << label << ":\n";
    std::cout << "    Size      : " << m.rows << " x " << m.cols << "\n";
    std::cout << "    Non-zeros : " << m.nnz << "\n";
    std::cout << "    Density   : " << std::scientific 
              << std::setprecision(2) << m.density << "\n";
    std::cout << "    Sparsity  : " << std::fixed 
              << std::setprecision(2) << (100.0 * m.sparsity) << "%\n";
    std::cout << "    Memory    : " << std::fixed 
              << std::setprecision(2) << (m.memory_bytes / 1024.0) << " KB\n";
}

AMDResult amd_reorder(const Eigen::SparseMatrix<double>& H) {
    AMDResult res;
    res.metrics_before = compute_metrics(H);
    res.fill_in_before = compute_fill_in(H);
    
    // Compute AMD ordering
    Eigen::AMDOrdering<int> amd;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm;
    amd(H.selfadjointView<Eigen::Lower>(), perm);
    
    res.permutation = perm.indices();
    
    // === FIX: build inverse permutation manually ===
    res.inverse.resize(perm.indices().size());
    for (int i = 0; i < perm.indices().size(); ++i) {
        res.inverse(perm.indices()(i)) = i;
    }
    
    // Apply permutation: H_reordered = P^T * H * P
    res.H_reordered = H.twistedBy(perm.inverse());
    
    res.metrics_after = compute_metrics(res.H_reordered);
    res.fill_in_after = compute_fill_in(res.H_reordered);
    return res;
}

Eigen::VectorXd permute_vector(const Eigen::VectorXd& v,
                               const Eigen::VectorXi& perm) {
    Eigen::VectorXd out(v.size());
    for (int i = 0; i < v.size(); ++i)
        out(i) = v(perm(i));
    return out;
}

Eigen::VectorXd unpermute_vector(const Eigen::VectorXd& v,
                                 const Eigen::VectorXi& inv_perm) {
    Eigen::VectorXd out(v.size());
    for (int i = 0; i < v.size(); ++i)
        out(inv_perm(i)) = v(i);
    return out;
}

int compute_fill_in(const Eigen::SparseMatrix<double>& H) {
    // === FIX: Convert TriangularView → SparseMatrix ===
    Eigen::SparseMatrix<double> H_copy = H;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(H_copy);
    if (solver.info() != Eigen::Success) return -1;
    
    // matrixL() returns TriangularView — convert to SparseMatrix
    Eigen::SparseMatrix<double> L = solver.matrixL();
    return (int)L.nonZeros();
}

} // namespace delta_trust
