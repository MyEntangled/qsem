#include <omp.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using uint8 = const unsigned short;

void set_thread_count(uint8 thread_count){
    omp_set_num_threads(thread_count);
}

extern "C" {
    void dstev_(char* jobz, int* n, double* d, double* e, double* z, int* ldz, double* work, int* info);
}

double compute_inner_product(const std::vector<double>& __restrict__ a, const std::vector<double>& __restrict__ b){
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (std::size_t i = 0; i < a.size(); i++){
        sum += a[i] * b[i];
    }
    return sum;
}

double compute_L2_norm(const std::vector<double>& __restrict__ vec){
    double inner_product = compute_inner_product(vec, vec);
    return std::sqrt(inner_product);
}

std::vector<double> compute_matrix_vector_product(const std::vector<double>& __restrict__ matrix_data, const std::vector<int>& __restrict__ row_idx, const std::vector<int>& __restrict__ row_ptr, const std::vector<double>& __restrict__ x){
    const std::size_t size = x.size();
    std::vector<double> result(size, 0.0);
    for (std::size_t i = 0; i < size; i++){
        double entry = 0.0;
        #pragma omp parallel for reduction(+:entry)
        for (std::size_t j = row_ptr[i]; j < row_ptr[i+1]; j++){
            entry += matrix_data[j] * x[row_idx[j]];
        }
        result[i] = entry;
    }
    return result;
}

struct LanczosDecomposition {
    std::vector<std::vector<double>> coefficients;
    std::vector<std::vector<double>> basis;
};

// std::vector<double> re_orthogonalize(){

// }

LanczosDecomposition lanczos(const std::vector<double>& __restrict__ matrix_data, const std::vector<int>& __restrict__ row_idx, const std::vector<int>& __restrict__ row_ptr, const std::vector<double>& x0, const int max_iter){
    const std::size_t mat_size = row_ptr.size() - 1;
    std::vector<double> q0(mat_size, 0.0);
    double alpha, beta;
    std::vector<std::vector<double>> coefficients;
    coefficients.reserve(static_cast<std::size_t>(std::max(0, max_iter)));
    std::vector<std::vector<double>> basis;
    basis.reserve(static_cast<std::size_t>(std::max(0, max_iter)));
    std::vector<double> qk(mat_size, 0.0);
    beta = 0.0;

    double x_norm = compute_L2_norm(x0);
    #pragma omp parallel for
    for (std::size_t i = 0; i < mat_size; i++){
        qk[i] = x0[i] / x_norm;
    }

    for (int k = 0; k < max_iter; k++){
        basis.push_back(qk);
        std::vector<double> v = compute_matrix_vector_product(matrix_data, row_idx, row_ptr, qk);
        alpha = compute_inner_product(qk, v);
        #pragma omp parallel for
        for (std::size_t i = 0; i < mat_size; i++){
            v[i] -= alpha * qk[i] + beta * q0[i];
        }
        beta = compute_L2_norm(v);
        coefficients.push_back({alpha, beta});
        if (beta < 1e-10){
            break;
        }
        q0 = qk;
        #pragma omp parallel for
        for (std::size_t i = 0; i < mat_size; i++){
            qk[i] = v[i] / beta;
        }
    }
    return {coefficients, basis};
}

// std::vector<std::vector<double>> build_tridiagonal_matrix(const std::vector<std::vector<double>>& lanczos_results){
//     const std::size_t size = lanczos_results.size();
//     std::vector<std::vector<double>> tridiagonal(size, std::vector<double>(size, 0.0));
//     for (std::size_t i = 0; i < size; i++){
//         tridiagonal[i][i] = lanczos_results[i][0];
//         if (i > 0 && i < size - 1){
//             tridiagonal[i][i-1] = lanczos_results[i][1];
//             tridiagonal[i-1][i] = lanczos_results[i][1];
//         }
//         if (i == size - 1){
//             tridiagonal[i][i-1] = lanczos_results[i][1];
//             tridiagonal[i-1][i] = lanczos_results[i][1];
//         }
//     }
//     return tridiagonal;
// }

std::pair<std::vector<double>, std::vector<std::vector<double>>> compute_m_eigenpairs(const std::vector<std::vector<double>>& lanczos_results, int num_eigenvalues, const bool find_max){
    int n = static_cast<int>(lanczos_results.size());
    if (n <= 0){
        return {{}, {}};
    }
    std::vector<double> diagonal(n), off_diagonal(std::max(0, n - 1));
    for (int i = 0; i < n; i++){
        diagonal[i] = lanczos_results[i][0];
        if (i < n - 1){
            off_diagonal[i] = lanczos_results[i][1];
        }
    }
    char jobz = 'V';
    int ldz = n;
    int info = 0;
    std::vector<double> z(static_cast<std::size_t>(n) * n, 0.0);
    std::vector<double> work(std::max(1, 2*n - 2), 0.0);

    dstev_(&jobz, &n, diagonal.data(), off_diagonal.data(), z.data(), &ldz, work.data(), &info);
    if (info != 0){
        throw std::runtime_error("Error in dstev_: " + std::to_string(info));
    }
    num_eigenvalues = std::max(1, std::min(num_eigenvalues, n));
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    if (find_max){
        std::reverse(idx.begin(), idx.end());
    }
    std::vector<double> selected_eigenvalues(num_eigenvalues);
    std::vector<std::vector<double>> selected_eigenvectors(num_eigenvalues, std::vector<double>(n, 0.0));
    for (int k = 0; k < num_eigenvalues; k++){
        selected_eigenvalues[k] = diagonal[idx[k]];
        for (int i = 0; i < n; i++){
            selected_eigenvectors[k][i] = z[i + static_cast<size_t>(idx[k]) * n];
        }
    }
    return {selected_eigenvalues, selected_eigenvectors};
}

std::vector<std::vector<double>> reconstruct_full_eigenvectors(const std::vector<std::vector<double>>& basis, const std::vector<std::vector<double>>& tridiagonal_eigenvectors){
    if (basis.empty() || tridiagonal_eigenvectors.empty()){
        return {};
    }
    const std::size_t krylov_dim = basis.size();
    const std::size_t vector_size = basis[0].size();
    std::vector<std::vector<double>> full_eigenvectors(tridiagonal_eigenvectors.size(), std::vector<double>(vector_size, 0.0));

    for (std::size_t ev = 0; ev < tridiagonal_eigenvectors.size(); ev++){
        for (std::size_t k = 0; k < krylov_dim; k++){
            const double weight = tridiagonal_eigenvectors[ev][k];
            #pragma omp parallel for
            for (std::size_t i = 0; i < vector_size; i++){
                full_eigenvectors[ev][i] += weight * basis[k][i];
            }
        }

        const double norm = compute_L2_norm(full_eigenvectors[ev]);
        if (norm > 0.0){
            #pragma omp parallel for
            for (std::size_t i = 0; i < vector_size; i++){
                full_eigenvectors[ev][i] /= norm;
            }
        }
    }
    return full_eigenvectors;
}

std::pair<std::vector<double>, std::vector<std::vector<double>>> compute_eigenvalues(const std::vector<double>& matrix_data, const std::vector<int>& row_idx, const std::vector<int>& row_ptr, const std::vector<double>& x0, const int max_iter, const int num_eigenvalues, const bool find_max){
    auto decomposition = lanczos(matrix_data, row_idx, row_ptr, x0, max_iter);
    auto [eigenvals, tridiagonal_eigenvecs] = compute_m_eigenpairs(decomposition.coefficients, num_eigenvalues, find_max);
    auto full_eigenvecs = reconstruct_full_eigenvectors(decomposition.basis, tridiagonal_eigenvecs);
    return {eigenvals, full_eigenvecs};
}

py::list Solve_Lanczos(py::array_t<double> data, py::array_t<int> row_idx, py::array_t<int> row_ptr, py::array_t<double> x0, int max_iter, int num_eigenvalues, bool find_max, uint8 thread_count){
    set_thread_count(thread_count);
    py::buffer_info data_info = data.request();
    py::buffer_info row_idx_info = row_idx.request();
    py::buffer_info row_ptr_info = row_ptr.request();
    py::buffer_info x0_info = x0.request();

    const double* matrix_data = static_cast<double *>(data_info.ptr);
    const int* row_idx_data = static_cast<int *>(row_idx_info.ptr);
    const int* row_ptr_data = static_cast<int *>(row_ptr_info.ptr);
    const double* x0_data = static_cast<double *>(x0_info.ptr);

    auto [eigenvals, eigenvecs] = compute_eigenvalues(std::vector<double>(matrix_data, matrix_data + data_info.size), std::vector<int>(row_idx_data, row_idx_data + row_idx_info.size), std::vector<int>(row_ptr_data, row_ptr_data + row_ptr_info.size), std::vector<double>(x0_data, x0_data + x0_info.size), max_iter, num_eigenvalues, find_max);
    py::list result;
    for (std::size_t i = 0; i < eigenvals.size(); i++){
        py::array_t<double> eigenvec(eigenvecs[i].size());
        py::buffer_info eigenvec_info = eigenvec.request();
        double* eigenvec_data = static_cast<double *>(eigenvec_info.ptr);
        std::copy(eigenvecs[i].begin(), eigenvecs[i].end(), eigenvec_data);
        result.append(py::make_tuple(eigenvals[i], eigenvec));  
    }
    return result;
}

PYBIND11_MODULE(Solve_Lanczos, m){
    m.doc() = "Lanczos algorithm for computing eigenvalues and eigenvectors of large sparse matrices.";
    m.def("solve_lanczos", &Solve_Lanczos, "Compute eigenvalues and eigenvectors using the Lanczos algorithm",
          py::arg("data"), py::arg("row_idx"), py::arg("row_ptr"), py::arg("x0"), py::arg("max_iter"), py::arg("num_eigenvalues"), py::arg("find_max"), py::arg("thread_count") = 1);
}