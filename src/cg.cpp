#include <omp.h>
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using uint8 = const unsigned short;

void set_thread_count(uint8 thread_count){
    omp_set_num_threads(thread_count);
}

double compute_inner_product(const std::vector<double>& __restrict__ a, const std::vector<double>& __restrict__ b){
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (std::size_t i = 0; i < a.size(); i++){
        sum += a[i] * b[i];
    }
    return sum;
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

std::vector<double> solve_cg(const std::vector<double>& __restrict__ matrix_data, const std::vector<int>& __restrict__ row_idx, const std::vector<int>& __restrict__ row_ptr, const std::vector<double>& __restrict__ b, const std::vector<double>& __restrict__ x0, int max_iter, double tol){
    if (b.size() != x0.size()){
        throw std::invalid_argument("Size of b and x0 must be the same");
    }
    if (b.size() != (row_ptr.size() - 1)){
        throw std::invalid_argument("Size of b must match the number of rows in the matrix");
    }

    std::vector<double> rk(b.size(), 0.0);
    std::vector<double> pk(b.size(), 0.0);
    std::vector<double> x(b.size(), 0.0);
    x = x0;
    std::vector<double> Ax = compute_matrix_vector_product(matrix_data, row_idx, row_ptr, x);
    for (std::size_t i = 0; i < rk.size(); i++){
        rk[i] = b[i] - Ax[i];
    }
    pk = rk;
    double rkT_rk = compute_inner_product(rk, rk);
    double alpha, beta, rkT_rk_new;

    for (int k = 0; k < max_iter; k++){
        std::vector<double> Apk = compute_matrix_vector_product(matrix_data, row_idx, row_ptr, pk);
        alpha = rkT_rk / compute_inner_product(pk, Apk); 
        for (std::size_t i = 0; i < x.size(); i++){
            x[i] += alpha * pk[i];
            rk[i] -= alpha * Apk[i];
        }
        rkT_rk_new = compute_inner_product(rk, rk);
        if (std::sqrt(rkT_rk_new) < tol){
            break;
        }
        beta = rkT_rk_new / rkT_rk;
        for (std::size_t i = 0; i < pk.size(); i++){
            pk[i] = rk[i] + beta * pk[i];
        }
        std::cout << "Iteration: " << k + 1 << "\tResidual: " << std::sqrt(rkT_rk_new) << std::endl; 
        rkT_rk = rkT_rk_new;
    }
    return x;
}

py::array_t<double> CG_Solve(py::array_t<double> data, py::array_t<int> row_idx, py::array_t<int> row_ptr, py::array_t<double> b, py::array_t<double> x0, int max_iter, uint8 thread_count, double tol=1e-6){
    set_thread_count(thread_count);
    py::buffer_info data_buf = data.request();
    py::buffer_info row_idx_buf = row_idx.request();
    py::buffer_info row_ptr_buf = row_ptr.request();
    py::buffer_info b_buf = b.request();
    py::buffer_info x0_buf = x0.request();
    const double *matrix_data = static_cast<double *>(data_buf.ptr);
    const int *row_idx_data = static_cast<int *>(row_idx_buf.ptr);
    const int *row_ptr_data = static_cast<int *>(row_ptr_buf.ptr);
    const double *b_data = static_cast<double *>(b_buf.ptr);
    const double *x0_data = static_cast<double *>(x0_buf.ptr);

    std::vector<double> x_sol = solve_cg(
        std::vector<double>(matrix_data, matrix_data + data_buf.size),
        std::vector<int>(row_idx_data, row_idx_data + row_idx_buf.size),
        std::vector<int>(row_ptr_data, row_ptr_data + row_ptr_buf.size),
        std::vector<double>(b_data, b_data + b_buf.size),
        std::vector<double>(x0_data, x0_data + x0_buf.size),
        max_iter,
        tol
    );

    py::array_t<double> out(x_sol.size());
    py::buffer_info out_buf = out.request();
    double *out_ptr = static_cast<double *>(out_buf.ptr);
    std::copy(x_sol.begin(), x_sol.end(), out_ptr);
    return out;
}

PYBIND11_MODULE(CG_Solver, m){
    m.doc() = "Conjugate Gradient Solver implemented in C++ with OpenMP for CSR Matrices";
    m.def("CG_Solve", &CG_Solve, "Solve a linear system using the Conjugate Gradient method", py::arg("data"), py::arg("row_idx"), py::arg("row_ptr"), py::arg("b"), py::arg("x0"), py::arg("max_iter"), py::arg("thread_count"), py::arg("tol"));
}