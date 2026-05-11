#ifndef TASSEL_TOOLS_JACOBIAN_CHECKER_H_
#define TASSEL_TOOLS_JACOBIAN_CHECKER_H_

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

namespace tassel_tools {

/// Virtual base for numerical Jacobian verification.
///
/// Usage: subclass, implement plus() and evaluate(). Then call check().
///
///   struct MyChecker : public JacobianChecker {
///     MyChecker() : JacobianChecker(2, {6, 6, 1}) {}
///
///     bool plus(const double* x, const double* delta,
///               double* x_out, int block_idx) const override {
///       if (block_idx < 2) { /* SE3 plus */ } else { /* scalar plus */ }
///       return true;
///     }
///
///     void evaluate(const std::vector<double*>& params,
///                   double* residuals, double** jacobians) const override {
///       compute_jacobian(..., residuals, jacobians);
///     }
///   };
///
///   MyChecker checker;
///   checker.set_params({pose_i, pose_j, &inv_depth});
///   checker.check();
class JacobianChecker {
public:
    /// @param num_residuals       output dim of residual function
    /// @param param_dims          tangent dim of each parameter block
    /// @param param_storage_dims  storage dim of each parameter block (default = param_dims)
    JacobianChecker(
        int num_residuals, std::vector<int> param_dims, std::vector<int> param_storage_dims = {})
        : num_residuals_(num_residuals), param_dims_(std::move(param_dims)) {
        if (param_storage_dims.empty()) {
            param_storage_dims_ = param_dims_;
        } else {
            param_storage_dims_ = std::move(param_storage_dims);
        }
        const int B = static_cast<int>(param_dims_.size());
        numeric_jacs_.resize(B);
        analytic_jacs_.resize(B);
        residuals_.resize(num_residuals_);

        for (int b = 0; b < B; ++b) {
            analytic_jacs_[b].assign(num_residuals_ * param_dims_[b], 0.0);
            numeric_jacs_[b].assign(num_residuals_ * param_dims_[b], 0.0);
        }
    }

    virtual ~JacobianChecker() = default;

    // ── user-implemented ────────────────────────────────

    /// Perturb parameter block @p block_idx along tangent vector @p delta.
    virtual bool plus(const double* x, const double* delta, double* x_out, int block_idx) const = 0;

    /// Compute analytic residuals + Jacobians.
    /// @param params      parameter pointers (unperturbed)
    /// @param residuals   output, size = num_residuals_
    /// @param jacobians   output, jacobians[i] size = num_residuals_ * param_dims_[i]
    virtual void evaluate(
        const std::vector<double*>& params, double* residuals, double** jacobians) const = 0;

    // ── runner ───────────────────────────────────────────

    /// Set parameter pointers for the current iteration.
    void set_params(std::vector<double*> params) { params_ = std::move(params); }

    /// Compare analytic vs central-difference numeric Jacobians.
    bool check(double eps = 1e-7, double thresh = 1e-4) {
        const int B = static_cast<int>(param_dims_.size());

        // 1. compute analytic
        fill_analytic_ptrs();
        evaluate(params_, residuals_.data(), jac_ptr_vec_.data());
        const double* r_analytic = residuals_.data();

        // 2. numeric via central differences
        std::vector<double> r_plus(num_residuals_);
        std::vector<double> r_minus(num_residuals_);

        std::vector<double> param_copy;
        {
            int max_dim = 0;
            for (int d : param_storage_dims_) max_dim = std::max(max_dim, d);
            param_copy.resize(max_dim);
        }
        std::vector<double> delta;
        {
            int max_dim = 0;
            for (int d : param_dims_) max_dim = std::max(max_dim, d);
            delta.assign(max_dim, 0.0);
        }

        for (int b = 0; b < B; ++b) {
            const int D = param_dims_[b];

            std::vector<double*> params_perturb = params_;
            params_perturb[b] = param_copy.data();

            for (int d = 0; d < D; ++d) {
                delta[d] = eps;
                plus(params_[b], delta.data(), param_copy.data(), b);
                evaluate(params_perturb, r_plus.data(), nullptr);
                delta[d] = -eps;
                plus(params_[b], delta.data(), param_copy.data(), b);
                evaluate(params_perturb, r_minus.data(), nullptr);
                delta[d] = 0.0;

                for (int r = 0; r < num_residuals_; ++r) {
                    numeric_jacs_[b][r * D + d] = (r_plus[r] - r_minus[r]) / (2.0 * eps);
                }
            }
        }

        // 3. compare
        int num_bad = 0;
        for (int b = 0; b < B; ++b) {
            const int D = param_dims_[b];
            const double* Ja = analytic_jacs_[b].data();
            const double* Jn = numeric_jacs_[b].data();

            for (int d = 0; d < D; ++d) {
                for (int r = 0; r < num_residuals_; ++r) {
                    int idx = r * D + d;
                    double a = Ja[idx];
                    double n = Jn[idx];
                    double denom = std::max(std::abs(a), std::abs(n));
                    double abs_err = std::abs(a - n);
                    double rel_err = (denom > 1e-15) ? abs_err / denom : 0.0;

                    if (rel_err > thresh && abs_err > 1e-9) {
                        if (num_bad == 0) {
                            std::cerr << "[JacobianChecker] mismatches:\n";
                        }
                        std::cerr << "  block " << b << " [" << r << ", " << d
                                  << "]  analytic=" << a << " numeric=" << n
                                  << " rel_err=" << rel_err << "\n";
                        num_bad++;
                    }
                }
            }
        }

        if (num_bad == 0) {
            std::cout << "[JacobianChecker] all " << num_residuals_ * total_dim()
                      << " elements OK.\n";
        } else {
            std::cerr << "[JacobianChecker] " << num_bad << " mismatch(es).\n";
        }
        return num_bad == 0;
    }

    int num_residuals() const { return num_residuals_; }
    int total_dim() const {
        int s = 0;
        for (int d : param_dims_) s += d;
        return s;
    }

private:
    void fill_analytic_ptrs() {
        const int B = static_cast<int>(param_dims_.size());
        jac_ptr_vec_.resize(B);
        for (int b = 0; b < B; ++b) {
            jac_ptr_vec_[b] = analytic_jacs_[b].data();
        }
    }

    int num_residuals_;
    std::vector<int> param_dims_;
    std::vector<int> param_storage_dims_;

    std::vector<double*> params_;
    std::vector<double*> jac_ptr_vec_;
    std::vector<double> residuals_;

    std::vector<std::vector<double>> analytic_jacs_;
    std::vector<std::vector<double>> numeric_jacs_;
};

}  // namespace tassel_tools

#endif  // TASSEL_TOOLS_JACOBIAN_CHECKER_H_
