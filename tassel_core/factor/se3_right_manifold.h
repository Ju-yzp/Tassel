#ifndef TASSEL_CORE_SE3_RIGHT_MANIFOLD_H_
#define TASSEL_CORE_SE3_RIGHT_MANIFOLD_H_

#include <ceres/manifold.h>

namespace tassel_core {
class SE3RightManifold : public ceres::Manifold {
public:
    bool Plus(const double* x, const double* delta, double* x_plus_delta) const override;
    bool PlusJacobian(const double* x, double* jacobian) const override;
    bool Minus(const double* y, const double* x, double* y_minus_x) const override;
    bool MinusJacobian(const double* x, double* jacobian) const override;
    int AmbientSize() const override { return 6; }
    int TangentSize() const override { return 6; }
};
}  // namespace tassel_core
#endif /* TASSEL_CORE_SE3_RIGHT_MANIFOLD_H_ */
