#ifndef TASSEL_CORE_INITIAL_DYNAMIC_INITIALIZER_H_
#define TASSEL_CORE_INITIAL_DYNAMIC_INITIALIZER_H_

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <memory>
#include <utility>
#include <vector>

#include "frond_end/feature_manager.h"

namespace tassel_core {
class DynamicInitializer {
public:
    /**
     * @brief 求解陀螺仪偏置增量.
     *
     * @param delta_q_dbg 预积分相对姿态对陀螺仪偏置的一阶线性雅可比 d(delta_q)/d(bg).
     * @param delta_qs IMU 预积分得到的相邻帧相对姿态 delta_q.
     * @param R_Cs_in_C0 相机坐标系在 C0 参考系下的姿态序列 R_C_i^C0.
     * @return Eigen::Vector3d 陀螺仪偏置增量.
     */
    Eigen::Vector3d solveGyroBias(
        const std::vector<Eigen::Matrix3d>& delta_q_dbg,
        const std::vector<Eigen::Matrix3d>& delta_qs,
        const std::vector<Eigen::Matrix3d>& R_Cs_in_C0) const;

    void solveRealtiveRotation(
        std::vector<std::pair<bool, Eigen::Matrix3d>>& all_Rij_with_state) const;

    void buildLinearSystem(
        std::vector<Eigen::Matrix3d>& Rs, std::vector<Eigen::Vector3d>& Ps,
        std::vector<Eigen::Matrix3d>& s, std::vector<double> all_ts, Eigen::Matrix3d ric,
        Eigen::Vector3d tic);

private:
    Eigen::Vector<double, 7> compute_coeff(Eigen::Matrix3d D, Eigen::Vector3d d, double g);

    std::shared_ptr<FeatureManager> feature_manager_;

    double min_rotation_angle_;
};
}  // namespace tassel_core
#endif
