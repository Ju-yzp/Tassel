#ifndef TASSEL_CORE_CAM_CAMERA_FACTORY_H_
#define TASSEL_CORE_CAM_CAMERA_FACTORY_H_

#include <concepts>
#include <memory>
#include <string>
#include <unordered_map>

#include "cam/camera_equi.h"
#include "cam/camera_rad_tan.h"

namespace tassel_core {

enum class CameraModel { kRadTan, kEqui };

using Camera = std::unique_ptr<CameraBase>;

inline const std::unordered_map<std::string, CameraModel> kCameraModelMap = {
    {"radtan", CameraModel::kRadTan},
    {"equi", CameraModel::kEqui},
};

template <typename T>
concept CameraPointer = std::same_as<T, std::unique_ptr<CameraBase>> ||
    std::same_as<T, std::shared_ptr<CameraBase>> || std::same_as<T, std::weak_ptr<CameraBase>>;

class CameraFactory {
public:
    template <CameraPointer PtrType = std::unique_ptr<CameraBase>>
    static PtrType create(
        const std::string& model, const cv::Mat& k, const cv::Mat& dist_coeffs, int width,
        int height) {
        auto it = kCameraModelMap.find(model);
        if (it == kCameraModelMap.end()) {
            throw std::invalid_argument("Unknown camera model: " + model);
        }
        CameraBase* raw = nullptr;
        switch (it->second) {
            case CameraModel::kRadTan:
                raw = new CameraRadTan(k, dist_coeffs, width, height);
                break;
            case CameraModel::kEqui:
                raw = new CameraEqui(k, dist_coeffs, width, height);
                break;
        }
        return PtrType(raw);
    }
};

}  // namespace tassel_core
#endif  // TASSEL_CORE_CAM_CAMERA_FACTORY_H_
