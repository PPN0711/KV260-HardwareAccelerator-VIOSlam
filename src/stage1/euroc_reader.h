#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
namespace delta_trust {
struct ImuMeasurement {
    double timestamp;
    Eigen::Vector3d gyro;
    Eigen::Vector3d accel;
};
struct GroundTruthPose {
    double timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d velocity;
    Eigen::Vector3d bias_gyro;
    Eigen::Vector3d bias_accel;
};
struct CameraCalibration {
    int width = 0, height = 0;
    Eigen::Vector2d focal = Eigen::Vector2d::Zero();
    Eigen::Vector2d principal = Eigen::Vector2d::Zero();
    Eigen::Vector4d distortion = Eigen::Vector4d::Zero();
    Eigen::Matrix4d T_BS = Eigen::Matrix4d::Identity();
};
class EurocReader {
public:
    explicit EurocReader(const std::string& dataset_path);
    bool loadAll();
    const std::vector<ImuMeasurement>& getImuData() const { return imu_data_; }
    const std::vector<GroundTruthPose>& getGroundTruth() const { return gt_data_; }
    const CameraCalibration& getCam0Calib() const { return cam0_calib_; }
    std::vector<ImuMeasurement> getImuBetween(double t0, double t1) const;
    std::vector<GroundTruthPose> getGtBetween(double t0, double t1) const;
private:
    std::string dataset_path_;
    std::vector<ImuMeasurement> imu_data_;
    std::vector<GroundTruthPose> gt_data_;
    CameraCalibration cam0_calib_, cam1_calib_;
    bool loadCalibration(const std::string& yaml_path);
    bool loadImuData(const std::string& csv_path);
    bool loadGroundTruth(const std::string& csv_path);
};
}
