#include "stage1/euroc_reader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <yaml-cpp/yaml.h>
namespace delta_trust {
EurocReader::EurocReader(const std::string& p) : dataset_path_(p) {}
bool EurocReader::loadAll() {
    if (!loadCalibration(dataset_path_ + "/sensor.yaml")) return false;
    if (!loadImuData(dataset_path_ + "/imu0/data.csv")) return false;
    if (!loadGroundTruth(dataset_path_ + "/state_groundtruth_estimate0/data.csv")) return false;
    std::cout << "[EurocReader] " << imu_data_.size() << " IMU, " << gt_data_.size() << " GT\n";
    return true;
}
bool EurocReader::loadCalibration(const std::string& path) {
    try {
        YAML::Node cfg = YAML::LoadFile(path);
        auto parse = [](const YAML::Node& n) {
            CameraCalibration c;
            c.width = n["resolution"][0].as<int>();
            c.height = n["resolution"][1].as<int>();
            c.focal(0) = n["intrinsics"][0].as<double>();
            c.focal(1) = n["intrinsics"][1].as<double>();
            c.principal(0) = n["intrinsics"][2].as<double>();
            c.principal(1) = n["intrinsics"][3].as<double>();
            auto d = n["distortion_coeffs"];
            for (int i = 0; i < 4 && i < (int)d.size(); ++i) c.distortion(i) = d[i].as<double>();
            return c;
        };
        cam0_calib_ = parse(cfg["cam0"]);
        cam1_calib_ = parse(cfg["cam1"]);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "YAML error: " << e.what() << "\n";
        return false;
    }
}
bool EurocReader::loadImuData(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line; std::getline(f, line);
    while (std::getline(f, line)) {
        std::stringstream ss(line); std::string tok; std::vector<double> v;
        while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
        if (v.size() >= 7) {
            ImuMeasurement m;
            m.timestamp = v[0] * 1e-9;
            m.gyro << v[1], v[2], v[3];
            m.accel << v[4], v[5], v[6];
            imu_data_.push_back(m);
        }
    }
    return !imu_data_.empty();
}
bool EurocReader::loadGroundTruth(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line; std::getline(f, line);
    while (std::getline(f, line)) {
        std::stringstream ss(line); std::string tok; std::vector<double> v;
        while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
        if (v.size() >= 17) {
            GroundTruthPose p;
            p.timestamp = v[0] * 1e-9;
            p.position << v[1], v[2], v[3];
            p.orientation = Eigen::Quaterniond(v[4], v[5], v[6], v[7]);
            p.velocity << v[8], v[9], v[10];
            p.bias_gyro << v[11], v[12], v[13];
            p.bias_accel << v[14], v[15], v[16];
            gt_data_.push_back(p);
        }
    }
    return !gt_data_.empty();
}
std::vector<ImuMeasurement> EurocReader::getImuBetween(double t0, double t1) const {
    std::vector<ImuMeasurement> r;
    for (const auto& m : imu_data_) if (m.timestamp >= t0 && m.timestamp <= t1) r.push_back(m);
    return r;
}
std::vector<GroundTruthPose> EurocReader::getGtBetween(double t0, double t1) const {
    std::vector<GroundTruthPose> r;
    for (const auto& p : gt_data_) if (p.timestamp >= t0 && p.timestamp <= t1) r.push_back(p);
    return r;
}
}
