#ifndef ZED_COMMON_HPP
#define ZED_COMMON_HPP

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>

// LED on /sys/class/leds (board power LED) 
class Led {
public:
    explicit Led(const std::string& name) : name_(name) {
        std::ifstream probe("/sys/class/leds/" + name_ + "/brightness");
        ok_ = probe.good();
        if (ok_) off();
    }
    ~Led() { off(); }
    void set(bool state) {
        if (!ok_) return;
        std::system(("echo none > /sys/class/leds/" + name_ + "/trigger").c_str());
        std::system(("echo " + std::string(state ? "1" : "0") +
                     " > /sys/class/leds/" + name_ + "/brightness").c_str());
    }
    void on()  { set(true); }
    void off() { set(false); }
    bool ok() const { return ok_; }
private:
    std::string name_;
    bool ok_ = false;
};

// Full-rate IMU logger
// Logs the camera's IMU stream deduplicated by the sample's own timestamp, in
// EuRoC units (gyro rad/s, accel m/s^2) on the camera reference clock.
// stop() MUST be called before zed.close() (it polls the same handle).
class ImuLogger {
public:
    ImuLogger(sl::Camera& zed, const std::string& csv_path) : zed_(zed) {
        csv_.open(csv_path);
        if (csv_) {
            csv_ << std::setprecision(9);
            csv_ << "#timestamp [ns],w_x [rad/s],w_y [rad/s],w_z [rad/s],"
                    "a_x [m/s^2],a_y [m/s^2],a_z [m/s^2]\n";
            running_ = true;
            worker_  = std::thread(&ImuLogger::run, this);
        }
    }
    ~ImuLogger() { stop(); }
    void stop() {
        if (running_.exchange(false)) {
            if (worker_.joinable()) worker_.join();
            if (csv_.is_open()) csv_.flush();
        }
    }
    bool ok() const { return csv_.is_open(); }
    uint64_t count() const { return count_.load(); }
private:
    void run() {
        const double DEG2RAD = M_PI / 180.0;
        sl::SensorsData sd;
        uint64_t last_ts = 0;
        while (running_) {
            if (zed_.getSensorsData(sd, sl::TIME_REFERENCE::CURRENT)
                    == sl::ERROR_CODE::SUCCESS) {
                const uint64_t ts = sd.imu.timestamp.getNanoseconds();
                if (ts != 0 && ts != last_ts) {
                    last_ts = ts;
                    const sl::float3& w = sd.imu.angular_velocity;     // deg/s
                    const sl::float3& a = sd.imu.linear_acceleration;  // m/s^2
                    csv_ << ts << ','
                         << (w.x * DEG2RAD) << ',' << (w.y * DEG2RAD) << ','
                         << (w.z * DEG2RAD) << ','
                         << a.x << ',' << a.y << ',' << a.z << '\n';
                    ++count_;
                }
            }
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
    }
    sl::Camera& zed_;
    std::ofstream csv_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> count_{0};
};

inline cv::Mat slMat2cvMat(sl::Mat& m) {
    int t = -1;
    switch (m.getDataType()) {
        case sl::MAT_TYPE::U8_C1:  t = CV_8UC1;  break;
        case sl::MAT_TYPE::U8_C3:  t = CV_8UC3;  break;
        case sl::MAT_TYPE::U8_C4:  t = CV_8UC4;  break;
        case sl::MAT_TYPE::F32_C1: t = CV_32FC1; break;
        default: throw std::runtime_error("Unsupported sl::Mat type");
    }
    return cv::Mat(static_cast<int>(m.getHeight()),
                   static_cast<int>(m.getWidth()), t,
                   m.getPtr<sl::uchar1>(sl::MEM::CPU),
                   m.getStepBytes(sl::MEM::CPU));
}

// Minimal .conf (INI) reader
inline std::map<std::string, double>
read_conf_section(const std::string& path, const std::string& section) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open conf: " + path);
    std::map<std::string, double> kv;
    std::string line, cur;
    auto trim = [](std::string s) {
        const char* ws = " \t\r\n";
        size_t a = s.find_first_not_of(ws);
        size_t b = s.find_last_not_of(ws);
        return (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
    };
    auto upper = [](std::string s) {
        for (char& c : s) c = std::toupper((unsigned char)c);
        return s;
    };
    const std::string want = upper(section);
    bool in = false;
    while (std::getline(f, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#' || t[0] == ';') continue;
        if (t.front() == '[' && t.back() == ']') {
            cur = upper(trim(t.substr(1, t.size() - 2)));
            in = (cur == want);
            continue;
        }
        if (!in) continue;
        size_t eq = t.find('=');
        if (eq == std::string::npos) continue;
        std::string k = upper(trim(t.substr(0, eq)));
        std::string v = trim(t.substr(eq + 1));
        try { kv[k] = std::stod(v); } catch (...) {}
    }
    return kv;
}

#endif // ZED_COMMON_HPP
