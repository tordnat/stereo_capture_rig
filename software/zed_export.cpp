// Reads a directory produced by zed_record:
//   <in>/recording.svo, <in>/imu0/data.csv, <in>/SN<serial>.conf
//
// Both formats export UNRECTIFIED (raw) images — COLMAP and EuRoC are both
// designed around raw images + a distortion model, not pre-rectified ones.
//
//   --format colmap (default):
//       cam0/data/*.png  cam1/data/*.png  cam{0,1}/data.csv
//       cameras.txt              (PINHOLE by default, or OPENCV via --camera-model)
//       rig_config.json          (cam_from_rig = inverse(stereo_transform), w-first quat)
//       cam0_list.txt cam1_list.txt
//       reconstruct.sh           (params filled from cameras.txt; CASPAR backend)
//       imu0/data.csv            (copied)
//
//   --format euroc:
//       mav0/cam0/data/*.png  mav0/cam1/data/*.png  mav0/cam{0,1}/data.csv
//       mav0/cam{0,1}/sensor.yaml   (T_BS = cam pose in IMU/body frame; radtan distortion)
//       mav0/imu0/data.csv          (copied)  mav0/imu0/sensor.yaml
//
// Camera model note: COLMAP PINHOLE ignores lens distortion (small on ZED, and
// the mapper absorbs the rest); --camera-model opencv writes the factory radtan
// so COLMAP models/refines it. EuRoC always records the radtan in sensor.yaml.

#include "zed_common.hpp"

#include <array>
#include <experimental/filesystem>
#include <future>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::experimental::filesystem;

struct Qd { double x, y, z, w; };
struct Vd { double x, y, z; };

static Qd quat_conj(const Qd& q) {
    return {-q.x, -q.y, -q.z, q.w};
}

static Vd quat_rot(const Qd& q, const Vd& v) {
    double tx = 2*(q.y*v.z - q.z*v.y);
    double ty = 2*(q.z*v.x - q.x*v.z);
    double tz = 2*(q.x*v.y - q.y*v.x);
    return { v.x + q.w*tx + (q.y*tz - q.z*ty),
             v.y + q.w*ty + (q.z*tx - q.x*tz),
             v.z + q.w*tz + (q.x*ty - q.y*tx) };
}

static std::array<double, 9> quat_to_rotmat(const Qd& q) {
    double x=q.x, y=q.y, z=q.z, w=q.w;
    return {{ 1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w),
              2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w),
              2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y) }};
}

static std::string find_conf(const std::string& in_dir) {
    if (!fs::exists(in_dir)) return "";
    for (auto& e : fs::directory_iterator(in_dir)) {
        const std::string n = e.path().filename().string();
        if (n.size() > 5 && n.substr(0, 2) == "SN" && n.substr(n.size()-5) == ".conf")
            return e.path().string();
    }
    return "";
}

static void conf_cam(const std::string& conf, const std::string& section,
                     double K[4], double dist[5]) {
    auto kv = read_conf_section(conf, section);
    auto get = [&](const char* key, double def = 0.0) {
        auto it = kv.find(key);
        return it == kv.end() ? def : it->second;
    };
    K[0] = get("FX");  K[1] = get("FY");  K[2] = get("CX");  K[3] = get("CY");
    dist[0] = get("K1"); dist[1] = get("K2"); dist[2] = get("P1");
    dist[3] = get("P2"); dist[4] = get("K3");
}

int main(int argc, char** argv) {
    std::string format   = "colmap";
    std::string cam_model = "pinhole";
    bool unrectified = true;
    int every_n = 1;
    std::string svo_path, conf_path;
    std::vector<std::string> pos;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        bool has_next = (i+1 < argc);
        if      (a == "--format")       { if (!has_next) { std::cerr << a << " needs a value\n"; return 1; } format    = argv[++i]; }
        else if (a == "--camera-model") { if (!has_next) { std::cerr << a << " needs a value\n"; return 1; } cam_model = argv[++i]; }
        else if (a == "--every-n")      { if (!has_next) { std::cerr << a << " needs a value\n"; return 1; } every_n   = std::atoi(argv[++i]); }
        else if (a == "--svo")          { if (!has_next) { std::cerr << a << " needs a value\n"; return 1; } svo_path  = argv[++i]; }
        else if (a == "--conf")         { if (!has_next) { std::cerr << a << " needs a value\n"; return 1; } conf_path = argv[++i]; }
        else if (a == "--rectified")    unrectified = false;
        else if (a[0] != '-')           pos.push_back(a);
        else { std::cerr << "Unknown argument: " << a << "\n"; return 1; }
    }

    if (pos.size() < 2) {
        std::cerr << "Usage: zed_export <in_dir> <out_dir> "
                     "[--format colmap|euroc] [--camera-model pinhole|opencv] "
                     "[--every-n N]\n";
        return 1;
    }

    const std::string in_dir = pos[0], out_dir = pos[1];
    if (every_n < 1) every_n = 1;
    if (svo_path.empty())  svo_path  = in_dir + "/recording.svo";
    if (conf_path.empty()) conf_path = find_conf(in_dir);

    const bool euroc = (format == "euroc");
    if (!euroc && format != "colmap") {
        std::cerr << "--format must be colmap|euroc\n"; return 1;
    }
    if (cam_model != "pinhole" && cam_model != "opencv") {
        std::cerr << "--camera-model must be pinhole|opencv\n"; return 1;
    }
    if (!fs::exists(svo_path)) {
        std::cerr << "SVO not found: " << svo_path << "\n"; return 1;
    }

    const bool need_conf = euroc || (cam_model == "opencv") || unrectified;
    if (need_conf && conf_path.empty()) {
        std::cerr << "Need factory .conf for distortion. "
                     "Pass --conf /usr/local/zed/settings/SNxxx.conf\n";
        return 1;
    }

    const std::string root = euroc ? (out_dir + "/mav0") : out_dir;
    const std::string c0   = root + "/cam0/data";
    const std::string c1   = root + "/cam1/data";
    std::error_code ec;
    fs::create_directories(c0, ec);
    fs::create_directories(c1, ec);
    fs::create_directories(root + "/imu0", ec);

    std::ofstream csv0(root + "/cam0/data.csv"), csv1(root + "/cam1/data.csv");
    csv0 << "#timestamp [ns],filename\n";
    csv1 << "#timestamp [ns],filename\n";
    std::ofstream list0, list1;
    if (!euroc) {
        list0.open(out_dir + "/cam0_list.txt");
        list1.open(out_dir + "/cam1_list.txt");
    }

    sl::Camera zed;
    sl::InitParameters init;
    init.input.setFromSVOFile(sl::String(svo_path.c_str()));
    init.coordinate_units = sl::UNIT::METER;
    init.depth_mode       = sl::DEPTH_MODE::NONE;
    sl::ERROR_CODE open_err = zed.open(init);
    if (open_err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "Open SVO failed: " << sl::toString(open_err) << "\n";
        return 1;
    }

    sl::CameraInformation info = zed.getCameraInformation();
    const auto& raw = info.camera_configuration.calibration_parameters_raw;
    const int W = raw.left_cam.image_size.width;
    const int H = raw.left_cam.image_size.height;
    const std::string suf = (W >= 2208) ? "2K" : "FHD";

    double KL[4]={}, dL[5]={}, KR[4]={}, dR[5]={};
    if (need_conf) {
        conf_cam(conf_path, "LEFT_CAM_"  + suf, KL, dL);
        conf_cam(conf_path, "RIGHT_CAM_" + suf, KR, dR);
    } else {
        KL[0]=raw.left_cam.fx;  KL[1]=raw.left_cam.fy;
        KL[2]=raw.left_cam.cx;  KL[3]=raw.left_cam.cy;
        KR[0]=raw.right_cam.fx; KR[1]=raw.right_cam.fy;
        KR[2]=raw.right_cam.cx; KR[3]=raw.right_cam.cy;
    }

    // Stereo transform from raw calibration (matches unrectified images).
    sl::Orientation o  = raw.stereo_transform.getOrientation();
    sl::Translation tt = raw.stereo_transform.getTranslation();
    Qd q_st{ o.ox, o.oy, o.oz, o.ow };
    Vd t_st{ tt.tx, tt.ty, tt.tz };
    Qd qi    = quat_conj(q_st);
    Vd rr    = quat_rot(qi, t_st);
    Vd t_inv = { -rr.x, -rr.y, -rr.z };

    // -------------------------------------------------------------------------
    // COLMAP output writers
    // -------------------------------------------------------------------------

    auto params_str = [&](double K[4], double d[5]) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(6);
        s << K[0] << "," << K[1] << "," << K[2] << "," << K[3];
        if (cam_model == "opencv")
            s << "," << d[0] << "," << d[1] << "," << d[2] << "," << d[3];
        return s.str();
    };

    auto write_cameras_txt = [&]() {
        std::ofstream f(out_dir + "/cameras.txt");
        f << std::fixed << std::setprecision(6);
        if (cam_model == "opencv") {
            f << "# COLMAP cameras (UNRECTIFIED, OPENCV radtan from factory .conf)\n"
              << "# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy k1 k2 p1 p2\n"
              << "1 OPENCV " << W << " " << H
              << " " << KL[0] << " " << KL[1] << " " << KL[2] << " " << KL[3]
              << " " << dL[0] << " " << dL[1] << " " << dL[2] << " " << dL[3] << "\n"
              << "2 OPENCV " << W << " " << H
              << " " << KR[0] << " " << KR[1] << " " << KR[2] << " " << KR[3]
              << " " << dR[0] << " " << dR[1] << " " << dR[2] << " " << dR[3] << "\n";
        } else {
            f << "# COLMAP cameras (UNRECTIFIED, PINHOLE; distortion unmodeled)\n"
              << "# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy\n"
              << "1 PINHOLE " << W << " " << H
              << " " << KL[0] << " " << KL[1] << " " << KL[2] << " " << KL[3] << "\n"
              << "2 PINHOLE " << W << " " << H
              << " " << KR[0] << " " << KR[1] << " " << KR[2] << " " << KR[3] << "\n";
        }
    };

    auto write_rig_json = [&]() {
        std::ofstream f(out_dir + "/rig_config.json");
        f << std::fixed << std::setprecision(8);
        f << "[\n  {\n    \"cameras\": [\n"
          << "      { \"image_prefix\": \"cam0/data/\", \"ref_sensor\": true },\n"
          << "      { \"image_prefix\": \"cam1/data/\",\n"
          << "        \"cam_from_rig_rotation\": ["
          << qi.w << ", " << qi.x << ", " << qi.y << ", " << qi.z << "],\n"
          << "        \"cam_from_rig_translation\": ["
          << t_inv.x << ", " << t_inv.y << ", " << t_inv.z << "] }\n"
          << "    ]\n  }\n]\n";
    };

    auto write_reconstruct_sh = [&]() {
        std::ofstream f(out_dir + "/reconstruct.sh");
        f << std::fixed << std::setprecision(6);
        const std::string model_up = (cam_model == "opencv") ? "OPENCV" : "PINHOLE";
        f << "#!/bin/bash\nset -e\n"
          << "DATASET=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
          << "COLMAP=${COLMAP:-colmap}   # set COLMAP=/path/to/colmap if not on PATH\n"
          << "DB=\"$DATASET/database.db\"\n\n"
          << "$COLMAP feature_extractor --database_path \"$DB\" --image_path \"$DATASET\" \\\n"
          << "  --image_list_path \"$DATASET/cam0_list.txt\" \\\n"
          << "  --ImageReader.camera_model " << model_up << " \\\n"
          << "  --ImageReader.camera_params \"" << params_str(KL, dL) << "\" \\\n"
          << "  --ImageReader.single_camera 1\n\n"
          << "$COLMAP feature_extractor --database_path \"$DB\" --image_path \"$DATASET\" \\\n"
          << "  --image_list_path \"$DATASET/cam1_list.txt\" \\\n"
          << "  --ImageReader.camera_model " << model_up << " \\\n"
          << "  --ImageReader.camera_params \"" << params_str(KR, dR) << "\" \\\n"
          << "  --ImageReader.single_camera 1\n\n"
          << "$COLMAP rig_configurator --database_path \"$DB\" \\\n"
          << "  --rig_config_path \"$DATASET/rig_config.json\"\n\n"
          << "$COLMAP sequential_matcher --database_path \"$DB\" \\\n"
          << "  --SequentialMatching.expand_rig_images 1\n\n"
          << "mkdir -p \"$DATASET/sparse\"\n"
          << "$COLMAP mapper --database_path \"$DB\" --image_path \"$DATASET\" \\\n"
          << "  --output_path \"$DATASET/sparse\" \\\n"
          << "  --Mapper.ba_global_backend CASPAR \\\n"
          << "  --Mapper.ba_refine_sensor_from_rig 0\n";
    };

    // -------------------------------------------------------------------------
    // EuRoC output writers
    // -------------------------------------------------------------------------

    auto write_euroc_cam_yaml = [&](const std::string& path, int cam_index,
                                    double K[4], double d[5]) {
        const sl::Transform& Tic = info.sensors_configuration.camera_imu_transform;
        sl::Orientation io = Tic.getOrientation();
        sl::Translation it = Tic.getTranslation();
        Qd qb{ io.ox, io.oy, io.oz, io.ow };
        Vd pb{ it.tx, it.ty, it.tz };
        if (cam_index == 1) {
            // right cam: same rotation as left (small stereo rot ignored),
            // translation shifted by stereo baseline rotated into IMU frame
            Vd off = quat_rot(qb, t_st);
            pb = { pb.x + off.x, pb.y + off.y, pb.z + off.z };
        }
        auto R = quat_to_rotmat(qb);

        std::ofstream y(path);
        y << std::fixed << std::setprecision(9);
        y << "%YAML:1.0\n"
          << "sensor_type: camera\n"
          << "comment: ZED " << (cam_index == 0 ? "left" : "right")
          << " (unrectified). T_BS is camera pose in IMU/body frame.\n"
          << "T_BS:\n  cols: 4\n  rows: 4\n  data: ["
          << R[0] << ", " << R[1] << ", " << R[2] << ", " << pb.x << ", "
          << R[3] << ", " << R[4] << ", " << R[5] << ", " << pb.y << ", "
          << R[6] << ", " << R[7] << ", " << R[8] << ", " << pb.z << ", "
          << "0.0, 0.0, 0.0, 1.0]\n"
          << "rate_hz: " << info.camera_configuration.fps << "\n"
          << "resolution: [" << W << ", " << H << "]\n"
          << "camera_model: pinhole\n"
          << "intrinsics: [" << K[0] << ", " << K[1] << ", " << K[2] << ", " << K[3] << "]\n"
          << "distortion_model: radial-tangential\n"
          << "distortion_coefficients: ["
          << d[0] << ", " << d[1] << ", " << d[2] << ", " << d[3] << "]\n";
    };

    auto write_euroc_imu_yaml = [&]() {
        std::ofstream y(root + "/imu0/sensor.yaml");
        y << "%YAML:1.0\n"
          << "sensor_type: imu\n"
          << "comment: ZED 2i IMU. Body frame = IMU, so T_BS = identity.\n"
          << "T_BS:\n  cols: 4\n  rows: 4\n  data: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]\n"
          << "rate_hz: 400\n"
          << "gyroscope_noise_density: 1.0e-3\n"
          << "gyroscope_random_walk: 1.0e-5\n"
          << "accelerometer_noise_density: 1.0e-2\n"
          << "accelerometer_random_walk: 1.0e-4\n";
    };

    if (euroc) {
        write_euroc_cam_yaml(root + "/cam0/sensor.yaml", 0, KL, dL);
        write_euroc_cam_yaml(root + "/cam1/sensor.yaml", 1, KR, dR);
        write_euroc_imu_yaml();
    } else {
        write_cameras_txt();
        write_rig_json();
        write_reconstruct_sh();
    }

    // -------------------------------------------------------------------------
    // Frame export loop
    // -------------------------------------------------------------------------

    const sl::VIEW vL = unrectified ? sl::VIEW::LEFT_UNRECTIFIED : sl::VIEW::LEFT;
    const sl::VIEW vR = unrectified ? sl::VIEW::RIGHT_UNRECTIFIED : sl::VIEW::RIGHT;
    sl::Mat zedL, zedR;
    sl::RuntimeParameters rt;
    std::vector<int> png{cv::IMWRITE_PNG_COMPRESSION, 0};
    long idx = 0, saved = 0;
    int total = zed.getSVONumberOfFrames();

    std::cout << "Exporting " << format << " (" << cam_model << ", unrectified) from "
              << svo_path << " (" << total << " frames, every " << every_n << ")...\n";

    while (true) {
        sl::ERROR_CODE g = zed.grab(rt);
        if (g == sl::ERROR_CODE::END_OF_SVOFILE_REACHED) break;
        if (g != sl::ERROR_CODE::SUCCESS) continue;
        if ((idx++ % every_n) != 0) continue;

        const uint64_t ts = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE).getNanoseconds();
        zed.retrieveImage(zedL, vL);
        zed.retrieveImage(zedR, vR);
        cv::Mat l, r;
        cv::cvtColor(slMat2cvMat(zedL), l, cv::COLOR_BGRA2BGR);
        cv::cvtColor(slMat2cvMat(zedR), r, cv::COLOR_BGRA2BGR);
        const std::string fn = std::to_string(ts) + ".png";

        auto fut = std::async(std::launch::async, [&]{ cv::imwrite(c0 + "/" + fn, l, png); });
        cv::imwrite(c1 + "/" + fn, r, png);
        fut.get();

        csv0 << ts << "," << fn << "\n";
        csv1 << ts << "," << fn << "\n";
        if (!euroc) {
            list0 << "cam0/data/" << fn << "\n";
            list1 << "cam1/data/" << fn << "\n";
        }
        if (++saved % 20 == 0)
            std::cout << "\r  saved " << saved << std::flush;
    }
    std::cout << "\r  saved " << saved << " stereo pairs\n";
    zed.close();

    const std::string imu_src = in_dir + "/imu0/data.csv";
    if (fs::exists(imu_src))
        fs::copy_file(imu_src, root + "/imu0/data.csv",
                      fs::copy_options::overwrite_existing, ec);
    else
        std::cerr << "Note: " << imu_src << " not found; imu0 not copied.\n";

    std::cout << "Done -> " << fs::absolute(out_dir) << "\n";
    return 0;
}
