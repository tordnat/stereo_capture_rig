
// Produces a self-contained dataset directory:
//   <out>/
//     recording.svo          # compressed stereo video (H.265 by default)
//     imu0/data.csv          # live ~400 Hz IMU, camera clock, EuRoC units
//     SN<serial>.conf        # factory calibration, copied for offline export
//
// Extract images with zed_export
//
// Usage:
//   zed_record [out_dir] [options]
//     --out DIR            output directory     (default "zed_rec")
//     --mode 2k15|1080p30  resolution/fps       (default 2k15)
//     --duration MINUTES   run length, minutes  (default 0 = until Ctrl+C)
//     --compression h265|h264|lossless          (default h265)


#include "zed_common.hpp"

#include <chrono>
#include <csignal>
#include <experimental/filesystem>
#include <string>

namespace fs = std::experimental::filesystem;

static volatile sig_atomic_t g_interrupted = 0;
static void on_signal(int) { g_interrupted = 1; }

int main(int argc, char** argv) {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGHUP, on_signal);

    std::string out_dir = "zed_rec";
    std::string mode    = "2k15";
    std::string comp    = "h265";
    double duration_min = 0.0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* n) -> std::string {
            if (i + 1 >= argc) { std::cerr << n << " needs a value\n"; std::exit(1); }
            return argv[++i];
        };
        if      (a == "--out")         out_dir      = next("--out");
        else if (a == "--mode")        mode         = next("--mode");
        else if (a == "--duration")    duration_min = std::atof(next("--duration").c_str());
        else if (a == "--compression") comp         = next("--compression");
        else if (a[0] != '-')          out_dir      = a;
        else { std::cerr << "Unknown argument: " << a << "\n"; return 1; }
    }

    sl::RESOLUTION res; int fps;
    if      (mode == "2k15")    { res = sl::RESOLUTION::HD2K;   fps = 15; }
    else if (mode == "1080p30") { res = sl::RESOLUTION::HD1080; fps = 30; }
    else { std::cerr << "--mode must be '2k15' or '1080p30'\n"; return 1; }

    sl::SVO_COMPRESSION_MODE cmode;
    if      (comp == "h265")     cmode = sl::SVO_COMPRESSION_MODE::H265;
    else if (comp == "h264")     cmode = sl::SVO_COMPRESSION_MODE::H264;
    else if (comp == "lossless") cmode = sl::SVO_COMPRESSION_MODE::LOSSLESS;
    else { std::cerr << "--compression must be h265|h264|lossless\n"; return 1; }

    std::error_code ec;
    fs::create_directories(out_dir + "/imu0", ec);
    if (ec) { std::cerr << "Cannot create " << out_dir << ": " << ec.message() << "\n"; return 1; }

    sl::Camera zed;
    sl::InitParameters init;
    init.camera_resolution        = res;
    init.camera_fps               = fps;
    init.depth_mode               = sl::DEPTH_MODE::NONE;
    init.coordinate_units         = sl::UNIT::METER;
    init.enable_image_enhancement = true;
    init.sdk_verbose              = 1;

    sl::ERROR_CODE err = zed.open(init);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "ZED open failed: " << sl::toString(err) << std::endl;
        return 1;
    }

    sl::CameraInformation info = zed.getCameraInformation();
    const unsigned int serial = info.serial_number;

    {
        const std::string conf = "/usr/local/zed/settings/SN"
                                 + std::to_string(serial) + ".conf";
        if (fs::exists(conf)) {
            fs::copy_file(conf, out_dir + "/SN" + std::to_string(serial) + ".conf",
                          fs::copy_options::overwrite_existing, ec);
            if (ec) std::cerr << "Warning: could not copy " << conf
                              << ": " << ec.message() << "\n";
            else std::cout << "Copied calibration SN" << serial << ".conf\n";
        } else {
            std::cerr << "Warning: " << conf << " not found; unrectified export "
                         "will need --conf pointed at it manually.\n";
        }
    }

    sl::RecordingParameters rec;
    rec.video_filename  = sl::String((out_dir + "/recording.svo").c_str());
    rec.compression_mode = cmode;
    err = zed.enableRecording(rec);
    if (err != sl::ERROR_CODE::SUCCESS) {
        std::cerr << "enableRecording failed: " << sl::toString(err) << std::endl;
        if (cmode == sl::SVO_COMPRESSION_MODE::H265)
            std::cerr << "Try --compression h264.\n";
        zed.close();
        return 1;
    }

    // LED steady on, recording in progress.
    Led led("pwr");
    led.on();

    // Live full-rate IMU log.
    ImuLogger imu(zed, out_dir + "/imu0/data.csv");
    if (!imu.ok())
        std::cerr << "Warning: could not open imu0/data.csv; IMU not logged.\n";

    typedef std::chrono::steady_clock clk;
    const clk::time_point t_start = clk::now();
    const bool limited = (duration_min > 0.0);
    const auto t_total = std::chrono::duration_cast<clk::duration>(
                             std::chrono::duration<double>(duration_min * 60.0));

    std::cout << "Recording " << mode << " (" << fps << " fps), " << comp
              << " to " << out_dir << "/recording.svo\n"
              << (limited ? "Duration: " + std::to_string(duration_min) + " min"
                          : "Until Ctrl+C")
              << ". Recording...\n";

    sl::RuntimeParameters rt;
    uint64_t frames = 0;
    while (!g_interrupted) {
        if (limited && clk::now() - t_start >= t_total) {
            std::cout << "Reached duration limit.\n";
            break;
        }
        // grab() writes the frame to the SVO automatically in SDK 3.
        if (zed.grab(rt) != sl::ERROR_CODE::SUCCESS) continue;
        ++frames;
        if (frames % fps == 0) {  // ~1 Hz progress
            sl::RecordingStatus rs = zed.getRecordingStatus();
            std::cout << "\r  frames: " << frames
                      << "  compression ratio: " << rs.average_compression_ratio
                      << std::flush;
        }
    }
    std::cout << "\n";

    imu.stop(); // before close(): IMU thread uses the handle
    zed.disableRecording();
    led.off();
    zed.close();

    std::cout << "Done. " << frames << " frames recorded, "
              << imu.count() << " IMU samples logged.\n"
              << "Dataset: " << fs::absolute(out_dir) << "\n";
    return 0;
}
