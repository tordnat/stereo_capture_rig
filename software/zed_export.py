#!/usr/bin/env python3
# Reads a directory produced by zed_record:
#   <in>/recording.svo, <in>/imu0/data.csv, <in>/SN<serial>.conf
#
# Both formats export UNRECTIFIED (raw) images.
#
#   --format colmap (default):
#       cam0/data/*.png  cam1/data/*.png  cam{0,1}/data.csv
#       cameras.txt              (PINHOLE by default, or OPENCV via --camera-model)
#       rig_config.json          (cam_from_rig = inverse(stereo_transform), w-first quat)
#       cam0_list.txt cam1_list.txt
#       reconstruct.sh
#       imu0/data.csv            (copied)
#
#   --format euroc:
#       mav0/cam0/data/*.png  mav0/cam1/data/*.png  mav0/cam{0,1}/data.csv
#       mav0/cam{0,1}/sensor.yaml
#       mav0/imu0/data.csv  mav0/imu0/sensor.yaml

import argparse
import configparser
import contextlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl


def find_conf(in_dir: Path) -> Path | None:
    for f in in_dir.iterdir():
        if f.name.startswith("SN") and f.suffix == ".conf":
            return f
    return None


def read_conf_cam(conf_path: Path, section: str):
    cp = configparser.ConfigParser()
    cp.read(conf_path)
    if section not in cp:
        raise ValueError(f"Section [{section}] not found in {conf_path}")
    s = cp[section]
    K    = [float(s["fx"]), float(s["fy"]), float(s["cx"]), float(s["cy"])]
    dist = [float(s.get("k1", 0)), float(s.get("k2", 0)),
            float(s.get("p1", 0)), float(s.get("p2", 0)), float(s.get("k3", 0))]
    return K, dist


def sl_transform_to_matrix(transform) -> np.ndarray:
    """Returns the sl.Transform (sl.Matrix4f) as a (4,4) numpy array."""
    return np.array(transform.m).reshape(4, 4)


def rotmat_to_quat(R: np.ndarray):
    """3x3 rotation matrix → (x, y, z, w) unit quaternion (Shepperd method)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return ((R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s, 0.25/s)
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return (0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s)
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return ((R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s)
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return ((R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s)


def main():
    ap = argparse.ArgumentParser(description="Export ZED SVO to COLMAP or EuRoC dataset")
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--format",       choices=["colmap", "euroc"], default="colmap")
    ap.add_argument("--camera-model", choices=["pinhole", "opencv"], default="pinhole")
    ap.add_argument("--every-n",      type=int, default=1)
    ap.add_argument("--svo",          help="override SVO path")
    ap.add_argument("--conf",         help="override .conf path")
    ap.add_argument("--rectified",    action="store_true")
    ap.add_argument("--jobs",         type=int, default=8,
                    help="parallel PNG writer threads (default 8)")
    args = ap.parse_args()

    in_dir      = Path(args.in_dir)
    out_dir     = Path(args.out_dir)
    every_n     = max(1, args.every_n)
    unrectified = not args.rectified
    euroc       = args.format == "euroc"
    cam_model   = args.camera_model

    svo_path  = Path(args.svo)  if args.svo  else in_dir / "recording.svo"
    conf_path = Path(args.conf) if args.conf else find_conf(in_dir)

    if not svo_path.exists():
        sys.exit(f"SVO not found: {svo_path}")

    need_conf = euroc or cam_model == "opencv" or unrectified
    if need_conf and conf_path is None:
        sys.exit("Need factory .conf for distortion. Pass --conf /path/to/SNxxx.conf")

    root = out_dir / "mav0" if euroc else out_dir
    c0   = root / "cam0" / "data"
    c1   = root / "cam1" / "data"
    c0.mkdir(parents=True, exist_ok=True)
    c1.mkdir(parents=True, exist_ok=True)
    (root / "imu0").mkdir(parents=True, exist_ok=True)

    zed  = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode       = sl.DEPTH_MODE.NONE

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"Open SVO failed: {err}")

    info       = zed.get_camera_information()
    raw_params = info.camera_configuration.calibration_parameters_raw
    W          = raw_params.left_cam.image_size.width
    H          = raw_params.left_cam.image_size.height
    fps        = info.camera_configuration.fps
    suf        = "2K" if W >= 2208 else "FHD"

    if need_conf:
        KL, dL = read_conf_cam(conf_path, f"LEFT_CAM_{suf}")
        KR, dR = read_conf_cam(conf_path, f"RIGHT_CAM_{suf}")
    else:
        L  = raw_params.left_cam
        R  = raw_params.right_cam
        KL = [L.fx, L.fy, L.cx, L.cy];  dL = [0.0] * 5
        KR = [R.fx, R.fy, R.cx, R.cy];  dR = [0.0] * 5

    # Stereo transform from raw calibration (matches unrectified images).
    m_st  = sl_transform_to_matrix(raw_params.stereo_transform)
    R_st  = m_st[:3, :3]
    t_st  = m_st[:3, 3]      # used in euroc cam1 offset
    R_inv = R_st.T            # inverse rotation
    t_inv = -R_inv @ t_st    # inverse translation
    qi    = rotmat_to_quat(R_inv)  # (x,y,z,w) for rig_config.json

    def params_str(K, d):
        s = f"{K[0]:.6f},{K[1]:.6f},{K[2]:.6f},{K[3]:.6f}"
        if cam_model == "opencv":
            s += f",{d[0]:.6f},{d[1]:.6f},{d[2]:.6f},{d[3]:.6f}"
        return s

    def write_cameras_txt():
        with open(out_dir / "cameras.txt", "w") as f:
            if cam_model == "opencv":
                f.write("# COLMAP cameras (UNRECTIFIED, OPENCV radtan from factory .conf)\n")
                f.write("# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy k1 k2 p1 p2\n")
                f.write(f"1 OPENCV {W} {H} {KL[0]:.6f} {KL[1]:.6f} {KL[2]:.6f} {KL[3]:.6f}"
                        f" {dL[0]:.6f} {dL[1]:.6f} {dL[2]:.6f} {dL[3]:.6f}\n")
                f.write(f"2 OPENCV {W} {H} {KR[0]:.6f} {KR[1]:.6f} {KR[2]:.6f} {KR[3]:.6f}"
                        f" {dR[0]:.6f} {dR[1]:.6f} {dR[2]:.6f} {dR[3]:.6f}\n")
            else:
                f.write("# COLMAP cameras (UNRECTIFIED, PINHOLE; distortion unmodeled)\n")
                f.write("# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy\n")
                f.write(f"1 PINHOLE {W} {H} {KL[0]:.6f} {KL[1]:.6f} {KL[2]:.6f} {KL[3]:.6f}\n")
                f.write(f"2 PINHOLE {W} {H} {KR[0]:.6f} {KR[1]:.6f} {KR[2]:.6f} {KR[3]:.6f}\n")

    def write_rig_json():
        data = [{"cameras": [
            {"image_prefix": "cam0/data/", "ref_sensor": True},
            {"image_prefix": "cam1/data/",
             "cam_from_rig_rotation":    [qi[3], qi[0], qi[1], qi[2]],
             "cam_from_rig_translation": list(t_inv)},
        ]}]
        (out_dir / "rig_config.json").write_text(json.dumps(data, indent=2) + "\n")

    def write_reconstruct_sh():
        model_up = "OPENCV" if cam_model == "opencv" else "PINHOLE"
        lines = [
            "#!/bin/bash", "set -e",
            'DATASET="$(cd "$(dirname "$0")" && pwd)"',
            'COLMAP=${COLMAP:-colmap}',
            'DB="$DATASET/database.db"', "",
            f'$COLMAP feature_extractor --database_path "$DB" --image_path "$DATASET" \\',
            f'  --image_list_path "$DATASET/cam0_list.txt" \\',
            f'  --ImageReader.camera_model {model_up} \\',
            f'  --ImageReader.camera_params "{params_str(KL, dL)}" \\',
            f'  --ImageReader.single_camera 1', "",
            f'$COLMAP feature_extractor --database_path "$DB" --image_path "$DATASET" \\',
            f'  --image_list_path "$DATASET/cam1_list.txt" \\',
            f'  --ImageReader.camera_model {model_up} \\',
            f'  --ImageReader.camera_params "{params_str(KR, dR)}" \\',
            f'  --ImageReader.single_camera 1', "",
            f'$COLMAP rig_configurator --database_path "$DB" \\',
            f'  --rig_config_path "$DATASET/rig_config.json"', "",
            f'$COLMAP sequential_matcher --database_path "$DB" \\',
            f'  --SequentialMatching.expand_rig_images 1', "",
            'mkdir -p "$DATASET/sparse"',
            f'$COLMAP mapper --database_path "$DB" --image_path "$DATASET" \\',
            f'  --output_path "$DATASET/sparse" \\',
            f'  --Mapper.ba_global_backend CASPAR \\',
            f'  --Mapper.ba_refine_sensor_from_rig 0',
        ]
        p = out_dir / "reconstruct.sh"
        p.write_text("\n".join(lines) + "\n")
        p.chmod(0o755)

    def write_euroc_cam_yaml(path: Path, cam_index: int, K, d):
        m_tic = sl_transform_to_matrix(info.sensors_configuration.camera_imu_transform)
        R  = m_tic[:3, :3]
        pb = list(m_tic[:3, 3])
        if cam_index == 1:
            off = R @ t_st
            pb  = [pb[0]+off[0], pb[1]+off[1], pb[2]+off[2]]
        row = (f"{R[0,0]:.9f}, {R[0,1]:.9f}, {R[0,2]:.9f}, {pb[0]:.9f}, "
               f"{R[1,0]:.9f}, {R[1,1]:.9f}, {R[1,2]:.9f}, {pb[1]:.9f}, "
               f"{R[2,0]:.9f}, {R[2,1]:.9f}, {R[2,2]:.9f}, {pb[2]:.9f}, "
               f"0.0, 0.0, 0.0, 1.0")
        side = "left" if cam_index == 0 else "right"
        path.write_text(
            f"%YAML:1.0\nsensor_type: camera\n"
            f"comment: ZED {side} (unrectified). T_BS is camera pose in IMU/body frame.\n"
            f"T_BS:\n  cols: 4\n  rows: 4\n  data: [{row}]\n"
            f"rate_hz: {fps}\n"
            f"resolution: [{W}, {H}]\n"
            f"camera_model: pinhole\n"
            f"intrinsics: [{K[0]:.9f}, {K[1]:.9f}, {K[2]:.9f}, {K[3]:.9f}]\n"
            f"distortion_model: radial-tangential\n"
            f"distortion_coefficients: [{d[0]:.9f}, {d[1]:.9f}, {d[2]:.9f}, {d[3]:.9f}]\n"
        )

    def write_euroc_imu_yaml():
        (root / "imu0" / "sensor.yaml").write_text(
            "%YAML:1.0\nsensor_type: imu\n"
            "comment: ZED 2i IMU. Body frame = IMU, so T_BS = identity.\n"
            "T_BS:\n  cols: 4\n  rows: 4\n  data: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]\n"
            "rate_hz: 400\n"
            "gyroscope_noise_density: 1.0e-3\n"
            "gyroscope_random_walk: 1.0e-5\n"
            "accelerometer_noise_density: 1.0e-2\n"
            "accelerometer_random_walk: 1.0e-4\n"
        )

    if euroc:
        write_euroc_cam_yaml(root / "cam0" / "sensor.yaml", 0, KL, dL)
        write_euroc_cam_yaml(root / "cam1" / "sensor.yaml", 1, KR, dR)
        write_euroc_imu_yaml()
    else:
        write_cameras_txt()
        write_rig_json()
        write_reconstruct_sh()

    view_L   = sl.VIEW.LEFT_UNRECTIFIED if unrectified else sl.VIEW.LEFT
    view_R   = sl.VIEW.RIGHT_UNRECTIFIED if unrectified else sl.VIEW.RIGHT
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    total    = zed.get_svo_number_of_frames()

    rect_str = "unrectified" if unrectified else "rectified"
    print(f"Exporting {args.format} ({cam_model}, {rect_str}) from {svo_path}"
          f" ({total} frames, every {every_n}, {args.jobs} writer threads)...")

    zed_L = sl.Mat()
    zed_R = sl.Mat()
    rt    = sl.RuntimeParameters()
    idx   = 0
    saved = 0

    with contextlib.ExitStack() as stack:
        csv0 = stack.enter_context(open(root / "cam0" / "data.csv", "w"))
        csv1 = stack.enter_context(open(root / "cam1" / "data.csv", "w"))
        if not euroc:
            list0 = stack.enter_context(open(out_dir / "cam0_list.txt", "w"))
            list1 = stack.enter_context(open(out_dir / "cam1_list.txt", "w"))
        else:
            list0 = list1 = None
        csv0.write("#timestamp [ns],filename\n")
        csv1.write("#timestamp [ns],filename\n")

        pool = stack.enter_context(ThreadPoolExecutor(max_workers=args.jobs))
        pending = []
        while True:
            err = zed.grab(rt)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                continue
            if idx % every_n != 0:
                idx += 1
                continue
            idx += 1

            ts = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            zed.retrieve_image(zed_L, view_L)
            zed.retrieve_image(zed_R, view_R)
            # .get_data() shares the SDK buffer; cvtColor copies it into a new array
            l  = cv2.cvtColor(zed_L.get_data(), cv2.COLOR_BGRA2BGR)
            r  = cv2.cvtColor(zed_R.get_data(), cv2.COLOR_BGRA2BGR)
            fn = f"{ts}.png"

            pending.append(pool.submit(cv2.imwrite, str(c0 / fn), l, png_params))
            pending.append(pool.submit(cv2.imwrite, str(c1 / fn), r, png_params))

            csv0.write(f"{ts},{fn}\n")
            csv1.write(f"{ts},{fn}\n")
            if not euroc:
                list0.write(f"cam0/data/{fn}\n")
                list1.write(f"cam1/data/{fn}\n")

            saved += 1
            if saved % 20 == 0:
                pending = [f for f in pending if not f.done()]
                print(f"\r  saved {saved}", end="", flush=True)

        for f in pending:
            f.result()

    print(f"\r  saved {saved} stereo pairs")

    zed.close()

    imu_src = in_dir / "imu0" / "data.csv"
    if imu_src.exists():
        shutil.copy2(imu_src, root / "imu0" / "data.csv")
    else:
        print(f"Note: {imu_src} not found; imu0 not copied.", file=sys.stderr)

    print(f"Done -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
