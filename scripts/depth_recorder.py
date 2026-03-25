from __future__ import annotations

import argparse
import ctypes
import multiprocessing.shared_memory as mp_shm
import os
import signal
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker

import cv2
import numpy as np
import yaml


class MpSharedHeader(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_double),
        ("writer_status", ctypes.c_uint32),
        ("writer_termination_signal", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32 * 4),
    ]


SIZE_OF_MP_SHARED_HEADER = ctypes.sizeof(MpSharedHeader)


@dataclass
class RecorderConfig:
    crop_region: tuple[int, int, int, int]
    final_resolution: tuple[int, int]
    clip_range: tuple[float, float]
    normalize_output: bool
    output_range: tuple[float, float] | None
    gaussian_enabled: bool


class DepthVideoWriter:
    def __init__(self, output_dir: str, fps: float):
        self.output_path = os.path.join(output_dir, "depth_recording.mp4")
        self.fps = fps if fps > 0 else 30.0
        self.writer = None
        self.frame_size = None

    def write(self, frame: np.ndarray):
        if self.writer is None:
            height, width = frame.shape[:2]
            self.frame_size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
        if self.writer is not None:
            self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class SharedMemoryDepthReader:
    def __init__(self, shm_name: str, resolution: tuple[int, int]):
        self.resolution = resolution
        self.shared_memory = mp_shm.SharedMemory(name=shm_name)
        try:
            resource_tracker.unregister(self.shared_memory._name, "shared_memory")
        except Exception:
            pass
        self.header = MpSharedHeader.from_buffer(self.shared_memory.buf)
        self.image_buffer = np.ndarray(
            resolution[::-1],
            dtype=np.float32,
            buffer=self.shared_memory.buf,
            offset=SIZE_OF_MP_SHARED_HEADER,
        )

    def read_latest(self, last_timestamp: float) -> tuple[np.ndarray | None, float]:
        if self.header.writer_status != 0:
            return None, last_timestamp
        timestamp = float(self.header.timestamp)
        if timestamp <= last_timestamp:
            return None, last_timestamp
        return np.array(self.image_buffer, copy=True), timestamp

    def close(self):
        self.image_buffer = None
        self.header = None
        self.shared_memory.close()


def build_recorder_config(logdir: str, rs_resolution: tuple[int, int]) -> RecorderConfig:
    params_dir = os.path.join(logdir, "params")
    env_cfg_path = os.path.join(params_dir, "env.yaml")
    agent_cfg_path = os.path.join(params_dir, "agent.yaml")

    cfg = None
    for cfg_path in (env_cfg_path, agent_cfg_path):
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path, encoding="utf-8") as f:
            candidate_cfg = yaml.unsafe_load(f)
        if isinstance(candidate_cfg, dict) and "scene" in candidate_cfg:
            cfg = candidate_cfg
            break

    if cfg is None:
        raise KeyError(f"Could not find scene camera config in {env_cfg_path} or {agent_cfg_path}.")

    sim_resolution_before_crop = (
        cfg["scene"]["camera"]["pattern_cfg"]["width"],
        cfg["scene"]["camera"]["pattern_cfg"]["height"],
    )
    sim_crop_region = cfg["scene"]["camera"]["noise_pipeline"]["crop_and_resize"]["crop_region"]
    real_crop_region = (
        int(sim_crop_region[0] * rs_resolution[1] / sim_resolution_before_crop[1]),
        int(sim_crop_region[1] * rs_resolution[1] / sim_resolution_before_crop[1]),
        int(sim_crop_region[2] * rs_resolution[0] / sim_resolution_before_crop[0]),
        int(sim_crop_region[3] * rs_resolution[0] / sim_resolution_before_crop[0]),
    )
    final_resolution = tuple(cfg["scene"]["camera"]["noise_pipeline"]["crop_and_resize"]["resize_shape"])
    clip_range = tuple(cfg["scene"]["camera"]["noise_pipeline"]["normalize"]["depth_range"])
    output_range = cfg["scene"]["camera"]["noise_pipeline"]["normalize"]["output_range"]
    gaussian_enabled = "gaussian_blur_noise" in cfg["scene"]["camera"]["noise_pipeline"]
    return RecorderConfig(
        crop_region=real_crop_region,
        final_resolution=final_resolution,
        clip_range=clip_range,
        normalize_output=cfg["scene"]["camera"]["noise_pipeline"]["normalize"]["normalize"],
        output_range=tuple(output_range) if output_range is not None else None,
        gaussian_enabled=gaussian_enabled,
    )


def preprocess_depth_image(depth_image: np.ndarray, config: RecorderConfig) -> np.ndarray:
    processed = np.clip(depth_image, config.clip_range[0], config.clip_range[1])
    if config.normalize_output:
        processed = (processed - config.clip_range[0]) / (config.clip_range[1] - config.clip_range[0])
        if config.output_range is not None:
            processed = (
                processed * (config.output_range[1] - config.output_range[0]) + config.output_range[0]
            )
    if config.gaussian_enabled:
        processed = cv2.GaussianBlur(processed, (3, 3), 0.5)

    up, down, left, right = config.crop_region
    row_end = None if down == 0 else -down
    col_end = None if right == 0 else -right
    processed = processed[up:row_end, left:col_end]
    processed = cv2.resize(
        processed,
        (config.final_resolution[1], config.final_resolution[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    processed = cv2.inpaint(processed, (processed < 0.2).astype(np.uint8), 3, cv2.INPAINT_NS)
    return processed


def save_depth_images(output_dir: str, timestamp: float, raw_depth: np.ndarray, processed_depth: np.ndarray):
    raw_depth_dir = os.path.join(output_dir, "raw_depth")
    depth_image_dir = os.path.join(output_dir, "depth_image")
    raw_depth_vis_dir = os.path.join(output_dir, "raw_depth_vis")
    depth_image_vis_dir = os.path.join(output_dir, "depth_image_vis")
    os.makedirs(raw_depth_dir, exist_ok=True)
    os.makedirs(depth_image_dir, exist_ok=True)
    os.makedirs(raw_depth_vis_dir, exist_ok=True)
    os.makedirs(depth_image_vis_dir, exist_ok=True)

    timestamp_str = f"{timestamp:.6f}"
    raw_depth_mm = np.clip(raw_depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    processed_uint16 = np.clip(processed_depth * 255.0 * 2.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    cv2.imwrite(os.path.join(raw_depth_dir, f"raw_{timestamp_str}.png"), raw_depth_mm)
    cv2.imwrite(os.path.join(depth_image_dir, f"proc_{timestamp_str}.png"), processed_uint16)


def create_colorbar(height: int, width: int, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    gradient = np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1)
    gradient = np.repeat(gradient, width, axis=1)
    colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    tick_depths = [min_depth_m, 0.2, 0.5, 1.0, 1.5, max_depth_m]
    tick_depths = [depth for depth in tick_depths if min_depth_m <= depth <= max_depth_m]
    tick_depths = sorted(set(round(depth, 3) for depth in tick_depths))

    for depth in tick_depths:
        ratio = 0.0 if max_depth_m == min_depth_m else (depth - min_depth_m) / (max_depth_m - min_depth_m)
        y = int(round((1.0 - ratio) * (height - 1)))
        cv2.line(colorbar, (0, y), (width // 3, y), (255, 255, 255), 1)
        cv2.putText(
            colorbar,
            f"{depth:.1f}m",
            (width // 3 + 4, min(height - 5, max(12, y + 4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(colorbar, "Depth", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return colorbar


def make_raw_depth_visualization(raw_depth: np.ndarray, clip_range: tuple[float, float]) -> np.ndarray:
    clipped = np.clip(raw_depth, clip_range[0], clip_range[1])
    denom = max(clip_range[1] - clip_range[0], 1e-6)
    normalized = ((clipped - clip_range[0]) / denom * 255.0).astype(np.uint8)
    # 去掉 255 -，让大深度对应 Warm Color (Red)
    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colorbar = create_colorbar(colorized.shape[0], 110, clip_range[0], clip_range[1])
    panel = np.hstack([colorized, colorbar])
    return panel


def make_processed_visualization(processed_depth: np.ndarray, output_range: tuple[float, float] | None) -> np.ndarray:
    if output_range is None:
        min_val, max_val = 0.0, 1.0
    else:
        min_val, max_val = output_range
    denom = max(max_val - min_val, 1e-6)
    normalized = np.clip((processed_depth - min_val) / denom, 0.0, 1.0)
    processed_gray = (normalized * 255.0).astype(np.uint8)
    # 去掉 255 -，让大数值对应 Warm Color (Red)
    return cv2.applyColorMap(processed_gray, cv2.COLORMAP_TURBO)


def build_video_frame(
    timestamp: float,
    raw_depth: np.ndarray,
    processed_depth: np.ndarray,
    config: RecorderConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_vis = make_raw_depth_visualization(raw_depth, config.clip_range)
    proc_vis = make_processed_visualization(processed_depth, config.output_range)
    proc_vis = cv2.resize(proc_vis, (raw_vis.shape[1], raw_vis.shape[0]), interpolation=cv2.INTER_NEAREST)

    raw_panel = raw_vis.copy()
    proc_panel = proc_vis.copy()
    cv2.putText(raw_panel, "Raw depth", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        proc_panel,
        "Processed depth",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    combined = np.hstack([raw_panel, proc_panel])
    cv2.putText(
        combined,
        f"t={timestamp:.3f}s",
        (12, combined.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return combined, raw_vis, proc_vis


def write_metadata(args, output_dir: str):
    metadata_path = os.path.join(output_dir, "recording_info.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(f"logdir={args.logdir}\n")
        f.write(f"shared_memory={args.shm_name}\n")
        f.write(f"resolution={args.width}x{args.height}\n")
        f.write(f"save_fps={args.save_fps}\n")
        f.write("video_file=depth_recording.mp4\n")
        f.write(f"started_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def main():
    parser = argparse.ArgumentParser(description="Save raw and processed depth images from shared memory.")
    parser.add_argument("--shm-name", type=str, required=True, help="Shared memory name created by the control node.")
    parser.add_argument("--logdir", type=str, required=True, help="Tracking model directory used to build depth preprocessing.")
    parser.add_argument("--width", type=int, required=True, help="Raw depth frame width.")
    parser.add_argument("--height", type=int, required=True, help="Raw depth frame height.")
    parser.add_argument("--output-dir", type=str, required=True, help="Per-run directory under ros_depth_images.")
    parser.add_argument(
        "--save-fps",
        type=float,
        default=10.0,
        help="Depth image save frequency. Set <=0 to save every fresh frame.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    write_metadata(args, args.output_dir)

    running = True

    def _stop_recorder(signum, frame):
        del signum, frame
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop_recorder)
    signal.signal(signal.SIGINT, _stop_recorder)

    resolution = (args.width, args.height)
    config = build_recorder_config(args.logdir, resolution)
    reader = SharedMemoryDepthReader(args.shm_name, resolution)
    video_writer = DepthVideoWriter(args.output_dir, args.save_fps)
    last_shared_timestamp = -1.0
    last_save_wall_time = 0.0
    min_save_period = 0.0 if args.save_fps <= 0 else 1.0 / args.save_fps

    try:
        while running:
            depth_image, shared_timestamp = reader.read_latest(last_shared_timestamp)
            if depth_image is None:
                time.sleep(0.002)
                continue

            last_shared_timestamp = shared_timestamp
            now = time.time()
            if min_save_period > 0.0 and (now - last_save_wall_time) < min_save_period:
                continue

            processed_image = preprocess_depth_image(depth_image, config)
            save_depth_images(args.output_dir, shared_timestamp, depth_image, processed_image)
            video_frame, raw_vis, proc_vis = build_video_frame(shared_timestamp, depth_image, processed_image, config)
            timestamp_str = f"{shared_timestamp:.6f}"
            cv2.imwrite(os.path.join(args.output_dir, "raw_depth_vis", f"raw_vis_{timestamp_str}.png"), raw_vis)
            cv2.imwrite(os.path.join(args.output_dir, "depth_image_vis", f"proc_vis_{timestamp_str}.png"), proc_vis)
            video_writer.write(video_frame)
            last_save_wall_time = now
    finally:
        video_writer.close()
        reader.close()


if __name__ == "__main__":
    main()
