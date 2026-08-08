#!/usr/bin/env python3
"""Create sharp Unitree Go2 motion trails with SAM 2 video segmentation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


def decode_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < 8:
        raise RuntimeError(f"Too few decoded frames in {path}")
    return frames


class FrameAligner:
    def __init__(self, reference: np.ndarray) -> None:
        self.reference = reference
        self.height, self.width = reference.shape[:2]
        self.detector = cv2.ORB_create(nfeatures=4000, fastThreshold=8)
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.reference_points, self.reference_descriptors = self.detector.detectAndCompute(
            reference_gray, None
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.reference_descriptors is None:
            return frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None:
            return frame.copy()
        pairs = self.matcher.knnMatch(descriptors, self.reference_descriptors, k=2)
        matches = [a for a, b in pairs if a.distance < 0.72 * b.distance]
        if len(matches) < 20:
            return frame.copy()
        source = np.float32([points[item.queryIdx].pt for item in matches])
        target = np.float32([self.reference_points[item.trainIdx].pt for item in matches])
        transform, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if transform is None or inliers is None or int(inliers.sum()) < 16:
            return frame.copy()
        return cv2.warpPerspective(
            frame,
            transform,
            (self.width, self.height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )


def stabilize_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    aligner = FrameAligner(frames[len(frames) // 2])
    return [aligner(frame) for frame in frames]


def make_background(frames: list[np.ndarray]) -> np.ndarray:
    indices = np.linspace(0, len(frames) - 1, min(61, len(frames)), dtype=int)
    return np.median(np.stack([frames[int(i)] for i in indices]), axis=0).astype(np.uint8)


def largest_component(mask: np.ndarray, minimum_area: int = 300) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    if stats[label, cv2.CC_STAT_AREA] < minimum_area:
        return np.zeros_like(mask)
    result = np.zeros_like(mask)
    result[labels == label] = 255
    return result


def find_prompt(
    frames: list[np.ndarray], background: np.ndarray
) -> tuple[int, np.ndarray, tuple[int, int]]:
    height, width = background.shape[:2]
    candidates = np.linspace(len(frames) * 0.30, len(frames) * 0.70, 25, dtype=int)
    middle = (len(frames) - 1) / 2.0
    best: tuple[float, int, np.ndarray] | None = None

    for index in np.unique(candidates):
        difference = cv2.absdiff(frames[int(index)], background)
        difference = cv2.GaussianBlur(difference, (5, 5), 0)
        motion = np.max(difference, axis=2).astype(np.uint8)
        _, mask = cv2.threshold(motion, 24, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for label in range(1, count):
            x, y, box_width, box_height, area = stats[label]
            if area < 900 or box_width < 45 or box_height < 45:
                continue
            if box_width > width * 0.48 or box_height > height * 0.58:
                continue
            fill = area / max(box_width * box_height, 1)
            if fill < 0.08:
                continue
            component = np.zeros_like(mask)
            component[labels == label] = 255
            center_x = x + box_width / 2.0
            center_y = y + box_height / 2.0
            spatial_distance = np.hypot(
                (center_x - width / 2.0) / width,
                (center_y - height / 2.0) / height,
            )
            time_distance = abs(index - middle) / max(len(frames), 1)
            edge_penalty = 0.38 if (
                x < width * 0.015
                or y < height * 0.015
                or x + box_width > width * 0.985
                or y + box_height > height * 0.985
            ) else 1.0
            score = (
                float(np.sqrt(area))
                * (0.55 + min(fill, 0.75))
                * max(0.45, 1.0 - spatial_distance * 0.65)
                * max(0.55, 1.0 - time_distance * 1.6)
                * edge_penalty
            )
            if best is None or score > best[0]:
                best = (score, int(index), component)

    if best is None:
        raise RuntimeError("Could not find an automatic robot prompt")

    _, frame_index, component = best
    x, y, box_width, box_height = cv2.boundingRect(component)
    pad_x = max(12, int(box_width * 0.10))
    pad_y = max(12, int(box_height * 0.10))
    box = np.array(
        [
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(width - 1, x + box_width + pad_x),
            min(height - 1, y + box_height + pad_y),
        ],
        dtype=np.float32,
    )

    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    _, _, _, point = cv2.minMaxLoc(distance)
    return frame_index, box, point


def write_frame_directory(frames: list[np.ndarray], directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for index, frame in enumerate(frames):
        output = directory / f"{index:05d}.jpg"
        if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 96]):
            raise RuntimeError(f"Could not write {output}")


def clean_sam_mask(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    binary = largest_component(binary, minimum_area=450)
    return binary


def track_masks(
    predictor,
    frame_directory: Path,
    prompt_index: int,
    box: np.ndarray,
    point: tuple[int, int],
) -> dict[int, np.ndarray]:
    state = predictor.init_state(
        video_path=str(frame_directory),
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
    )
    masks: dict[int, np.ndarray] = {}
    points = np.array([[point[0], point[1]]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    _, _, prompt_logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=prompt_index,
        obj_id=1,
        points=points,
        labels=labels,
        box=box,
    )
    masks[prompt_index] = clean_sam_mask(
        (prompt_logits[0, 0] > 0).detach().cpu().numpy()
    )

    for frame_index, _, logits in predictor.propagate_in_video(
        state, start_frame_idx=prompt_index, reverse=False
    ):
        masks[int(frame_index)] = clean_sam_mask(
            (logits[0, 0] > 0).detach().cpu().numpy()
        )
    if prompt_index > 0:
        for frame_index, _, logits in predictor.propagate_in_video(
            state, start_frame_idx=prompt_index, reverse=True
        ):
            masks[int(frame_index)] = clean_sam_mask(
                (logits[0, 0] > 0).detach().cpu().numpy()
            )
    del state
    torch.cuda.empty_cache()
    return masks


def mask_center(mask: np.ndarray) -> tuple[float, float] | None:
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] <= 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def select_pose_indices(masks: dict[int, np.ndarray], poses: int) -> list[int]:
    records: list[tuple[int, float, float, int]] = []
    for index in sorted(masks):
        mask = masks[index]
        center = mask_center(mask)
        area = int(np.count_nonzero(mask))
        x, y, width, height = cv2.boundingRect(mask)
        image_height, image_width = mask.shape[:2]
        touches_edge = (
            x <= 5
            or y <= 5
            or x + width >= image_width - 5
            or y + height >= image_height - 5
        )
        if center is not None and area >= 450 and not touches_edge:
            records.append((index, center[0], center[1], area))
    if len(records) < poses:
        raise RuntimeError(f"Only {len(records)} valid tracked masks")

    median_area = float(np.median([record[3] for record in records]))
    records = [
        record
        for record in records
        if median_area * 0.32 <= record[3] <= median_area * 3.0
    ]
    if len(records) < poses:
        raise RuntimeError("Too few masks after area consistency filtering")

    distances = [0.0]
    for previous, current in zip(records, records[1:]):
        step = float(np.hypot(current[1] - previous[1], current[2] - previous[2]))
        distances.append(distances[-1] + min(step, 90.0))

    total = distances[-1]
    if total < 20:
        positions = np.linspace(0, len(records) - 1, poses, dtype=int)
        return [records[int(position)][0] for position in positions]

    selected: list[int] = []
    for target in np.linspace(0, total, poses):
        position = int(np.argmin(np.abs(np.asarray(distances) - target)))
        index = records[position][0]
        if index not in selected:
            selected.append(index)
    if len(selected) < poses:
        for record in records:
            if record[0] not in selected:
                selected.append(record[0])
            if len(selected) == poses:
                break
    return sorted(selected[:poses])


def composite_trail(
    frames: list[np.ndarray],
    background: np.ndarray,
    masks: dict[int, np.ndarray],
    pose_indices: list[int],
    style: str,
    min_opacity: float,
) -> np.ndarray:
    canvas = background.astype(np.float32)

    for order, index in enumerate(pose_indices):
        mask = masks[index]
        frame = frames[index]

        # Make motion progression legible without softening its endpoints. The
        # sinusoidal envelope is zero at the first/last pose and strongest in
        # the middle, so both endpoints stay fully sharp and opaque.
        opacity = 1.0
        if style == "temporal" and len(pose_indices) > 1:
            progress = order / (len(pose_indices) - 1)
            softness = float(np.sin(np.pi * progress))
            sigma = 1.15 * softness
            opacity = 1.0 - (1.0 - min_opacity) * softness
            if sigma > 0.05:
                frame = cv2.GaussianBlur(
                    frame, (0, 0), sigmaX=sigma, sigmaY=sigma
                )

        # Preserve the native SAM2 silhouette. A tiny feather only antialiases the
        # boundary; no artificial outline or mask expansion is applied.
        alpha = (
            cv2.GaussianBlur(mask, (3, 3), 0.45).astype(np.float32) / 255.0
            * opacity
        )[..., None]
        canvas = frame.astype(np.float32) * alpha + canvas * (1.0 - alpha)

    return np.clip(canvas, 0, 255).astype(np.uint8)


def save_debug(
    frame: np.ndarray,
    prompt_index: int,
    box: np.ndarray,
    point: tuple[int, int],
    destination: Path,
) -> None:
    debug = frame.copy()
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.circle(debug, point, 6, (0, 255, 255), -1)
    cv2.putText(
        debug,
        f"prompt frame {prompt_index}",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), debug, [cv2.IMWRITE_JPEG_QUALITY, 94])


def save_metadata(
    video: Path,
    prompt_index: int,
    box: np.ndarray,
    point: tuple[int, int],
    masks: dict[int, np.ndarray],
    pose_indices: list[int],
    destination: Path,
) -> None:
    selected = []
    for index in pose_indices:
        mask = masks[index]
        x, y, width, height = cv2.boundingRect(mask)
        center = mask_center(mask)
        selected.append(
            {
                "frame": index,
                "area": int(np.count_nonzero(mask)),
                "bbox": [x, y, width, height],
                "center": [round(center[0], 2), round(center[1], 2)] if center else None,
            }
        )
    payload = {
        "video": str(video),
        "prompt_frame": prompt_index,
        "prompt_box_xyxy": box.astype(int).tolist(),
        "prompt_point": list(point),
        "selected": selected,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_video(
    predictor,
    video: Path,
    output_directory: Path,
    work_root: Path,
    poses: int,
    style: str,
    min_opacity: float,
) -> Path:
    print(f"[{video.name}] decoding", flush=True)
    original_frames = decode_video(video)
    print(f"[{video.name}] stabilizing {len(original_frames)} frames", flush=True)
    frames = stabilize_frames(original_frames)
    background = make_background(frames)
    prompt_index, box, point = find_prompt(frames, background)
    print(
        f"[{video.name}] prompt frame={prompt_index} box={box.astype(int).tolist()} point={point}",
        flush=True,
    )

    frame_directory = work_root / video.stem / "frames"
    write_frame_directory(frames, frame_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    save_debug(
        frames[prompt_index],
        prompt_index,
        box,
        point,
        output_directory / "debug" / f"{video.stem}_prompt.jpg",
    )

    print(f"[{video.name}] SAM2 tracking", flush=True)
    masks = track_masks(predictor, frame_directory, prompt_index, box, point)
    pose_indices = select_pose_indices(masks, poses)
    print(f"[{video.name}] selected poses={pose_indices}", flush=True)
    save_metadata(
        video,
        prompt_index,
        box,
        point,
        masks,
        pose_indices,
        output_directory / "debug" / f"{video.stem}_metadata.json",
    )
    result = composite_trail(
        frames, background, masks, pose_indices, style, min_opacity
    )
    destination = output_directory / f"{video.stem}_trail.png"
    if not cv2.imwrite(str(destination), result):
        raise RuntimeError(f"Could not write {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create stabilized SAM2-segmented robot motion trails."
    )
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--poses", type=int, default=8)
    parser.add_argument(
        "--style",
        choices=("sharp", "temporal"),
        default="temporal",
        help=(
            "sharp keeps every pose crisp; temporal softly blurs intermediate "
            "poses while keeping the first and last sharp"
        ),
    )
    parser.add_argument(
        "--min-opacity",
        type=float,
        default=0.68,
        help="minimum opacity of the center poses for temporal style",
    )
    parser.add_argument(
        "--sam2-root",
        type=Path,
        default=Path("/mnt/data/huangbinling/project/sam2"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/mnt/data/huangbinling/project/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        ),
    )
    parser.add_argument(
        "--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/real/trail/sam2"),
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("/tmp/occplanner_sam2_trails")
    )
    args = parser.parse_args()

    if args.poses < 2:
        parser.error("--poses must be at least 2")
    if not 0.0 < args.min_opacity <= 1.0:
        parser.error("--min-opacity must be in the range (0, 1]")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    print("Loading SAM2 video predictor", flush=True)
    predictor = build_sam2_video_predictor(
        args.config,
        str(args.checkpoint),
        device="cuda",
        apply_postprocessing=True,
    )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[Path, str]] = []
    for video in args.videos:
        try:
            destination = process_video(
                predictor,
                video,
                args.output_dir,
                args.work_dir,
                args.poses,
                args.style,
                args.min_opacity,
            )
            print(destination, flush=True)
        except Exception as error:
            failures.append((video, str(error)))
            print(f"[{video.name}] FAILED: {error}", flush=True)
            torch.cuda.empty_cache()

    if failures:
        print("Failed videos:", flush=True)
        for video, message in failures:
            print(f"  {video}: {message}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
