#!/usr/bin/env python3
"""Create stroboscopic motion-trail stills from fixed-camera videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {index}")
    return frame


def keep_motion_regions(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == label] = 255
    return cleaned


def align_to_reference(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Align a handheld-camera frame to a reference view with a robust homography."""
    detector = cv2.ORB_create(nfeatures=3500, fastThreshold=8)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    reference_points, reference_descriptors = detector.detectAndCompute(reference_gray, None)
    frame_points, frame_descriptors = detector.detectAndCompute(frame_gray, None)

    if reference_descriptors is None or frame_descriptors is None:
        return frame

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(frame_descriptors, reference_descriptors, k=2)
    matches = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    if len(matches) < 18:
        return frame

    source_points = np.float32([frame_points[item.queryIdx].pt for item in matches])
    target_points = np.float32([reference_points[item.trainIdx].pt for item in matches])
    transform, inliers = cv2.findHomography(source_points, target_points, cv2.RANSAC, 3.0)
    if transform is None or inliers is None or int(inliers.sum()) < 14:
        return frame

    height, width = reference.shape[:2]
    return cv2.warpPerspective(
        frame,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def build_cutout_mask(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Extract a crisp moving-robot mask while rejecting textured background noise."""
    difference = cv2.absdiff(frame, background)
    difference = cv2.GaussianBlur(difference, (5, 5), 0)
    motion = np.max(difference, axis=2).astype(np.uint8)
    _, mask = cv2.threshold(motion, 25, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = keep_motion_regions(mask, minimum_area=500)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def make_trail(source: Path, destination: Path, poses: int, stabilize: bool) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {source}")

    # Decode HEVC only once. Repeated random seeks are both slow and unreliable
    # around long GOP boundaries.
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    frame_count = len(frames)
    if frame_count < poses:
        raise RuntimeError(f"{source} contains too few frames")

    reference = frames[frame_count // 2]

    def prepare(frame: np.ndarray) -> np.ndarray:
        return align_to_reference(frame, reference) if stabilize else frame

    # The temporal median removes the moving robot and reconstructs a clean background.
    # Fixed-camera videos remain in their native pixels to preserve maximum sharpness.
    background_indices = np.linspace(0, frame_count - 1, min(61, frame_count), dtype=int)
    background_frames = [
        prepare(frames[int(index)])
        for index in background_indices
    ]
    background = np.median(np.stack(background_frames), axis=0).astype(np.uint8)

    # Avoid endpoints, which often contain cuts or a nearly stationary robot.
    pose_indices = np.linspace(frame_count * 0.08, frame_count * 0.92, poses, dtype=int)
    canvas = background.astype(np.float32)

    for index in pose_indices:
        frame = prepare(frames[int(index)])
        mask = build_cutout_mask(frame, background)

        # Keep robot interiors fully opaque. Only a 1--2 px feather remains at the
        # silhouette boundary for antialiasing, so individual poses stay sharp.
        alpha = (cv2.GaussianBlur(mask, (3, 3), 0.65).astype(np.float32) / 255.0)[..., None]
        canvas = frame.astype(np.float32) * alpha + canvas * (1.0 - alpha)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), np.clip(canvas, 0, 255).astype(np.uint8)):
        raise RuntimeError(f"Could not write {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one stabilized motion-trail PNG for each input video."
    )
    parser.add_argument("videos", nargs="+", type=Path, help="Input video files")
    parser.add_argument("--poses", type=int, default=7, help="Number of overlaid poses")
    parser.add_argument(
        "--stabilize",
        dest="stabilize",
        action="store_true",
        default=True,
        help="Align footage before compositing (enabled by default)",
    )
    parser.add_argument(
        "--no-stabilize",
        dest="stabilize",
        action="store_false",
        help="Skip alignment only for perfectly locked-off footage",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated PNGs (defaults to each video's directory)",
    )
    args = parser.parse_args()

    if args.poses < 2:
        parser.error("--poses must be at least 2")

    for video in args.videos:
        output_name = f"{video.stem}_trail.png"
        output = args.output_dir / output_name if args.output_dir else video.with_name(output_name)
        make_trail(video, output, args.poses, args.stabilize)
        print(output)


if __name__ == "__main__":
    main()
