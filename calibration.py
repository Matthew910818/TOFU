import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import numpy as np
import cv2
import torch

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_cv import SpatioTemporalContrastAlgorithm
from metavision_sdk_ui import BaseWindow, EventLoop, UIAction, UIKeyEvent, Window
from dot_tracking_v2 import EvetacDotTracker

# Allow Windows + Conda to find DLLs if needed
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    dll_dir = Path(conda_prefix) / "Library" / "bin"
    if dll_dir.exists():
        os.add_dll_directory(str(dll_dir))

# ROI definition (same as tracking script)
ROI_W_PX = 640
ROI_H_PX = 720

# Reference grid specification (defaults; overridable via CLI)
REF_ROWS_DEFAULT = 9
REF_COLS_DEFAULT = 7
GRID_MARGIN_X_DEFAULT = 0.1   # fraction of ROI width trimmed on each side
GRID_MARGIN_Y_DEFAULT = 0.05  # fraction of ROI height trimmed on each side (smaller -> tighter rows)
ROW_Y_EXTRA_PX = 5.0

MAX_POINTS = 63
TRACKER_NEIGHBOR_RADIUS = 105.0
TRACKER_DOWNSAMPLE = 1


def draw_cross(img, pt, size=6, color=(255, 0, 255), thickness=2):
    """Draw a cross at (x, y) to mark a dot center."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def put_hud(img, text, org=(10, 24), color=(0, 200, 255)):
    """Render a small HUD text on the image."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def compute_centered_roi(width: int, height: int, roi_w: int, roi_h: int):
    cx = width // 2
    cy = height // 2
    rx0 = max(0, cx - roi_w // 2)
    ry0 = max(0, cy - roi_h // 2)
    rx1 = min(width, rx0 + roi_w)
    ry1 = min(height, ry0 + roi_h)
    return rx0, ry0, rx1, ry1


def apply_hard_mask(img, roi):
    rx0, ry0, rx1, ry1 = roi
    img[:ry0, :] = 0
    img[ry1:, :] = 0
    img[:, :rx0] = 0
    img[:, rx1:] = 0


def draw_reference_grid(img, refs, color=(0, 200, 255)):
    for pt in refs:
        draw_cross(img, pt, size=4, color=color, thickness=1)


def clamp_margin(value):
    return max(0.0, min(0.49, float(value)))


def build_reference_grid(rx0, ry0, rx1, ry1, rows, cols, margin_x, margin_y, row_y_extra_px=0.0):
    width = rx1 - rx0
    height = ry1 - ry0
    margin_x = clamp_margin(margin_x)
    margin_y = clamp_margin(margin_y)
    inner_w = width * (1 - 2 * margin_x)
    inner_h = height * (1 - 2 * margin_y)
    offset_x = rx0 + (width - inner_w) * 0.5
    offset_y = ry0 + (height - inner_h) * 0.5
    cell_w = inner_w / cols
    cell_h = inner_h / rows
    points = []
    row_center = (rows - 1) * 0.5
    for r in range(rows):
        for c in range(cols):
            x = offset_x + (c + 0.5) * cell_w
            y = offset_y + (r + 0.5) * cell_h
            y += (r - row_center) * row_y_extra_px
            points.append((float(x), float(y)))
    if row_y_extra_px != 0.0:
        ys = [p[1] for p in points]
        shift = 0.0
        if min(ys) < ry0:
            shift = ry0 - min(ys)
        if max(ys) > ry1:
            shift = min(shift, ry1 - max(ys))
        if shift != 0.0:
            points = [(x, y + shift) for (x, y) in points]
    return points


def export_calibration(path, references, measurements, radii):
    ref_arr = np.asarray(references, dtype=np.float32)
    meas_arr = np.asarray(measurements, dtype=np.float32)
    if meas_arr.shape != ref_arr.shape:
        raise ValueError("Measurement array shape mismatch.")
    offsets = meas_arr - ref_arr
    np.savez(path,
             reference_points=ref_arr,
             measured_points=meas_arr,
             offsets=offsets,
             radii=np.asarray(radii, dtype=np.float32),
             counts=np.ones(len(references), dtype=np.int32))


def detect_blobs(gray_full, roi, detector):
    rx0, ry0, rx1, ry1 = roi
    gray_roi = gray_full[ry0:ry1, rx0:rx1]
    g = cv2.equalizeHist(gray_roi)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    kps = detector.detect(g)
    detections = []
    for k in kps:
        detections.append((rx0 + float(k.pt[0]), ry0 + float(k.pt[1])))
    return detections


def refine_centers_with_blobs(centers_yx, detections_xy, alpha=0.85, match_radius=25.0):
    if centers_yx is None or detections_xy is None or len(detections_xy) == 0:
        return centers_yx
    centers = np.asarray(centers_yx, dtype=np.float32).copy()
    dets = np.asarray(detections_xy, dtype=np.float32)
    mr2 = match_radius * match_radius
    for i, (cy, cx) in enumerate(centers):
        best_idx = -1
        best_d2 = mr2
        for j, (dx, dy) in enumerate(dets):
            d2 = (cx - dx) ** 2 + (cy - dy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_idx = j
        if best_idx >= 0:
            dx, dy = dets[best_idx]
            centers[i, 0] = alpha * dy + (1.0 - alpha) * cy
            centers[i, 1] = alpha * dx + (1.0 - alpha) * cx
    return centers


def create_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 12000
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByInertia = True
    params.minInertiaRatio = 0.4
    params.filterByConvexity = True
    params.minConvexity = 0.7
    params.filterByColor = False
    return cv2.SimpleBlobDetector_create(params)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive calibration using Evetac-style dot tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_input = parser.add_argument_group("Input")
    g_input.add_argument(
        "-i", "--input-event-file",
        dest="event_file_path",
        default="",
        help="Path to RAW/ERF file (empty string = live camera, or provide serial)."
    )
    g_input.add_argument(
        "-r", "--replay_factor",
        type=float,
        default=1.0,
        help="Replay factor for file inputs (>0)."
    )

    g_time = parser.add_argument_group("Timing")
    g_time.add_argument(
        "--dt-step", type=int, default=33333, dest="dt_step",
        help="Event accumulation window (us) used for frame generation."
    )

    g_noise = parser.add_argument_group("Noise Filtering (STC)")
    g_noise.add_argument(
        "--disable-stc", dest="disable_stc", action="store_true",
        help="Disable STC filtering."
    )
    g_noise.add_argument(
        "--stc-filter-thr", dest="stc_filter_thr", type=int, default=40000,
        help="STC window (us)."
    )
    g_noise.add_argument(
        "--disable-stc-cut-trail", dest="stc_cut_trail", default=True, action="store_false",
        help="Disable STC trail cutting."
    )

    g_grid = parser.add_argument_group("Reference grid")
    g_grid.add_argument("--grid-rows", dest="grid_rows", type=int, default=REF_ROWS_DEFAULT,
                        help="Number of marker rows.")
    g_grid.add_argument("--grid-cols", dest="grid_cols", type=int, default=REF_COLS_DEFAULT,
                        help="Number of marker columns.")
    g_grid.add_argument("--grid-margin-x", dest="grid_margin_x", type=float, default=GRID_MARGIN_X_DEFAULT,
                        help="Horizontal margin ratio (0..0.49). Larger = columns closer together.")
    g_grid.add_argument("--grid-margin-y", dest="grid_margin_y", type=float, default=GRID_MARGIN_Y_DEFAULT,
                        help="Vertical margin ratio (0..0.49). Larger = rows closer together.")

    g_out = parser.add_argument_group("Output")
    g_out.add_argument("--output", default="calibration_result.npz", help="Path where calibration data is stored.")

    g_trk = parser.add_argument_group("Tracker")
    g_trk.add_argument("--neighbor-radius", type=float, default=TRACKER_NEIGHBOR_RADIUS,
                       help="Neighbor radius for distance regularization.")
    g_trk.add_argument("--downsample-factor", type=int, default=TRACKER_DOWNSAMPLE,
                       help="Internal downsample factor (affects update threshold).")
    g_trk.add_argument("--disable-regularizer", action="store_true",
                       help="Disable inter-dot distance regularization.")

    return parser.parse_args()


def main():
    args = parse_args()
    args.grid_rows = max(1, int(args.grid_rows))
    args.grid_cols = max(1, int(args.grid_cols))
    args.grid_margin_x = clamp_margin(args.grid_margin_x)
    args.grid_margin_y = clamp_margin(args.grid_margin_y)

    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=args.dt_step)
    if hasattr(mv_iterator.reader, "device") and mv_iterator.reader.device:
        erc = mv_iterator.reader.device.get_i_erc_module()
        if erc:
            erc.set_cd_event_rate(20000000)
            erc.enable(True)

    if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)

    height, width = mv_iterator.get_size()
    roi = compute_centered_roi(width, height, ROI_W_PX, ROI_H_PX)
    rx0, ry0, rx1, ry1 = roi

    stc_filter = SpatioTemporalContrastAlgorithm(width, height, args.stc_filter_thr, args.stc_cut_trail)
    events_buf = SpatioTemporalContrastAlgorithm.get_empty_output_buffer()
    event_frame_gen = OnDemandFrameGenerationAlgorithm(width, height, args.dt_step)
    event_frame_gen.set_color_palette(ColorPalette.Gray)

    reference_points = build_reference_grid(
        rx0, ry0, rx1, ry1,
        args.grid_rows, args.grid_cols,
        args.grid_margin_x, args.grid_margin_y,
        row_y_extra_px=ROW_Y_EXTRA_PX,
    )
    centers_xy_init = np.array(reference_points, dtype=np.float32)
    radii_init = [20.0] * len(reference_points)

    tracker = EvetacDotTracker(
        centers_xy=centers_xy_init,
        radii_list=radii_init,
        regularize_tracking=not args.disable_regularizer,
        neighbor_radius=args.neighbor_radius,
        downsample_factor_internally=args.downsample_factor
    )
    print(f"[INFO] Tracker initialized with {centers_xy_init.shape[0]} reference centers.")

    latest_centers_xy = centers_xy_init.copy()
    blob_detector = create_blob_detector()

    with Window(title="Calibration (Evetac-style tracking)", width=width, height=height,
                mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            nonlocal latest_centers_xy
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()
                return
            if key == UIKeyEvent.KEY_C:
                tracker.reset_to_initial()
                latest_centers_xy = centers_xy_init.copy()
                return
            if key in (UIKeyEvent.KEY_ENTER, getattr(UIKeyEvent, "KEY_KP_ENTER", UIKeyEvent.KEY_ENTER)):
                export_calibration(args.output, reference_points, latest_centers_xy,
                                   tracker.calib_radius.detach().cpu().numpy().flatten())
                print(f"[INFO] Calibration saved to {args.output}")
                return

        window.set_keyboard_callback(keyboard_cb)

        gray_frame = np.zeros((height, width), np.uint8)
        display_bgr = np.zeros((height, width, 3), np.uint8)
        processing_ts = mv_iterator.start_ts

        for evs in mv_iterator:
            if window.should_close():
                break
            processing_ts += mv_iterator.delta_t
            EventLoop.poll_and_dispatch()

            if args.disable_stc:
                events_buf = evs
            else:
                stc_filter.process_events(evs, events_buf)

            event_frame_gen.process_events(events_buf)
            event_frame_gen.generate(processing_ts, gray_frame)
            cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR, dst=display_bgr)
            apply_hard_mask(display_bgr, roi)

            detections_xy = detect_blobs(gray_frame, roi, blob_detector)

            ev_np = events_buf.numpy() if hasattr(events_buf, "numpy") else events_buf
            if ev_np.size > 0:
                ex = ev_np["x"]
                ey = ev_np["y"]
                mask_roi = (ex >= rx0) & (ex < rx1) & (ey >= ry0) & (ey < ry1)
                ex_roi = ex[mask_roi]
                ey_roi = ey[mask_roi]
                centers_yx = tracker.track(ex_roi, ey_roi)
            else:
                centers_yx = tracker.get_centers_yx()

            centers_yx = refine_centers_with_blobs(centers_yx, detections_xy, alpha=0.85, match_radius=25.0)
            tracker.calib_center = torch.from_numpy(centers_yx.copy()).to(tracker.device)

            latest_centers_xy = np.fliplr(centers_yx.copy())

            for (cy, cx) in centers_yx:
                if rx0 <= cx < rx1 and ry0 <= cy < ry1:
                    draw_cross(display_bgr, (cx, cy), size=6, color=(255, 0, 255), thickness=2)

            draw_reference_grid(display_bgr, reference_points, color=(0, 200, 255))

            put_hud(display_bgr, "Enter: save calibration  |  C: reset  |  Q/Esc: quit", org=(10, 24))
            window.show(display_bgr)

        if latest_centers_xy is not None and latest_centers_xy.shape[0] == len(reference_points):
            export_calibration(args.output, reference_points, latest_centers_xy,
                               tracker.calib_radius.detach().cpu().numpy().flatten())
            print(f"[INFO] Calibration saved to {args.output}")


if __name__ == "__main__":
    main()
