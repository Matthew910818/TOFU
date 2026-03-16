import argparse
import time
import os
import sys
import numpy as np
import socket
import threading
import msvcrt
import cv2

from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_cv import SpatioTemporalContrastAlgorithm
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device

# Standard setup for custom modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dot_tracking_v2 import (
    compute_centered_roi,
    load_calibration_file,
    EvetacDotTracker,
)


DEFAULT_CALIBRATION_FILE = os.environ.get(
    "TACTILE_CALIBRATION_FILE",
    os.path.join(SCRIPT_DIR, "calibration_result.npz"),
)

# Material Mapping
MATERIAL_MAP = {
    "1": ("Plastic", "Hard"),
    "2": ("Plastic", "Soft"),
    "3": ("Metal", "Hard"),
    "4": ("Metal", "Soft"),
    "5": ("Paper", "Hard"),
    "6": ("Paper", "Soft"),
    "7": ("Glass", "Hard"),
    "8": ("Silicone", "Soft"),
}


class RobotSyncController:
    def __init__(self, tm_ip, wsl_ip, wsl_port=65432):
        self.wsl_addr = (wsl_ip, int(wsl_port))
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_cmd(self, cmd, listen_port):
        msg = f"{cmd},{int(listen_port)}"
        self.udp_sock.sendto(msg.encode("utf-8"), self.wsl_addr)
        print(f"[Controller] Sent: {msg}")


def compute_event_grid(ev_roi, rx0, rx1, ry0, ry1, rows=9, cols=7):
    if ev_roi is None or ev_roi.size == 0:
        return np.zeros(rows * cols, dtype=np.float32)

    x_edges = np.linspace(rx0, rx1, cols + 1)
    y_edges = np.linspace(ry0, ry1, rows + 1)
    H, _, _ = np.histogram2d(ev_roi["y"], ev_roi["x"], bins=[y_edges, x_edges])
    return H.flatten().astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", default="tactile_dataset_v6")
    parser.add_argument("--fps", type=float, default=1500.0)
    parser.add_argument("--tm-ip", default="192.168.10.2")
    parser.add_argument("--wsl-ip", default="172.26.77.94")
    parser.add_argument("--listen-port", type=int, default=60000)
    parser.add_argument(
        "--calibration-file",
        default=DEFAULT_CALIBRATION_FILE,
        help=(
            "Path to calibration_result.npz. "
            "Override with --calibration-file or the TACTILE_CALIBRATION_FILE environment variable."
        ),
    )
    parser.add_argument("--track-interval", type=int, default=1)
    parser.add_argument("--contact-thr", type=int, default=50)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def make_empty_buffer():
    return {
        "ev": [],
        "dxdy": [],
        "dxdy_valid": [],
        "ts": [],
        "phase": [],
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ctrl = RobotSyncController(args.tm_ip, args.wsl_ip, 65432)

    gripper_done_event = threading.Event()

    def _udp_listen():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", args.listen_port))
        while True:
            data, _ = sock.recvfrom(1024)
            if b"DONE" in data.upper():
                gripper_done_event.set()

    threading.Thread(target=_udp_listen, daemon=True).start()

    device = initiate_device("")
    i_geometry = device.get_i_geometry()
    width, height = i_geometry.get_width(), i_geometry.get_height()
    dt_step_us = int(1_000_000 / args.fps)

    mv_iterator = EventsIterator.from_device(device, delta_t=dt_step_us)
    stc_filter = SpatioTemporalContrastAlgorithm(width, height, 40000, True)
    events_buf = SpatioTemporalContrastAlgorithm.get_empty_output_buffer()

    if not args.headless:
        frame_gen = OnDemandFrameGenerationAlgorithm(width, height, dt_step_us)
        output_img = np.zeros((height, width, 3), dtype=np.uint8)

    rx0, ry0, rx1, ry1 = compute_centered_roi(width, height, 640, 720)

    tracker = None
    if os.path.exists(args.calibration_file):
        cal = load_calibration_file(args.calibration_file, 63)
        tracker = EvetacDotTracker(cal[0], cal[1], regularize_tracking=True)

    # --- State Machine ---
    stage = "IDLE"
    is_recording = False

    data_buffer = make_empty_buffer()
    current_label = {}

    frame_idx = 0

    last_centers_yx = None
    last_valid_centers_xy = None
    baseline_centers_xy = None
    self_pause_start = None

    print("\n" + "=" * 60)
    print(" 1-8: Texture | w: Re-grip & Lift | s: Slow Release | d: End")
    print("=" * 60)

    try:
        for evs in mv_iterator:
            frame_idx += 1

            stc_filter.process_events(evs, events_buf)
            ev_np = events_buf.numpy()

            mask = (
                (ev_np["x"] >= rx0) & (ev_np["x"] < rx1) &
                (ev_np["y"] >= ry0) & (ev_np["y"] < ry1)
            )
            ev_roi = ev_np[mask]

            # ---------------- Dot Tracking ----------------
            centers_xy = None
            tracker_updated_this_frame = False

            if tracker is not None:
                if frame_idx % args.track_interval == 0:
                    yx = tracker.track(ev_roi["x"], ev_roi["y"])
                    last_centers_yx = yx
                    tracker_updated_this_frame = True
                else:
                    yx = last_centers_yx

                if yx is not None:
                    centers_xy = np.fliplr(yx).astype(np.float32)
                    last_valid_centers_xy = centers_xy.copy()
                else:
                    centers_xy = None

            # ---------------- Visualization ----------------
            if not args.headless:
                frame_gen.process_events(evs)
                frame_gen.generate(evs["t"][-1] if evs.size > 0 else 0, output_img)

                cv2.putText(
                    output_img,
                    f"Stage: {stage}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    output_img,
                    f"Recording: {is_recording}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    output_img,
                    f"ROI ev: {ev_roi.size}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Tactile Collection", output_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ---------------- Keyboard Input ----------------
            if msvcrt.kbhit():
                key = msvcrt.getch().decode().lower()

                if key == "q":
                    break

                # [1] Start Texture Recording
                if stage == "IDLE" and key in MATERIAL_MAP:
                    mat, comp = MATERIAL_MAP[key]
                    current_label = {
                        "mat": mat,
                        "comp": comp,
                        "id": int(time.time()),
                    }

                    data_buffer = make_empty_buffer()
                    baseline_centers_xy = None
                    self_pause_start = None

                    ctrl.send_cmd("SQUEEZE", args.listen_port)
                    ctrl.send_cmd("TM_START_ROBOT", args.listen_port)
                    stage = "WAIT_CONTACT_TEXTURE"

                # Re-grip and Lift Logic
                elif stage == "WAIT_W_KEY" and key == "w":
                    print("-> [Key w] Opening fully before re-gripping...")
                    ctrl.send_cmd("RESET_OPEN", args.listen_port)
                    gripper_done_event.clear()
                    stage = "REGRIPPING_OPEN"

                # [5] Start Slow Release Slip Phase
                elif stage == "WAIT_S_KEY" and key == "s":
                    print("-> [Key s] START SLOW RELEASE. RECORDING ON.")
                    ctrl.send_cmd("START_SLOW_RELEASE", args.listen_port)
                    is_recording = True
                    stage = "COLLECTING_SLIP"

                # [6] Save and Reset
                elif (stage == "COLLECTING_SLIP" or stage == "WAIT_S_KEY") and key == "d":
                    print("-> [Key d] Finished. Saving data...")
                    is_recording = False

                    fname = f"{current_label['id']}_{current_label['mat']}_{current_label['comp']}.npz"
                    np.savez_compressed(
                        os.path.join(args.output_dir, fname),
                        ev=np.asarray(data_buffer["ev"], dtype=np.float32),
                        dxdy=np.asarray(data_buffer["dxdy"], dtype=np.float32),
                        dxdy_valid=np.asarray(data_buffer["dxdy_valid"], dtype=np.uint8),
                        ts=np.asarray(data_buffer["ts"]),
                        phase=np.asarray(data_buffer["phase"]),
                        material=current_label["mat"],
                        compliance=current_label["comp"],
                    )

                    ctrl.send_cmd("STOP_RELEASE", args.listen_port)
                    ctrl.send_cmd("TM_STOP_ROBOT", args.listen_port)
                    ctrl.send_cmd("RESET_OPEN", args.listen_port)
                    stage = "IDLE"

            # ---------------- State Machine Transitions ----------------

            # Step 1 -> 2: Texture Contact
            if stage == "WAIT_CONTACT_TEXTURE" and ev_roi.size > args.contact_thr:
                print("-> Contact detected. Performing STOP before squeeze motion recording...")
                ctrl.send_cmd("STOP", args.listen_port)
                gripper_done_event.clear()
                self_pause_start = time.time()
                stage = "WAIT_STOP_PAUSE"

            if stage == "WAIT_STOP_PAUSE" and gripper_done_event.is_set():
                if self_pause_start is not None and (time.time() - self_pause_start) > 1.0:
                    print("-> [Texture] RECORDING ON. Start recording squeeze motion now.")

                    # 這裡保留你原本的設計：
                    # 先打開 recording，再送 PERFORM_SQUEEZE，
                    # 讓你錄到完整擠壓過程的 event / dxdy 變化
                    is_recording = True

                    # baseline 不再只靠當前 centers_xy，改用最後一次有效 tracking 結果
                    if last_valid_centers_xy is not None:
                        baseline_centers_xy = last_valid_centers_xy.copy()
                    elif centers_xy is not None:
                        baseline_centers_xy = centers_xy.copy()
                    else:
                        baseline_centers_xy = None
                        print("[WARN] baseline_centers_xy is None. dxdy may be invalid for this trial.")

                    ctrl.send_cmd("PERFORM_SQUEEZE", args.listen_port)
                    gripper_done_event.clear()
                    stage = "RECORDING_TEXTURE"

            # Step 2 -> 3: Return to start
            if stage == "RECORDING_TEXTURE" and gripper_done_event.is_set():
                is_recording = False
                ctrl.send_cmd("RETURN_TO_START", args.listen_port)
                gripper_done_event.clear()
                stage = "WAIT_W_KEY"

            # Re-gripping Sequence
            if stage == "REGRIPPING_OPEN" and gripper_done_event.is_set():
                print("-> Gripper opened. Closing now to re-grip...")
                ctrl.send_cmd("SQUEEZE", args.listen_port)
                gripper_done_event.clear()
                stage = "WAIT_CONTACT_LIFT"

            if stage == "WAIT_CONTACT_LIFT" and ev_roi.size > args.contact_thr:
                time.sleep(0.2)
                print("-> Contact detected.")
                ctrl.send_cmd("MINIMAL_SQUEEZE", args.listen_port)
                gripper_done_event.clear()
                stage = "SECURING_FOR_LIFT"

            if stage == "SECURING_FOR_LIFT" and gripper_done_event.is_set():
                print("-> Secured. TM Lifting...")
                ctrl.send_cmd("TM_LIFT_OBJECT", args.listen_port)
                stage = "WAIT_S_KEY"

            # ---------------- Recording ----------------
            if is_recording:
                cell_ev = compute_event_grid(ev_roi, rx0, rx1, ry0, ry1)

                # 若當前 frame 沒有新 centers_xy，就退回最後有效值
                cur_centers_xy = centers_xy if centers_xy is not None else last_valid_centers_xy

                if cur_centers_xy is not None and baseline_centers_xy is not None:
                    dxdy = (cur_centers_xy - baseline_centers_xy).astype(np.float32)
                    dxdy_valid = 1
                else:
                    dxdy = np.zeros((63, 2), dtype=np.float32)
                    dxdy_valid = 0

                ts = evs["t"][-1] if evs.size > 0 else 0

                data_buffer["ev"].append(cell_ev)
                data_buffer["dxdy"].append(dxdy)
                data_buffer["dxdy_valid"].append(dxdy_valid)
                data_buffer["ts"].append(ts)
                data_buffer["phase"].append(stage)

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# import argparse
# import time
# import os
# import sys
# import numpy as np
# import socket
# import threading
# import msvcrt
# import cv2 
# from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
# from metavision_sdk_cv import SpatioTemporalContrastAlgorithm
# from metavision_core.event_io import EventsIterator
# from metavision_core.event_io.raw_reader import initiate_device

# # Standard setup for custom modules
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
# if REPO_ROOT not in sys.path:
#     sys.path.insert(0, REPO_ROOT)

# from dot_tracking_v2 import (
#     compute_centered_roi,
#     load_calibration_file,
#     EvetacDotTracker,
# )

# # Material Mapping
# MATERIAL_MAP = {
#     "1": ("Plastic", "Hard"), "2": ("Plastic", "Soft"),
#     "3": ("Metal",   "Hard"), "4": ("Metal",   "Soft"),
#     "5": ("Paper",   "Hard"), "6": ("Paper",   "Soft"),
#     "7": ("Glass",   "Hard"), "8": ("Silicone", "Soft"),
# }

# class RobotSyncController:
#     def __init__(self, tm_ip, wsl_ip, wsl_port=65432):
#         self.wsl_addr = (wsl_ip, int(wsl_port))
#         self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

#     def send_cmd(self, cmd, listen_port):
#         msg = f"{cmd},{int(listen_port)}"
#         self.udp_sock.sendto(msg.encode("utf-8"), self.wsl_addr)
#         print(f"[Controller] Sent: {msg}")

# def compute_event_grid(ev_roi, rx0, rx1, ry0, ry1, rows=9, cols=7):
#     if ev_roi is None or ev_roi.size == 0: return np.zeros(rows * cols, dtype=np.float32)
#     x_edges = np.linspace(rx0, rx1, cols + 1)
#     y_edges = np.linspace(ry0, ry1, rows + 1)
#     H, _, _ = np.histogram2d(ev_roi["y"], ev_roi["x"], bins=[y_edges, x_edges])
#     return H.flatten().astype(np.float32)

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-o", "--output-dir", default="tactile_dataset_v5")
#     parser.add_argument("--fps", type=float, default=1500.0)
#     parser.add_argument("--tm-ip", default="192.168.10.2")
#     parser.add_argument("--wsl-ip", default="172.26.77.94")
#     parser.add_argument("--listen-port", type=int, default=60000)
#     parser.add_argument("--calibration-file", default=os.path.join(SCRIPT_DIR, "calibration_result.npz"))
#     parser.add_argument("--track-interval", type=int, default=10)
#     parser.add_argument("--contact-thr", type=int, default=30) 
#     parser.add_argument("--headless", action="store_true")
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     os.makedirs(args.output_dir, exist_ok=True)
#     ctrl = RobotSyncController(args.tm_ip, args.wsl_ip, 65432)
    
#     gripper_done_event = threading.Event()
#     def _udp_listen():
#         sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         sock.bind(("0.0.0.0", args.listen_port))
#         while True:
#             data, _ = sock.recvfrom(1024)
#             if b"DONE" in data.upper(): gripper_done_event.set()
#     threading.Thread(target=_udp_listen, daemon=True).start()

#     device = initiate_device("")
#     i_geometry = device.get_i_geometry()
#     width, height = i_geometry.get_width(), i_geometry.get_height()
#     dt_step_us = int(1_000_000 / args.fps)
#     mv_iterator = EventsIterator.from_device(device, delta_t=dt_step_us)
#     stc_filter = SpatioTemporalContrastAlgorithm(width, height, 40000, True)
#     events_buf = SpatioTemporalContrastAlgorithm.get_empty_output_buffer()

#     if not args.headless:
#         frame_gen = OnDemandFrameGenerationAlgorithm(width, height, dt_step_us)
#         output_img = np.zeros((height, width, 3), dtype=np.uint8)

#     rx0, ry0, rx1, ry1 = compute_centered_roi(width, height, 640, 720) 
#     tracker = None
#     if os.path.exists(args.calibration_file):
#         cal = load_calibration_file(args.calibration_file, 63)
#         tracker = EvetacDotTracker(cal[0], cal[1], regularize_tracking=True)

#     # --- Enhanced State Machine ---
#     stage = "IDLE" 
#     is_recording = False
#     data_buffer = {"ev": [], "dxdy": [], "ts": [], "phase": []}
#     current_label = {}
#     frame_idx = 0
#     last_centers_yx, baseline_centers_xy = None, None

#     print("\n" + "="*60)
#     print(" 1-8: Texture | w: Re-grip & Lift | s: Slow Release | d: End")
#     print("="*60)
#     current_max = 0

#     try:
#         for evs in mv_iterator:
#             frame_idx += 1
#             stc_filter.process_events(evs, events_buf)
#             ev_np = events_buf.numpy()
#             mask = (ev_np['x'] >= rx0) & (ev_np['x'] < rx1) & (ev_np['y'] >= ry0) & (ev_np['y'] < ry1)
#             ev_roi = ev_np[mask]
#             # if frame_idx % 100 == 0:
#             # print(f"Frame {frame_idx} | ROI Events: {ev_roi.size}")
            
#             # Dot Tracking logic
#             centers_xy = None
#             if tracker:
#                 if frame_idx % args.track_interval == 0:
#                     yx = tracker.track(ev_roi['x'], ev_roi['y']) 
#                     last_centers_yx = yx
#                 else: 
#                     yx = last_centers_yx
#                 if yx is not None: 
#                     centers_xy = np.fliplr(yx).astype(np.float32)

#             if not args.headless:
#                 frame_gen.process_events(evs)
#                 frame_gen.generate(evs['t'][-1] if evs.size > 0 else 0, output_img)
#                 cv2.putText(output_img, f"Stage: {stage}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#                 cv2.imshow("Tactile Collection", output_img)
#                 if cv2.waitKey(1) & 0xFF == ord('q'): break

#             # Keyboard Input Handling
#             if msvcrt.kbhit():
#                 key = msvcrt.getch().decode().lower()
#                 if key == 'q': break
                
#                 # [1] Start Texture Recording
#                 if stage == "IDLE" and key in MATERIAL_MAP:
#                     mat, comp = MATERIAL_MAP[key]
#                     current_label = {"mat": mat, "comp": comp, "id": int(time.time())}
#                     data_buffer = {k: [] for k in data_buffer}
#                     ctrl.send_cmd("SQUEEZE", args.listen_port)
#                     ctrl.send_cmd("TM_START_ROBOT", args.listen_port)
#                     stage = "WAIT_CONTACT_TEXTURE"
                
#                 # Re-grip and Lift Logic
#                 elif stage == "WAIT_W_KEY" and key == 'w':
#                     print("-> [Key w] Opening fully before re-gripping...")
#                     ctrl.send_cmd("RESET_OPEN", args.listen_port) # Open fully
#                     gripper_done_event.clear()
#                     stage = "REGRIPPING_OPEN"

#                 # [5] Start Slow Release Slip Phase
#                 elif stage == "WAIT_S_KEY" and key == 's':
#                     print("-> [Key s] START SLOW RELEASE. RECORDING ON.")
#                     ctrl.send_cmd("START_SLOW_RELEASE", args.listen_port) 
#                     is_recording = True
#                     stage = "COLLECTING_SLIP"

#                 # [6] Save and Reset
#                 elif (stage == "COLLECTING_SLIP" or stage == "WAIT_S_KEY") and key == 'd':
#                     print("-> [Key d] Finished. Saving data...")
#                     is_recording = False
#                     fname = f"{current_label['id']}_{current_label['mat']}_{current_label['comp']}.npz"
#                     np.savez_compressed(os.path.join(args.output_dir, fname), **data_buffer, material=current_label['mat'], compliance=current_label['comp'])
#                     ctrl.send_cmd("STOP_RELEASE", args.listen_port)
#                     ctrl.send_cmd("TM_STOP_ROBOT", args.listen_port)
#                     ctrl.send_cmd("RESET_OPEN", args.listen_port)
#                     stage = "IDLE"

#             # --- State Machine Transitions ---
            
#             # Step 1 -> 2: Texture Contact
#             if stage == "WAIT_CONTACT_TEXTURE" and ev_roi.size > args.contact_thr:
#                 print("-> Contact detected. Performing +4% squeeze and starting texture recording...")
#                 ctrl.send_cmd("STOP", args.listen_port) 
#                 gripper_done_event.clear()
#                 self_pause_start = time.time()
#                 stage = "WAIT_STOP_PAUSE"

#             if stage == "WAIT_STOP_PAUSE" and gripper_done_event.is_set():
#                 if (time.time() - self_pause_start) > 1.0: 
#                     print("-> [2. Contact] RECORDING ON (Texture Phase). Performing +4% squeeze...")
#                     ctrl.send_cmd("PERFORM_SQUEEZE", args.listen_port)
#                     is_recording = True
#                     baseline_centers_xy = centers_xy.copy() if centers_xy is not None else None
#                     gripper_done_event.clear()
#                     stage = "RECORDING_TEXTURE"

#             # Step 2 -> 3: Return to start
#             if stage == "RECORDING_TEXTURE" and gripper_done_event.is_set():
#                 is_recording = False
#                 ctrl.send_cmd("RETURN_TO_START", args.listen_port)
#                 gripper_done_event.clear()
#                 stage = "WAIT_W_KEY"

#             # Re-gripping Sequence
#             if stage == "REGRIPPING_OPEN" and gripper_done_event.is_set():
#                 print("-> Gripper opened. Closing now to re-grip...")
#                 ctrl.send_cmd("SQUEEZE", args.listen_port) 
#                 gripper_done_event.clear()
#                 stage = "WAIT_CONTACT_LIFT"

#             if stage == "WAIT_CONTACT_LIFT" and ev_roi.size > args.contact_thr:
#                 time.sleep(0.2) 
#                 print("-> Contact detected.")
#                 ctrl.send_cmd("MINIMAL_SQUEEZE", args.listen_port) 
#                 gripper_done_event.clear()
#                 stage = "SECURING_FOR_LIFT"

#             if stage == "SECURING_FOR_LIFT" and gripper_done_event.is_set():
#                 print("-> Secured. TM Lifting...")
#                 ctrl.send_cmd("TM_LIFT_OBJECT", args.listen_port) # RELEASE TM Listen_Lift
#                 stage = "WAIT_S_KEY"

#             if is_recording:
#                 cell_ev = compute_event_grid(ev_roi, rx0, rx1, ry0, ry1)
#                 dxdy = centers_xy - baseline_centers_xy if (centers_xy is not None and baseline_centers_xy is not None) else np.zeros((63, 2), dtype=np.float32)
#                 data_buffer["ev"].append(cell_ev); data_buffer["dxdy"].append(dxdy)
#                 data_buffer["ts"].append(evs['t'][-1] if evs.size > 0 else 0); data_buffer["phase"].append(stage)

#     except Exception as e: print(f"[ERROR] {e}")
#     finally:
#         if not args.headless: cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
