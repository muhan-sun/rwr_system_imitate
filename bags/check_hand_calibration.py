"""Overlay the ORCA hand model on camera frames to check the camera -> tag -> hand_root calibration.

Chain (same as experiments/launch/cube_spin_v2.launch.py):
    cam_T_hand = cam_T_tag (camera_extrinsics.npy) @ tag_T_hand (hand_to_tag_* launch args, inverted)

Draws the hand meshes (translucent + outline), the hand_root axes, the fingertips, and the tag square/axes.
If the tag is visible in the frame, the drawn square should sit on it (checks cam_T_tag).
If the drawn hand sits on the real hand, hand_to_tag_* is right.

Run from the bags folder, with the workspace sourced:
    pixi run --manifest-path ~/ros_ws/pixi.toml bash -c "source ~/ros_ws/install/setup.bash && \
        python check_hand_calibration.py bag_still --out overlay_still"
    # tune the hand_to_tag values live with sliders (needs a display):
        python check_hand_calibration.py bag_still --interactive
    # override values / joints:
        python check_hand_calibration.py bag_still --roll 58 --z 0.44 --joint wrist=0.2
"""
import argparse
import os
import re
import struct
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "experiments")
LAUNCH = os.path.join(EXP, "launch", "cube_spin_v2.launch.py")
CALIB = os.path.join(EXP, "rl_inference", "camera_extrinsics.npy")
MOUNT_OFFSET = np.array([-0.01, 0.0, -0.058])  # mount in hand_root, as in the launch file
KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]


# ---------------------------------------------------------------- transforms
def launch_defaults():
    """Read the hand_to_tag_* defaults from the launch file so this stays in sync with it."""
    src = open(LAUNCH).read()
    out = {}
    for k in KEYS:
        m = re.search(rf"'hand_to_tag_{k}',\s*default_value='([^']+)'", src)
        out[k] = float(m.group(1))
    return out


def tag_T_hand(v):
    """Identical to cube_spin_v2.launch.py: hand_T_tag from the args, then inverted."""
    T = np.eye(4)
    T[:3, :3] = R.from_euler("XYZ", [v["roll"], v["pitch"], v["yaw"]], degrees=True).as_matrix()
    T[:3, 3] = np.array([v["x"], v["y"], v["z"]]) + MOUNT_OFFSET
    return np.linalg.inv(T)


def origin_T(el):
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(s) for s in el.get("xyz", "0 0 0").split()]
    rpy = [float(s) for s in el.get("rpy", "0 0 0").split()]
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()  # URDF rpy = fixed-axis XYZ
    T[:3, 3] = xyz
    return T


# ---------------------------------------------------------------- URDF + meshes
def load_stl(path):
    data = open(path, "rb").read()
    if data[:5] == b"solid" and b"facet" in data[:300]:  # ASCII
        v = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)
        return np.array(v, dtype=float)
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3).astype(float)


def resolve(uri):
    if uri.startswith("package://"):
        pkg, rest = uri[len("package://"):].split("/", 1)
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory(pkg), rest)
    return uri.replace("file://", "")


class Hand:
    def __init__(self, urdf_xml, max_tris_per_mesh=None):
        root = ET.fromstring(urdf_xml)
        self.joints = []  # (name, type, parent, child, origin, axis)
        for j in root.findall("joint"):
            ax = j.find("axis")
            axis = np.array([float(s) for s in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
            self.joints.append((j.get("name"), j.get("type"), j.find("parent").get("link"),
                                j.find("child").get("link"), origin_T(j.find("origin")), axis))
        self.meshes = {}  # link -> triangle vertices in link frame, (3N, 3), consecutive triples
        rng = np.random.default_rng(0)
        for link in root.findall("link"):
            vis = link.find("visual")
            if vis is None or vis.find("geometry/mesh") is None:
                continue
            m = vis.find("geometry/mesh")
            tris = load_stl(resolve(m.get("filename"))).reshape(-1, 3, 3)
            tris = tris * np.array([float(s) for s in m.get("scale", "1 1 1").split()])
            if max_tris_per_mesh and len(tris) > max_tris_per_mesh:  # keep the biggest triangles
                area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
                tris = tris[np.argsort(area)[-max_tris_per_mesh:]]
            pts = tris.reshape(-1, 3)
            To = origin_T(vis.find("origin"))
            self.meshes[link.get("name")] = pts @ To[:3, :3].T + To[:3, 3]
        children = {j[3] for j in self.joints}
        self.root = next(j[2] for j in self.joints if j[2] not in children)

    def fk(self, q):
        """Link poses in the root (hand_root) frame. q: joint name -> angle (rad), missing = 0."""
        T = {self.root: np.eye(4)}
        pending = list(self.joints)
        while pending:
            rest = []
            for name, typ, parent, child, To, axis in pending:
                if parent not in T:
                    rest.append((name, typ, parent, child, To, axis))
                    continue
                Tj = np.eye(4)
                if typ in ("revolute", "continuous"):
                    Tj[:3, :3] = R.from_rotvec(axis / np.linalg.norm(axis) * q.get(name, 0.0)).as_matrix()
                elif typ == "prismatic":
                    Tj[:3, 3] = axis * q.get(name, 0.0)
                T[child] = T[parent] @ To @ Tj
            if len(rest) == len(pending):
                break
            pending = rest
        return T


# ---------------------------------------------------------------- bag
def read_bag(bag, camera, n_frames):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    img_topic, intr_topic = f"/{camera}/color", f"/{camera}/intrinsics"
    want = [img_topic, intr_topic, "/right/hand/robot_description", "/right/hand/joint_states"]
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in want if t in types]))
    # first pass: timestamps only, to pick evenly spaced frames without decoding every image
    meta = rosbag2_py.Info().read_metadata(bag, "mcap")
    n_img = next(t.message_count for t in meta.topics_with_message_count if t.topic_metadata.name == img_topic)
    pick = set(np.linspace(0, n_img - 1, n_frames).round().astype(int).tolist())
    frames, intr, urdf, js, i = [], None, None, {}, 0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == img_topic:
            if i in pick:
                m = deserialize_message(data, get_message(types[topic]))
                img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
                if m.encoding == "rgb8":
                    img = img[..., ::-1]
                frames.append((i, img.copy(), dict(js)))
            i += 1
        elif topic == intr_topic and intr is None:
            intr = deserialize_message(data, get_message(types[topic]))
        elif topic == "/right/hand/robot_description" and urdf is None:
            urdf = deserialize_message(data, get_message(types[topic])).data
        elif topic == "/right/hand/joint_states":
            m = deserialize_message(data, get_message(types[topic]))
            js = dict(zip(m.name, m.position))
    return frames, intr, urdf


def detect_tag(img, intr, tag_size, tag_id, family="tag36h11"):
    """cam_T_tag from this frame, same detector settings as apriltag_calibration.py (no distortion)."""
    from dt_apriltags import Detector
    det = Detector(families=family, nthreads=4, quad_decimate=2.0, quad_sigma=0.0, refine_edges=1,
                   decode_sharpening=0.25)
    ds = det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), estimate_tag_pose=True,
                    camera_params=[intr.fx, intr.fy, intr.cx, intr.cy], tag_size=tag_size)
    ds = [d for d in ds if d.tag_id == tag_id]
    if not ds:
        return None
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = ds[0].pose_R, ds[0].pose_t.ravel()
    return T


def current_urdf():
    import xacro
    from ament_index_python.packages import get_package_share_directory
    path = os.path.join(get_package_share_directory("viz"), "models", "orcahand_v2", "orcahand_v2.urdf.xacro")
    return xacro.process_file(path, mappings={"hand_chirality": "right"}).toxml()


# ---------------------------------------------------------------- drawing
def project(P, cam_T, K, dist):
    Pc = P @ cam_T[:3, :3].T + cam_T[:3, 3]
    ok = Pc[:, 2] > 0.01
    uv, _ = cv2.projectPoints(Pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return uv.reshape(-1, 2), ok


def draw_axes(img, cam_T, K, dist, L, label, thick=2):
    uv, ok = project(np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], float), cam_T, K, dist)
    if not ok.all():
        return
    o = tuple(uv[0].astype(int))
    for k, c in zip(range(1, 4), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.line(img, o, tuple(uv[k].astype(int)), c, thick, cv2.LINE_AA)
    cv2.putText(img, label, (o[0] + 5, o[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def render(img, hand, q, cam_T_tag, vals, K, dist, tag_size, show_mesh=True):
    out = img.copy()
    h, w = out.shape[:2]
    cam_T_hand = cam_T_tag @ tag_T_hand(vals)
    links = hand.fk(q)

    if show_mesh:
        # each link is filled from all its triangles; the union gets a light tint, and every link gets
        # its own outline so neighbouring fingers stay distinguishable and the real hand shows through
        union = np.zeros((h, w), np.uint8)
        outlines = []
        for link, pts in hand.meshes.items():
            uv, ok = project(pts, cam_T_hand @ links[link], K, dist)
            tri = uv.reshape(-1, 3, 2)
            keep = ok.reshape(-1, 3).all(1) & (np.abs(tri) < 4 * max(h, w)).all((1, 2))
            mask = np.zeros((h, w), np.uint8)
            for t in np.round(tri[keep] * 4).astype(np.int32):  # 2 bits of sub-pixel precision
                cv2.fillConvexPoly(mask, t, 255, cv2.LINE_8, 2)
            union |= mask
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            outlines += cnts
        sel = union > 0
        out[sel] = (0.7 * out[sel] + 0.3 * np.array([255, 0, 255])).astype(np.uint8)
        cv2.drawContours(out, outlines, -1, (255, 0, 255), 1, cv2.LINE_AA)

    for link in links:
        if link.endswith("fingertip"):
            uv, ok = project(links[link][:3, 3][None], cam_T_hand, K, dist)
            if ok[0]:
                cv2.circle(out, tuple(uv[0].astype(int)), 4, (0, 255, 255), -1, cv2.LINE_AA)
    draw_axes(out, cam_T_hand, K, dist, 0.05, "hand_root")

    s = tag_size / 2
    sq = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]])
    uv, ok = project(sq, cam_T_tag, K, dist)
    if ok.all():
        cv2.polylines(out, [uv.round().astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
    draw_axes(out, cam_T_tag, K, dist, tag_size / 2, "tag")

    txt = "  ".join(f"{k}={vals[k]:.4g}" for k in KEYS)
    cv2.putText(out, txt, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, txt, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- main
def interactive(frames, hand, cam_T_tags, vals, K, dist, tag_size, trange=0.05, rrange=15.0):
    # sliders: translation +-5 cm in 1 mm steps, rotation +-15 deg in 0.5 deg steps, around the start values
    base = dict(vals)
    spec = {k: (trange, 0.001) for k in KEYS[:3]} | {k: (rrange, 0.5) for k in KEYS[3:]}
    win = "hand calibration (q: quit, p: print, n/b: next/prev frame, m: toggle mesh)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for k, (rng, step) in spec.items():
        cv2.createTrackbar(k, win, int(rng / step), int(2 * rng / step), lambda _: None)
    fi, mesh = 0, True
    while True:
        for k, (rng, step) in spec.items():
            vals[k] = base[k] + (cv2.getTrackbarPos(k, win) * step - rng)
        idx, img, q = frames[fi]
        cv2.imshow(win, render(img, hand, q, cam_T_tags[fi], vals, K, dist, tag_size, mesh))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("n"):
            fi = (fi + 1) % len(frames)
        if key == ord("b"):
            fi = (fi - 1) % len(frames)
        if key == ord("m"):
            mesh = not mesh
        if key == ord("p"):
            print_vals(vals)
    print_vals(vals)
    cv2.destroyAllWindows()


def print_vals(vals):
    print("launch args: " + " ".join(f"hand_to_tag_{k}:={vals[k]:.4f}" for k in KEYS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--camera", default="oakd_bottom_view")
    ap.add_argument("--calib", default=CALIB, help="cam_T_tag .npy")
    ap.add_argument("--tag-size", type=float, default=0.08)
    ap.add_argument("--tag-id", type=int, default=25)
    ap.add_argument("--detect-tag", action="store_true",
                    help="detect the tag in each frame instead of using --calib (checks hand_to_tag only, "
                         "works even if the camera moved since calibrating)")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--out", default="calib_overlay")
    ap.add_argument("--no-dist", action="store_true", help="project without lens distortion")
    ap.add_argument("--bag-urdf", action="store_true", help="use the URDF stored in the bag instead of the current xacro")
    ap.add_argument("--joint", action="append", default=[], help="name=rad, overrides joint states (repeatable)")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--trange", type=float, default=0.05, help="slider range +- in m (x/y/z)")
    ap.add_argument("--rrange", type=float, default=15.0, help="slider range +- in deg (roll/pitch/yaw)")
    ap.add_argument("--only", help="comma-separated links to draw, e.g. forearm,tower (the links rigid to the tag)")
    for k in KEYS:
        ap.add_argument(f"--{k}", type=float, help=f"override hand_to_tag_{k}")
    a = ap.parse_args()

    vals = launch_defaults()
    for k in KEYS:
        if getattr(a, k) is not None:
            vals[k] = getattr(a, k)
    cam_T_tag = np.load(a.calib)

    frames, intr, bag_urdf = read_bag(a.bag, a.camera, a.frames)
    assert frames and intr is not None, "no images or intrinsics in bag"
    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]])
    dist = np.zeros(5) if a.no_dist else np.array(intr.disto, float)
    hand = Hand(bag_urdf if (a.bag_urdf and bag_urdf) else current_urdf())
    if a.only:
        keep = set(a.only.split(","))
        hand.meshes = {k: v for k, v in hand.meshes.items() if k in keep}

    overrides = {k: float(v) for k, v in (s.split("=") for s in a.joint)}
    frames = [(i, img, q | overrides) for i, img, q in frames]
    if not any(q for _, _, q in frames):
        print("note: no joint states in bag -> all joints at 0 (use --joint wrist=... if the wrist was bent)")
    print(f"calib file: tag at {cam_T_tag[:3, 3].round(4)} m in camera, {np.linalg.norm(cam_T_tag[:3, 3]):.3f} m away")
    cam_T_tags = []
    for i, img, _ in frames:
        T = detect_tag(img, intr, a.tag_size, a.tag_id) if a.detect_tag else None
        if a.detect_tag:
            if T is None:
                print(f"frame {i}: tag {a.tag_id} not detected, using calib file")
            else:
                dt = T[:3, 3] - cam_T_tag[:3, 3]
                ang = np.degrees(np.linalg.norm(R.from_matrix(T[:3, :3] @ cam_T_tag[:3, :3].T).as_rotvec()))
                print(f"frame {i}: detected tag at {T[:3, 3].round(4)} -> differs from calib file by "
                      f"{np.linalg.norm(dt) * 1000:.1f} mm, {ang:.1f} deg")
        cam_T_tags.append(cam_T_tag if T is None else T)
    print_vals(vals)

    if a.interactive:
        interactive(frames, hand, cam_T_tags, vals, K, dist, a.tag_size, a.trange, a.rrange)
        return
    os.makedirs(a.out, exist_ok=True)
    for (i, img, q), T in zip(frames, cam_T_tags):
        p = os.path.join(a.out, f"frame_{i:05d}.png")
        cv2.imwrite(p, render(img, hand, q, T, vals, K, dist, a.tag_size))
        print("wrote", p)


if __name__ == "__main__":
    main()
