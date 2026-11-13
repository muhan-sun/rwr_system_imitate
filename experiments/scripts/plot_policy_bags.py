#!/usr/bin/env python3
"""
Plot and compare policy runs recorded with ros2 bag on the ORCA v2 hand.

Reads /right/hand/policy_output (commanded joint targets) and /right/hand/joint_position_from_motor
(measured joint positions), both in rad in robot joint order, and keeps only the time while the policy ran.

Outputs, per bag:
  <name>_joints.png   commanded vs measured position of every joint over time
and for all bags together:
  comparison.png      per-joint lag, tracking gain and time at the joint limits, one bar per bag
  summary.csv         the per-joint numbers behind comparison.png

Example (inside `pixi shell`, from the rwr_system root):
    python src/rwr_system/experiments/scripts/plot_policy_bags.py bags/palmaxis_6000_try1 bags/palmaxis_5500_try1
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from std_msgs.msg import Float32MultiArray

CMD_TOPIC = "/right/hand/policy_output"
POS_TOPIC = "/right/hand/joint_position_from_motor"

# robot joint order (joint_ids in configs/orcahand_v2/right/config.yaml)
JOINTS = ["wrist", "thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip",
          "index_abd", "index_mcp", "index_pip", "middle_abd", "middle_mcp", "middle_pip",
          "ring_abd", "ring_mcp", "ring_pip", "pinky_abd", "pinky_mcp", "pinky_pip"]

# same values as JOINT_LIMITS_USD_DEG in experiments/rl_inference/policy_inference_cube_spin_v2.py
JOINT_LIMITS_DEG = {
    "wrist": (-65.0, 35.0), "index_abd": (-25.0, 30.0), "middle_abd": (-27.0, 27.0),
    "pinky_abd": (-30.0, 30.0), "ring_abd": (-27.0, 27.0), "thumb_cmc": (-45.0, 33.0),
    "index_mcp": (-25.0, 100.0), "middle_mcp": (-25.0, 100.0), "pinky_mcp": (-25.0, 100.0),
    "ring_mcp": (-25.0, 100.0), "thumb_abd": (-18.0, 55.0), "index_pip": (-15.0, 107.0),
    "middle_pip": (-15.0, 107.0), "pinky_pip": (-15.0, 107.0), "ring_pip": (-15.0, 107.0),
    "thumb_mcp": (-25.0, 100.0), "thumb_dip": (-15.0, 107.0),
}

GRID_HZ = 100.0          # common time grid for lag and error
MAX_LAG_S = 1.0          # largest lag searched by cross-correlation
MIN_ACTIVE_DEG = 3.0     # joints whose command moves less than this are not analysed
LIMIT_MARGIN_DEG = 1.0   # a command within this of a joint limit counts as "at the limit"


def read_bag(path):
    storage_id = "mcap" if any(f.endswith(".mcap") for f in os.listdir(path)) else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=storage_id), rosbag2_py.ConverterOptions("", ""))
    data = {CMD_TOPIC: ([], []), POS_TOPIC: ([], [])}
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if topic in data:
            data[topic][0].append(stamp * 1e-9)
            data[topic][1].append(deserialize_message(raw, Float32MultiArray).data)
    out = {}
    for topic, (t, q) in data.items():
        if not t:
            raise RuntimeError(f"{path}: no messages on {topic}")
        out[topic] = (np.array(t), np.rad2deg(np.array(q, dtype=np.float64)))
    return out


def lag_seconds(cmd, meas, dt):
    """Delay of meas behind cmd (s), from the peak of the cross-correlation of the detrended signals."""
    a = cmd - cmd.mean()
    b = meas - meas.mean()
    if b.std() < 1e-6:
        return np.nan
    max_k = int(MAX_LAG_S / dt)
    corr = [np.dot(a[: len(a) - k], b[k:]) / (len(a) - k) for k in range(max_k + 1)]
    return int(np.argmax(corr)) * dt


def dominant_freq(x, dt):
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), dt)
    band = (freqs > 0.2) & (freqs < 5.0)
    return freqs[band][np.argmax(spec[band])]


def analyse(name, bag):
    t_cmd, q_cmd = bag[CMD_TOPIC]
    t_pos, q_pos = bag[POS_TOPIC]
    t0, t1 = t_cmd[0], t_cmd[-1]
    dt = 1.0 / GRID_HZ
    grid = np.arange(t0, t1, dt)
    # commands are held between messages (zero-order hold), measurements are interpolated
    idx = np.clip(np.searchsorted(t_cmd, grid, side="right") - 1, 0, len(t_cmd) - 1)
    cmd = q_cmd[idx]
    meas = np.stack([np.interp(grid, t_pos, q_pos[:, j]) for j in range(len(JOINTS))], axis=1)

    rows = []
    freqs = []
    for j, joint in enumerate(JOINTS):
        c, m = cmd[:, j], meas[:, j]
        cmd_range = np.percentile(c, 95) - np.percentile(c, 5)
        meas_range = np.percentile(m, 95) - np.percentile(m, 5)
        lo, hi = JOINT_LIMITS_DEG[joint]
        at_limit = np.mean((c <= lo + LIMIT_MARGIN_DEG) | (c >= hi - LIMIT_MARGIN_DEG))
        active = cmd_range >= MIN_ACTIVE_DEG
        lag = lag_seconds(c, m, dt) if active else np.nan
        if active:
            freqs.append(dominant_freq(c, dt))
        rows.append({
            "bag": name, "joint": joint,
            "cmd_range_deg": cmd_range, "meas_range_deg": meas_range,
            "gain": meas_range / cmd_range if active else np.nan,
            "lag_ms": lag * 1000 if active else np.nan,
            "rms_err_deg": float(np.sqrt(np.mean((c - m) ** 2))),
            "at_limit_frac": at_limit,
        })
    info = {"duration_s": t1 - t0, "cmd_rate_hz": (len(t_cmd) - 1) / (t1 - t0),
            "gait_hz": float(np.median(freqs)) if freqs else np.nan}
    return grid - t0, cmd, meas, rows, info


def plot_joints(name, t, cmd, meas, rows, out_dir):
    fig, axes = plt.subplots(6, 3, figsize=(16, 18), sharex=True)
    for j, (ax, joint) in enumerate(zip(axes.flat, JOINTS)):
        ax.plot(t, cmd[:, j], label="commanded", lw=1.0)
        ax.plot(t, meas[:, j], label="measured", lw=1.0)
        lo, hi = JOINT_LIMITS_DEG[joint]
        ax.axhline(lo, color="grey", ls=":", lw=0.8)
        ax.axhline(hi, color="grey", ls=":", lw=0.8)
        r = rows[j]
        sub = f"lag {r['lag_ms']:.0f} ms, gain {r['gain']:.2f}" if np.isfinite(r["gain"]) else "inactive"
        ax.set_title(f"{joint}  ({sub})", fontsize=10)
        ax.set_ylabel("deg")
    axes.flat[-1].axis("off")
    axes.flat[0].legend(loc="upper right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time since policy start (s)")
    fig.suptitle(name, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(out_dir, f"{name}_joints.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_comparison(all_rows, names, out_dir):
    metrics = [("lag_ms", "lag measured behind commanded (ms)"),
               ("gain", "tracking gain (measured range / commanded range)"),
               ("at_limit_frac", "fraction of time commanded at a joint limit")]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 12), sharex=True)
    x = np.arange(len(JOINTS))
    width = 0.8 / len(names)
    for ax, (key, label) in zip(axes, metrics):
        for i, name in enumerate(names):
            vals = [r[key] for r in all_rows if r["bag"] == name]
            ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width, label=name)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[1].axhline(1.0, color="black", lw=0.8)
    axes[0].legend(fontsize=9)
    axes[-1].set_xticks(x, JOINTS, rotation=45, ha="right")
    fig.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bags", nargs="+", help="bag folders (as written by ros2 bag record -o)")
    parser.add_argument("--out", default="bags/plots", help="output folder (default: bags/plots)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names, all_rows = [], []
    for path in args.bags:
        name = os.path.basename(os.path.normpath(path))
        t, cmd, meas, rows, info = analyse(name, read_bag(path))
        names.append(name)
        all_rows.extend(rows)
        png = plot_joints(name, t, cmd, meas, rows, args.out)
        active = [r for r in rows if np.isfinite(r["gain"])]
        print(f"\n{name}: {info['duration_s']:.1f} s of policy, commands at {info['cmd_rate_hz']:.1f} Hz, "
              f"gait ~{info['gait_hz']:.2f} Hz -> {png}")
        print(f"  {'joint':<11} {'cmd range':>9} {'meas range':>10} {'gain':>5} {'lag ms':>6} {'rms err':>7} {'at limit':>8}")
        for r in rows:
            print(f"  {r['joint']:<11} {r['cmd_range_deg']:9.1f} {r['meas_range_deg']:10.1f} {r['gain']:5.2f} "
                  f"{r['lag_ms']:6.0f} {r['rms_err_deg']:7.1f} {r['at_limit_frac']:8.0%}")
        if active:
            print(f"  median over active joints: gain {np.median([r['gain'] for r in active]):.2f}, "
                  f"lag {np.nanmedian([r['lag_ms'] for r in active]):.0f} ms, "
                  f"at limit {np.median([r['at_limit_frac'] for r in active]):.0%}")
        stuck = [r["joint"] for r in active if r["gain"] < 0.3]
        if stuck:
            print(f"  barely following its command (gain < 0.3): {', '.join(stuck)}")

    print(f"\n-> {plot_comparison(all_rows, names, args.out)}")
    csv_path = os.path.join(args.out, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"-> {csv_path}")


if __name__ == "__main__":
    main()
