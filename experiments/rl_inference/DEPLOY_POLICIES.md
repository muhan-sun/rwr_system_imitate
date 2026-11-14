# Deploying a policy on the ORCA v2 hand

How to run the shared policies in `policies/` on the real hand, record a rosbag of the run, and plot it.
You don't need Isaac Lab or `rwr_faive_lab` for this. Only exporting a new checkpoint needs them (see the end).

Paths below assume the workspace is at `~/Documents/ETH/RWR/rwr_system`, with this repo at
`~/Documents/ETH/RWR/rwr_system/src/rwr_system`. Change them if yours is elsewhere.

## 1. Get the latest version

```bash
cd ~/Documents/ETH/RWR/rwr_system/src/rwr_system && git pull
```

## 2. Open the ROS environment (in every new terminal)

```bash
conda deactivate          # repeat until no (base) / (isaaclab) prefix is left
cd ~/Documents/ETH/RWR/rwr_system && pixi shell
```

`pixi shell` sources `install/setup.bash` for you. Don't run ROS commands in a conda env: it mixes Pythons and breaks.

## 3. The policies

| File in `policies/` | Trained on | Run with |
|---|---|---|
| `palmaxis_6000.pt` | `orca-v2-inhand-repose-wristlock-palmaxis-v0`, iteration 6000 | `-p wrist_lock_deg:=15.0 -p goal_ang_vel:=2.4` |
| `palmaxis_5500.pt` | same task, iteration 5500 | `-p wrist_lock_deg:=15.0 -p goal_ang_vel:=2.4` |
| `wristlock_2250.pt` | `orca-v2-inhand-repose-wristlock-v0`, iteration 2250 | `-p wrist_lock_deg:=15.0 -p goal_ang_vel:=1.2` |
| `friend_9000.pt` | base or wristlock task, iteration 9000 | `-p goal_ang_vel:=1.2` (add `wrist_lock_deg:=15.0` if trained with the wrist locked) |

- `wrist_lock_deg` holds the wrist at that angle and ignores the policy's wrist output. Training samples 5-25 deg.
- `goal_ang_vel` is the commanded cube speed (rad/s). Use a value inside the training range: 2.4 for the palm-axis
  policies (trained on 1.8-3.0), 1.2 for the others (trained on 0.9-1.5).
- **Float parameters need a decimal point**: `15.0`, not `15`. Otherwise ROS reads an integer and the node crashes
  with `InvalidParameterTypeException`.

## 4. Prepare the hand

Only one program can use the hand's serial port at a time (controller, joint GUI, `chirp.py`, tensioning script).
Close the others first. This must print nothing:

```bash
fuser /dev/ttyACM0
```

If it prints a PID, stop that program (Ctrl+C in its terminal, or `kill <PID>`).

**Calibrate** only after tensioning, or when the tendons were slack (keep your fingers away, the joints move to their end stops):

```bash
ros2 launch hand_control orcahand_v2_controller_launch.yaml hand_chirality:=right calibrate:=true launch_rviz:=true
```

The calibration is saved in `calibration.yaml` next to `config.yaml` (`src/hand_control/orca/configs/orcahand_v2/right/`).
It is not in git, so each PC calibrates once.
If the log says `Motors did not settle ... (still short of goal: [...])`, those motors (IDs, see `joint_to_motor_map` in
`src/hand_control/orca/configs/orcahand_v2/right/config.yaml`) can't reach neutral: check that finger's tendons and
cable. Recalibrating won't fix it. Make sure every finger moves in the joint GUI before testing a policy:

```bash
ros2 launch experiments orcahand_v2_joint_gui.launch.py
```

## 5. Run a policy

Run all commands from `~/Documents/ETH/RWR/rwr_system` inside `pixi shell`. The examples use `palmaxis_6000`.
For another policy, change the file name and its parameters from the table.

**Terminal 1: hand controller**

```bash
ros2 launch hand_control orcahand_v2_controller_launch.yaml hand_chirality:=right launch_rviz:=true
```

**Terminal 2: dry run first** (computes and logs actions, does not move the hand)

```bash
ros2 run experiments policy_inference_cube_spin_v2.py --ros-args -p policy_ckpt_path:=$PWD/src/rwr_system/policies/palmaxis_6000.pt -p wrist_lock_deg:=15.0 -p goal_ang_vel:=2.4 -p dry_run:=true
```

Check that it says `Loaded policy ... (74 observations -> 17 actions)` and that all normalised joint positions in the
log are between -1 and 1. Then Ctrl+C.

**Terminal 3: emergency stop, keep it ready**

```bash
ros2 service call /right/hand/estop std_srvs/srv/SetBool "{data: true}"
```

**Terminal 4: record a rosbag** (start it just before the real run)

```bash
ros2 bag record -o bags/palmaxis_6000_try1 /right/hand/policy_output /right/hand/joint_position_from_motor /right/hand/joint_velocity_from_motor /right/joint_states /right/motor_sensed/joint_states /tf /tf_static
```

`/right/joint_states` is the commanded pose and `/right/motor_sensed/joint_states` the measured pose, both as
`sensor_msgs/JointState` with joint names (published by the hand controller launch for RViz).

Use a new name for each run (`try2`, ...): `ros2 bag record` won't overwrite an existing folder.

**Terminal 2: real run.** The hand moves as soon as the node starts.

```bash
ros2 run experiments policy_inference_cube_spin_v2.py --ros-args -p policy_ckpt_path:=$PWD/src/rwr_system/policies/palmaxis_6000.pt -p wrist_lock_deg:=15.0 -p goal_ang_vel:=2.4 -p dry_run:=false -p csv_log_path:=$PWD/palmaxis_6000_try1.csv
```

The CSV log has the policy's 74 observations and its joint targets at every step (the bag doesn't have the observations).
To stop: Ctrl+C the policy (the hand holds its last pose), then Ctrl+C the bag.

## 6. Look at the runs

Check a bag:

```bash
ros2 bag info bags/palmaxis_6000_try1
```

Plot commanded vs measured joints and compare runs (from `~/Documents/ETH/RWR/rwr_system/src/rwr_system`, in `pixi shell`):

```bash
python experiments/scripts/plot_policy_bags.py bags/shared/palmaxis_5500_try1 bags/shared/palmaxis_6000_try1 --out bags/plots
```

For your own bags in the workspace's `bags/`, pass e.g. `../../bags/palmaxis_6000_try1`. It writes to the `--out` folder:

- `<bag>_joints.png`: commanded and measured position of every joint over time, with the joint limits.
- `comparison.png`: per joint and per bag, the lag, the tracking gain (measured range / commanded range, 1.0 =
  the joint follows the full motion) and the fraction of time commanded at a joint limit.
- `summary.csv`: the numbers behind it. The terminal also prints a table per bag, the gait frequency and joints
  that barely follow their command (gain < 0.3: stuck finger or tendon).

These numbers show how well the hand follows the policy. They don't show if the cube turned: for that, use the video
or the cube tracker.

To browse a bag interactively, open its `.mcap` file in Foxglove (app.foxglove.dev, "Open local file") and plot e.g.
`/right/hand/policy_output.data[10]` against `/right/hand/joint_position_from_motor.data[10]`.

Joint order of both topics (index in `data[...]`, values in rad):
0 wrist, 1 thumb_cmc, 2 thumb_abd, 3 thumb_mcp, 4 thumb_dip, 5 index_abd, 6 index_mcp, 7 index_pip, 8 middle_abd,
9 middle_mcp, 10 middle_pip, 11 ring_abd, 12 ring_mcp, 13 ring_pip, 14 pinky_abd, 15 pinky_mcp, 16 pinky_pip.

`bags/` is git-ignored except `bags/shared/`. Put only small bags you want to share there (GitHub rejects files
over 100 MB).

The shared bags kept their recording names: `palmaxis_2200_try1` is the `wristlock_2250` policy and
`palmaxis_9000_try1` is `friend_9000`, not palm-axis policies.

## Troubleshooting

| Error | Cause and fix |
|---|---|
| `Command 'python' not found` / `ros2: command not found` | Not inside `pixi shell`. |
| `Could not connect to the ORCA hand ... a port held open by another process` | Another controller or GUI still runs: `fuser /dev/ttyACM0`, stop it. |
| `InvalidParameterTypeException ... expecting type 'DOUBLE'` | Write float parameters with a decimal point (`15.0`). |
| Dry run prints values but the hand doesn't move | Expected: dry run never commands the hand. Run with `dry_run:=false`. |
| `Waiting for hand state...` forever | The hand controller (terminal 1) isn't running. |
| Package `experiments` or `hand_control` not found | Rebuild: `colcon build --symlink-install` in the workspace, then reopen `pixi shell`. |

## Exporting a new checkpoint (needs Isaac Lab and rwr_faive_lab)

Training saves checkpoints `model_<iteration>.pt`. The hand needs the exported TorchScript `policy.pt`.
Training logs are git-ignored, so get the checkpoint from whoever trained it (scp from the cluster, Drive, ...).

1. Put each checkpoint in its own folder under `rwr_faive_lab/logs/rsl_rl/orca_v2_repose/<name>/`.
   `play.py` exports into `exported/` next to the checkpoint, so two checkpoints in one folder overwrite each other.
2. Export with the task it was trained on and an explicit checkpoint (without `--checkpoint`, `play.py` loads the
   newest run in your own `logs/`):

   ```bash
   conda activate isaaclab   # or your Isaac Lab env
   cd rwr_faive_lab
   python scripts/rsl_rl/play.py --task=<task> --num_envs 16 --checkpoint logs/rsl_rl/orca_v2_repose/<name>/model_XXXX.pt
   ```

   Check the line `Loading model checkpoint from: ...`. The result is `logs/rsl_rl/orca_v2_repose/<name>/exported/policy.pt`.
3. If you don't know the task: a size mismatch on `critic.0.weight` tells you the critic input size the checkpoint
   expects. 93 = `orca-v2-inhand-repose-wristlock-palmaxis-v0`; 90 = `orca-v2-inhand-repose-v0` or
   `orca-v2-inhand-repose-wristlock-v0` (same network, so the exported policy is the same; the difference is only
   whether you deploy with `wrist_lock_deg`).
4. Copy `exported/policy.pt` to `policies/<name>.pt` in this repo, commit and push. Then anyone can deploy it.
