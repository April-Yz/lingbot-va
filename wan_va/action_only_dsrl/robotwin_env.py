import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


LOGGER = logging.getLogger("lingbot_action_only_dsrl")


def bootstrap_robowin_root(robowin_root):
    robowin_root = Path(robowin_root).expanduser().resolve()
    if str(robowin_root) not in sys.path:
        sys.path.insert(0, str(robowin_root))
    os.chdir(robowin_root)
    return robowin_root


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_camera_config(robowin_root, camera_type):
    config = _load_yaml(robowin_root / "task_config" / "_camera_config.yml")
    return config[camera_type]


def get_embodiment_config(robot_file):
    return _load_yaml(Path(robot_file) / "config.yml")


def build_task_args(robowin_root, task_name, task_config, save_root, policy_name):
    from envs import CONFIGS_PATH

    args = _load_yaml(robowin_root / "task_config" / f"{task_config}.yml")
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = 0
    args["save_root"] = str(save_root)
    args["policy_name"] = policy_name
    args["eval_mode"] = True
    args["eval_video_log"] = False
    args["render_freq"] = 0

    embodiment_type = args.get("embodiment")
    embodiment_types = _load_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")
    camera_config = _load_yaml(Path(CONFIGS_PATH) / "_camera_config.yml")

    def get_embodiment_file(embodiment_name):
        return embodiment_types[embodiment_name]["file_path"]

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def configure_episode_video_logging(args, episode_video_dir=None, enable=False):
    args["eval_video_log"] = bool(enable)
    if enable and episode_video_dir is not None:
        episode_video_dir = Path(episode_video_dir)
        episode_video_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = str(episode_video_dir)
    else:
        args.pop("eval_video_save_dir", None)
    return args


def _fallback_prompt(task_name):
    return task_name.replace("_", " ")


def _maybe_generate_instruction(task_name, episode_info, instruction_type):
    try:
        from description.utils.generate_episode_instructions import (
            generate_episode_descriptions,
        )

        results = generate_episode_descriptions(task_name, [episode_info], 1)
        return np.random.choice(results[0][instruction_type])
    except Exception as exc:
        LOGGER.warning("Instruction generation fallback: %s", exc)
        return _fallback_prompt(task_name)


def format_obs(observation, prompt):
    return {
        "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
        "observation.state": observation["joint_action"]["vector"],
        "task": prompt,
    }


def add_eef_pose(new_pose, init_pose):
    new_pose_r = R.from_quat(new_pose[3:7][None])
    init_pose_r = R.from_quat(init_pose[3:7][None])
    out_rot = (init_pose_r * new_pose_r).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])


def add_init_pose(new_pose, init_pose):
    left_pose = add_eef_pose(new_pose[:8], init_pose[:8])
    right_pose = add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left_pose, right_pose])


def _initial_eef_pose(observation):
    return np.array(
        observation["endpose"]["left_endpose"]
        + [observation["endpose"]["left_gripper"]]
        + observation["endpose"]["right_endpose"]
        + [observation["endpose"]["right_gripper"]],
        dtype=np.float64,
    )


def find_valid_seed(task_env, args, start_seed, max_tries=64, require_expert_check=False):
    from envs.utils.create_actor import UnStableError

    seed = start_seed
    for episode_idx in range(max_tries):
        try:
            task_env.setup_demo(now_ep_num=episode_idx, seed=seed, is_test=True, **args)
            if require_expert_check:
                episode_info = task_env.play_once()
                plan_success = task_env.plan_success and task_env.check_success()
            else:
                episode_info = {"info": {}}
                plan_success = True
            task_env.close_env()
            if plan_success:
                return seed, episode_info
        except UnStableError:
            task_env.close_env()
        except Exception:
            task_env.close_env()
        seed += 1
    raise RuntimeError(f"Unable to find a valid seed for task {args['task_name']}")


def prepare_episode(task_env, args, episode_idx, seed, instruction_type="seen"):
    task_env.setup_demo(now_ep_num=episode_idx, seed=seed, is_test=True, **args)
    episode_info = {"info": {}}
    prompt = _maybe_generate_instruction(
        args["task_name"],
        episode_info["info"],
        instruction_type,
    )
    task_env.set_instruction(instruction=prompt)
    initial_obs = task_env.get_obs()
    return {
        "prompt": prompt,
        "initial_obs": initial_obs,
        "formatted_obs": format_obs(initial_obs, prompt),
        "init_eef_pose": _initial_eef_pose(initial_obs),
    }


def start_episode_video(task_env, args, episode_idx, fps=10):
    video_dir = args.get("eval_video_save_dir")
    if not video_dir:
        return None

    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"episode{episode_idx}.mp4"
    video_size = f"{args['head_camera_w']}x{args['head_camera_h']}"
    task_env.test_num = episode_idx
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
    )
    task_env._set_eval_video_ffmpeg(ffmpeg)
    return video_path


def finish_episode_video(task_env, episode_idx, success):
    video_dir = getattr(task_env, "eval_video_path", None)
    if video_dir is None:
        return None
    task_env._del_eval_video_ffmpeg()
    original_path = Path(video_dir) / f"episode{episode_idx}.mp4"
    if not original_path.exists():
        return None
    outcome = "succ" if success else "fail"
    renamed_path = Path(video_dir) / f"episode{episode_idx}-{outcome}.mp4"
    if renamed_path.exists():
        renamed_path.unlink()
    original_path.rename(renamed_path)
    return renamed_path


def execute_action_chunk(
    task_env,
    action_chunk,
    prompt,
    init_eef_pose,
    first_chunk=True,
):
    assert action_chunk.shape[2] % 4 == 0
    action_per_frame = action_chunk.shape[2] // 4
    start_idx = 1 if first_chunk else 0
    key_frame_list = []
    dense_reward = 0.0
    success = False

    for frame_idx in range(start_idx, action_chunk.shape[1]):
        for step_idx in range(action_chunk.shape[2]):
            ee_action = action_chunk[:, frame_idx, step_idx]
            if action_chunk.shape[0] == 14:
                from evaluation.robotwin.geometry import euler2quat

                ee_action = np.concatenate(
                    [
                        ee_action[:3],
                        euler2quat(ee_action[3], ee_action[4], ee_action[5]),
                        ee_action[6:10],
                        euler2quat(ee_action[10], ee_action[11], ee_action[12]),
                        ee_action[13:14],
                    ]
                )
            elif action_chunk.shape[0] == 16:
                ee_action = add_init_pose(ee_action, init_eef_pose)
                ee_action = np.concatenate(
                    [
                        ee_action[:3],
                        ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
                        ee_action[7:11],
                        ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
                        ee_action[15:16],
                    ]
                )
            else:
                raise NotImplementedError(f"Unsupported action shape {action_chunk.shape}")

            task_env.take_action(ee_action, action_type="ee")
            if hasattr(task_env, "stage_reward"):
                dense_reward = float(task_env.stage_reward())
            if task_env.eval_success or task_env.check_success():
                success = True
                break

            if (step_idx + 1) % action_per_frame == 0:
                key_frame_list.append(format_obs(task_env.get_obs(), prompt))
        if success:
            break

    if not key_frame_list:
        key_frame_list.append(format_obs(task_env.get_obs(), prompt))

    done = success or task_env.take_action_cnt >= task_env.step_lim
    reward = 1.0 if success else dense_reward
    return {
        "reward": reward,
        "success": success,
        "done": done,
        "key_frame_list": key_frame_list,
        "next_formatted_obs": key_frame_list[-1],
    }
