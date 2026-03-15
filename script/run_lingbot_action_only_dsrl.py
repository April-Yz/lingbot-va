import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.action_only_dsrl import ActionOnlyDsrlTrainer
from wan_va.action_only_dsrl.robotwin_env import (
    bootstrap_robowin_root,
    build_task_args,
    class_decorator,
    execute_action_chunk,
    find_valid_seed,
    prepare_episode,
)


LOGGER = logging.getLogger("lingbot_action_only_dsrl")


def setup_logging(save_root):
    save_root.mkdir(parents=True, exist_ok=True)
    log_path = save_root / "action_only_dsrl.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    return log_path


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def log_dict(title, payload):
    LOGGER.info("%s: %s", title, json.dumps(payload, sort_keys=True, default=str))


def validate_inference_paths(trainer, episode):
    trainer.policy.reset(episode["prompt"])
    baseline = trainer.policy.act(
        episode["formatted_obs"],
        deterministic=True,
        use_dsrl=False,
    )
    trainer.policy.reset(episode["prompt"])
    steered = trainer.policy.act(
        episode["formatted_obs"],
        deterministic=True,
        use_dsrl=True,
    )
    log_dict("validation/use_dsrl_false", baseline["diagnostics"])
    log_dict("validation/use_dsrl_true", steered["diagnostics"])
    return baseline, steered


def build_mock_episode(trainer):
    example_root = REPO_ROOT / "example" / "robotwin"
    formatted_obs = {
        "observation.images.cam_high": np.array(
            Image.open(example_root / "observation.images.cam_high.png").convert("RGB")
        ),
        "observation.images.cam_left_wrist": np.array(
            Image.open(example_root / "observation.images.cam_left_wrist.png").convert("RGB")
        ),
        "observation.images.cam_right_wrist": np.array(
            Image.open(example_root / "observation.images.cam_right_wrist.png").convert("RGB")
        ),
        "observation.state": np.zeros(trainer.policy.raw_state_dim, dtype=np.float32),
        "task": "press the bell",
    }
    return {
        "prompt": "press the bell",
        "formatted_obs": formatted_obs,
    }


def run_mock_validation(trainer):
    mock_episode = build_mock_episode(trainer)
    baseline, steered = validate_inference_paths(trainer, mock_episode)
    trainer.add_transition(
        steered["step_batch"],
        steered["steer_noise_flat"],
        reward=0.25,
        done=True,
        next_step_batch=trainer.policy.zero_step_batch(),
    )
    metrics = trainer.update()
    if metrics is not None:
        log_dict("validation/mock_sac_metrics", metrics)
    return {
        "baseline": baseline["diagnostics"],
        "steered": steered["diagnostics"],
        "mock_metrics": metrics,
    }


def train_single_task(config):
    save_root = Path(config["runner"]["save_root"]).expanduser().resolve()
    setup_logging(save_root)

    log_dict("startup/config_path", {"save_root": str(save_root)})
    trainer = ActionOnlyDsrlTrainer(config)
    log_dict("startup/report", trainer.startup_report())
    mock_report = run_mock_validation(trainer)
    log_dict("startup/mock_validation", mock_report)

    metrics_history = []
    total_success = 0
    task_env = None
    try:
        robowin_root = bootstrap_robowin_root(config["env"]["robowin_root"])
        args = build_task_args(
            robowin_root=robowin_root,
            task_name=config["env"]["task_name"],
            task_config=config["env"]["task_config"],
            save_root=save_root,
            policy_name="lingbot_action_only_dsrl",
        )

        task_env = class_decorator(config["env"]["task_name"])
        base_seed = int(config["runner"].get("seed", 0))
        max_episodes = int(config["runner"].get("max_episodes", 1))
        max_action_chunks = int(config["runner"].get("max_action_chunks", 2))

        for episode_idx in range(max_episodes):
            episode_seed, _ = find_valid_seed(
                task_env,
                args,
                base_seed + episode_idx,
                require_expert_check=bool(
                    config["env"].get("require_expert_check", False)
                ),
            )

            episode = prepare_episode(
                task_env,
                args,
                episode_idx=episode_idx,
                seed=episode_seed,
                instruction_type=config["env"].get("instruction_type", "seen"),
            )

            trainer.policy.reset(episode["prompt"])
            current_step = trainer.policy.act(
                episode["formatted_obs"],
                deterministic=False,
                use_dsrl=trainer.use_dsrl,
            )
            log_dict("episode/step0", current_step["diagnostics"])

            first_chunk = True
            episode_return = 0.0
            for chunk_idx in range(max_action_chunks):
                rollout = execute_action_chunk(
                    task_env,
                    current_step["action_chunk"],
                    prompt=episode["prompt"],
                    init_eef_pose=episode["init_eef_pose"],
                    first_chunk=first_chunk,
                )
                episode_return += rollout["reward"]

                if rollout["done"]:
                    next_step_batch = trainer.policy.zero_step_batch()
                    trainer.add_transition(
                        current_step["step_batch"],
                        current_step["steer_noise_flat"],
                        rollout["reward"],
                        True,
                        next_step_batch,
                    )
                    metrics = trainer.update()
                    if metrics is not None:
                        log_dict("train/metrics", metrics)
                        metrics_history.append(metrics)
                    total_success += int(rollout["success"])
                    break

                trainer.policy.compute_kv_cache(
                    rollout["key_frame_list"],
                    current_step["action_chunk"],
                )
                next_step = trainer.policy.act(
                    rollout["next_formatted_obs"],
                    deterministic=False,
                    use_dsrl=trainer.use_dsrl,
                )
                trainer.add_transition(
                    current_step["step_batch"],
                    current_step["steer_noise_flat"],
                    rollout["reward"],
                    False,
                    next_step["step_batch"],
                )
                metrics = trainer.update()
                if metrics is not None:
                    log_dict("train/metrics", metrics)
                    metrics_history.append(metrics)
                current_step = next_step
                first_chunk = False
            else:
                trainer.add_transition(
                    current_step["step_batch"],
                    current_step["steer_noise_flat"],
                    0.0,
                    True,
                    trainer.policy.zero_step_batch(),
                )
                metrics = trainer.update()
                if metrics is not None:
                    log_dict("train/metrics", metrics)
                    metrics_history.append(metrics)

            task_env.close_env(clear_cache=True)
            episode_summary = {
                "episode_idx": episode_idx,
                "seed": episode_seed,
                "return": episode_return,
                "successes": total_success,
                "replay_size": len(trainer.replay_buffer),
            }
            log_dict("episode/summary", episode_summary)
    except Exception as exc:
        log_dict(
            "robowin/blocker",
            {
                "error": str(exc),
                "current_run_status": "blocked_on_robowin_env",
            },
        )
    finally:
        try:
            if task_env is not None:
                task_env.close_env(clear_cache=True)
        except Exception:
            pass

    final_report = {
        "episodes": int(config["runner"].get("max_episodes", 1)),
        "successes": total_success,
        "metrics_logged": len(metrics_history),
        "current_run_status": (
            "finished" if total_success > 0 else "mock_validated_or_blocked"
        ),
    }
    log_dict("final/report", final_report)
    return final_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path.")
    args = parser.parse_args()

    config = load_config(args.config)
    train_single_task(config)


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()
