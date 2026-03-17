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

from wan_va.action_only_dsrl.robotwin_env import (
    bootstrap_robowin_root,
    build_task_args,
    class_decorator,
    execute_action_chunk,
    find_valid_seed,
    prepare_episode,
)
from wan_va.frozen_noise_dsrl import LingBotFrozenNoiseV2Trainer


LOGGER = logging.getLogger("lingbot_frozen_noise_dsrl_v2")


def setup_logging(save_root):
    save_root.mkdir(parents=True, exist_ok=True)
    log_path = save_root / "frozen_noise_dsrl_v2.log"
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


def init_wandb(config):
    wandb_cfg = config.get("wandb", {})
    if not bool(wandb_cfg.get("enable", False)):
        return None

    import wandb

    wandb_base_url = os.getenv("WANDB_BASE_URL")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        login_kwargs = {"key": wandb_api_key}
        if wandb_base_url:
            login_kwargs["host"] = wandb_base_url
        wandb.login(**login_kwargs)

    return wandb.init(
        entity=os.getenv(
            "WANDB_TEAM_NAME",
            wandb_cfg.get("entity", "haoyuan-lingbot"),
        ),
        project=os.getenv(
            "WANDB_PROJECT",
            wandb_cfg.get("project", "lingbot"),
        ),
        name=os.getenv("WANDB_RUN_NAME", wandb_cfg.get("run_name")),
        config=config,
        mode=wandb_cfg.get("mode", "online"),
    )


def maybe_log_wandb(run, payload, step=None):
    if run is None or payload is None:
        return
    run.log(payload, step=step)


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
    wandb_run = init_wandb(config)

    log_dict("startup/config_path", {"save_root": str(save_root)})
    maybe_log_wandb(wandb_run, {"startup/save_root": str(save_root)})
    trainer = LingBotFrozenNoiseV2Trainer(config)
    startup_report = trainer.startup_report()
    log_dict("startup/report", startup_report)
    maybe_log_wandb(wandb_run, {f"startup/{k}": v for k, v in startup_report.items()})
    mock_report = run_mock_validation(trainer)
    log_dict("startup/mock_validation", mock_report)
    if mock_report.get("mock_metrics") is not None:
        maybe_log_wandb(
            wandb_run,
            mock_report["mock_metrics"],
            step=int(mock_report["mock_metrics"].get("train/global_step", 0)),
        )

    metrics_history = []
    total_success = 0
    episodes_completed = 0
    blocked_error = None
    task_env = None
    try:
        robowin_root = bootstrap_robowin_root(config["env"]["robowin_root"])
        args = build_task_args(
            robowin_root=robowin_root,
            task_name=config["env"]["task_name"],
            task_config=config["env"]["task_config"],
            save_root=save_root,
            policy_name="lingbot_frozen_noise_dsrl_v2",
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
                        maybe_log_wandb(
                            wandb_run,
                            metrics,
                            step=int(metrics.get("train/global_step", 0)),
                        )
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
                    maybe_log_wandb(
                        wandb_run,
                        metrics,
                        step=int(metrics.get("train/global_step", 0)),
                    )
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
                    maybe_log_wandb(
                        wandb_run,
                        metrics,
                        step=int(metrics.get("train/global_step", 0)),
                    )
                    metrics_history.append(metrics)

            task_env.close_env(clear_cache=True)
            episodes_completed += 1
            episode_summary = {
                "episode_idx": episode_idx,
                "seed": episode_seed,
                "return": episode_return,
                "successes": total_success,
                "replay_size": len(trainer.replay_buffer),
            }
            log_dict("episode/summary", episode_summary)
            maybe_log_wandb(
                wandb_run,
                {
                    "episode/index": episode_idx,
                    "episode/seed": episode_seed,
                    "episode/return": episode_return,
                    "episode/successes": total_success,
                    "episode/replay_size": len(trainer.replay_buffer),
                },
                step=episode_idx + 1,
            )
    except Exception as exc:
        blocked_error = str(exc)
        blocker_payload = {
            "error": blocked_error,
            "current_run_status": "blocked_on_robowin_env",
        }
        log_dict("robowin/blocker", blocker_payload)
        maybe_log_wandb(wandb_run, {"robowin/blocked": 1, "robowin/error": blocked_error})
    finally:
        try:
            if task_env is not None:
                task_env.close_env(clear_cache=True)
        except Exception:
            pass

    if blocked_error is not None:
        current_run_status = "blocked_on_robowin_env"
    elif total_success > 0:
        current_run_status = "finished_success"
    elif episodes_completed > 0:
        current_run_status = "finished_no_success"
    else:
        current_run_status = "mock_validated_only"

    final_report = {
        "episodes": int(config["runner"].get("max_episodes", 1)),
        "episodes_completed": episodes_completed,
        "successes": total_success,
        "metrics_logged": len(metrics_history),
        "current_run_status": current_run_status,
        "blocked_error": blocked_error,
    }
    log_dict("final/report", final_report)
    maybe_log_wandb(wandb_run, {f"final/{k}": v for k, v in final_report.items()})
    if wandb_run is not None:
        wandb_run.finish()
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
