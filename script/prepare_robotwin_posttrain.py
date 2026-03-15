#!/usr/bin/env python3

import argparse
import io
import json
import gc
import math
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_VA_ROOT = REPO_ROOT / "wan_va"
if str(WAN_VA_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_VA_ROOT))

from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from modules.utils import load_text_encoder, load_tokenizer, load_vae


CAMERA_SPECS = [
    ("head_camera", "observation.images.cam_high", (256, 320)),
    ("left_camera", "observation.images.cam_left_wrist", (128, 160)),
    ("right_camera", "observation.images.cam_right_wrist", (128, 160)),
]
RAW_CAMERA_HEIGHT = 480
RAW_CAMERA_WIDTH = 640

STATE_NAMES = [
    "left_eef_x",
    "left_eef_y",
    "left_eef_z",
    "left_eef_qx",
    "left_eef_qy",
    "left_eef_qz",
    "left_eef_qw",
    "left_gripper",
    "right_eef_x",
    "right_eef_y",
    "right_eef_z",
    "right_eef_qx",
    "right_eef_qy",
    "right_eef_qz",
    "right_eef_qw",
    "right_gripper",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert RoboTwin raw demos into a LingBot-VA post-training bundle."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, default="place_can_basket_demo_clean_lerobot")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin"),
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--instruction-key", type=str, default="seen", choices=["seen", "unseen"])
    parser.add_argument("--instruction-index", type=int, default=0)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--video-chunk-size", type=int, default=512)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def decode_image(blob: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(blob)) as image:
        return np.array(image.convert("RGB"))


def validate_raw_camera_shape(image: np.ndarray, raw_name: str, episode_index: int):
    expected_shape = (RAW_CAMERA_HEIGHT, RAW_CAMERA_WIDTH, 3)
    if image.shape != expected_shape:
        raise ValueError(
            f"Episode {episode_index} camera '{raw_name}' has shape {image.shape}, "
            f"but this converter is configured for Large_D435 raw frames {expected_shape}. "
            "Please recollect data with Large_D435 or adjust the converter explicitly."
        )


def pick_instruction(raw_dir: Path, episode_index: int, key: str, item_index: int) -> str:
    path = raw_dir / "instructions" / f"episode{episode_index}.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    choices = payload[key]
    if not choices:
        raise ValueError(f"No instructions under '{key}' in {path}")
    return choices[item_index % len(choices)]


def list_episode_indices(raw_dir: Path, episode_limit: int | None) -> list[int]:
    indices = sorted(
        int(path.stem.replace("episode", ""))
        for path in (raw_dir / "data").glob("episode*.hdf5")
    )
    if episode_limit is not None:
        indices = indices[:episode_limit]
    return indices


def load_episode(raw_dir: Path, episode_index: int) -> dict:
    hdf5_path = raw_dir / "data" / f"episode{episode_index}.hdf5"
    with h5py.File(hdf5_path, "r") as handle:
        left_pose = handle["endpose/left_endpose"][:]
        right_pose = handle["endpose/right_endpose"][:]
        left_gripper = handle["endpose/left_gripper"][:][:, None]
        right_gripper = handle["endpose/right_gripper"][:][:, None]

        state = np.concatenate([left_pose, left_gripper, right_pose, right_gripper], axis=1).astype(np.float32)
        images = {}
        for raw_name, _, _ in CAMERA_SPECS:
            dataset = handle[f"observation/{raw_name}/rgb"]
            decoded_images = [decode_image(blob) for blob in dataset]
            for image in decoded_images:
                validate_raw_camera_shape(image, raw_name, episode_index)
            images[raw_name] = decoded_images

    raw_frames = state.shape[0]
    action_frames = int(math.ceil(raw_frames / 4.0) * 4)
    latent_video_frames = action_frames - 3
    if latent_video_frames < 1:
        raise ValueError(f"Episode {episode_index} has only {state.shape[0]} frames.")

    if raw_frames < latent_video_frames:
        raise ValueError(
            f"Episode {episode_index} has {raw_frames} frames, which is insufficient for latent window {latent_video_frames}."
        )

    if raw_frames < action_frames:
        pad_num = action_frames - raw_frames
        state = np.concatenate([state, np.repeat(state[-1:], pad_num, axis=0)], axis=0)
        for key in images:
            images[key] = images[key] + [images[key][-1].copy() for _ in range(pad_num)]
    else:
        state = state[:action_frames]
        for key in images:
            images[key] = images[key][:action_frames]

    for key in images:
        assert len(images[key]) == action_frames

    return {
        "episode_index": episode_index,
        "state": state,
        "action": state.copy(),
        "images": images,
        "action_frames": action_frames,
        "latent_video_frames": latent_video_frames,
    }


def create_dataset(repo_root: Path, repo_id: str, fps: int) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": [STATE_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (16,),
            "names": [STATE_NAMES],
        },
    }
    for _, feature_name, _ in CAMERA_SPECS:
        features[feature_name] = {
            "dtype": "video",
            "shape": (3, RAW_CAMERA_HEIGHT, RAW_CAMERA_WIDTH),
            "names": ["channels", "height", "width"],
        }

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=repo_root,
        fps=fps,
        robot_type="robotwin",
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
    )


def write_lerobot_dataset(dataset: LeRobotDataset, raw_dir: Path, episode_indices: list[int], instruction_key: str, instruction_index: int):
    specs = []
    for episode_index in tqdm(episode_indices, desc="Convert episodes -> LeRobot"):
        episode = load_episode(raw_dir, episode_index)
        task_text = pick_instruction(raw_dir, episode_index, instruction_key, instruction_index)
        for frame_idx in range(episode["action_frames"]):
            frame = {
                "observation.state": episode["state"][frame_idx],
                "action": episode["action"][frame_idx],
            }
            for raw_name, feature_name, _ in CAMERA_SPECS:
                frame[feature_name] = episode["images"][raw_name][frame_idx]
            dataset.add_frame(frame, task=task_text)
        dataset.save_episode()
        specs.append(
            {
                "episode_index": episode_index,
                "action_frames": episode["action_frames"],
                "latent_video_frames": episode["latent_video_frames"],
                "task_text": task_text,
            }
        )
    dataset.stop_image_writer()
    return specs


def add_action_config(repo_root: Path, specs: list[dict]):
    by_episode = {item["episode_index"]: item for item in specs}
    episodes_path = repo_root / "meta" / "episodes.jsonl"
    updated = []
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            spec = by_episode[payload["episode_index"]]
            payload["length"] = spec["action_frames"]
            payload["action_config"] = [
                {
                    "start_frame": 0,
                    "end_frame": spec["action_frames"],
                    "action_text": spec["task_text"],
                }
            ]
            updated.append(json.dumps(payload, ensure_ascii=False))
    episodes_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


class PromptEncoder:
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.tokenizer = load_tokenizer(str(model_path / "tokenizer"))
        self.text_encoder = load_text_encoder(
            str(model_path / "text_encoder"),
            torch_dtype=dtype,
            torch_device=device,
        )

    @torch.no_grad()
    def encode(self, prompt: str, max_sequence_length: int = 226) -> torch.Tensor:
        tokens = self.tokenizer(
            [prompt_clean(prompt)],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(
            tokens.input_ids.to(self.device),
            tokens.attention_mask.to(self.device),
        ).last_hidden_state
        return prompt_embeds[0].to(dtype=self.dtype, device="cpu")


class CameraLatentEncoder:
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype, video_chunk_size: int):
        self.device = device
        self.dtype = dtype
        self.video_chunk_size = video_chunk_size
        self.vae = load_vae(
            str(model_path / "vae"),
            torch_dtype=dtype,
            torch_device=device,
        )
        self.latents_mean = torch.tensor(self.vae.config.latents_mean, device=device)
        self.inv_latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=device)

    @torch.no_grad()
    def encode_video(self, frames: list[np.ndarray], image_size: tuple[int, int]) -> dict:
        height, width = image_size
        video = torch.from_numpy(np.stack(frames)).float().permute(3, 0, 1, 2)
        video = F.interpolate(
            video,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(0)
        video = video / 255.0 * 2.0 - 1.0

        posterior = self.vae.encode(video.to(self.device, dtype=self.dtype)).latent_dist
        mu = posterior.mean
        mu = ((mu.float() - self.latents_mean.view(1, -1, 1, 1, 1)) * self.inv_latents_std.view(1, -1, 1, 1, 1))
        latent = mu[0].permute(1, 2, 3, 0).reshape(-1, mu.shape[1]).to(dtype=self.dtype, device="cpu")
        return {
            "latent": latent,
            "latent_num_frames": int(mu.shape[2]),
            "latent_height": int(mu.shape[3]),
            "latent_width": int(mu.shape[4]),
            "video_num_frames": len(frames),
            "video_height": height,
            "video_width": width,
        }


def save_empty_embedding(bundle_dir: Path, prompt_encoder: PromptEncoder):
    torch.save(prompt_encoder.encode(""), bundle_dir / "empty_emb.pt")


def precompute_prompt_embeddings(specs: list[dict], prompt_encoder: PromptEncoder) -> dict[int, torch.Tensor]:
    embeddings = {}
    for spec in tqdm(specs, desc="Encode text prompts"):
        embeddings[spec["episode_index"]] = prompt_encoder.encode(spec["task_text"])
    return embeddings


def write_latents(repo_root: Path, raw_dir: Path, specs: list[dict], metadata: LeRobotDatasetMetadata, prompt_embeddings: dict[int, torch.Tensor], latent_encoder: CameraLatentEncoder):
    latents_root = repo_root / "latents"
    for spec in tqdm(specs, desc="Extract text/video latents"):
        episode = load_episode(raw_dir, spec["episode_index"])
        prompt_emb = prompt_embeddings[spec["episode_index"]]
        frame_ids = list(range(spec["latent_video_frames"]))
        chunk_id = metadata.get_episode_chunk(spec["episode_index"])
        for raw_name, feature_name, image_size in CAMERA_SPECS:
            camera_latent = latent_encoder.encode_video(
                episode["images"][raw_name][: spec["latent_video_frames"]],
                image_size,
            )
            payload = {
                **camera_latent,
                "text_emb": prompt_emb.clone(),
                "text": spec["task_text"],
                "frame_ids": frame_ids,
                "start_frame": 0,
                "end_frame": spec["action_frames"],
                "fps": metadata.fps,
                "ori_fps": metadata.fps,
            }
            out_dir = latents_root / f"chunk-{chunk_id:03d}" / feature_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"episode_{spec['episode_index']:06d}_0_{spec['action_frames']}.pth"
            torch.save(payload, out_path)


def write_bundle_readme(bundle_dir: Path, repo_id: str, raw_dir: Path, model_path: Path):
    repo_root = bundle_dir / repo_id
    content = f"""# LingBot-VA Post-Training Bundle

This bundle was generated from:

- Raw RoboTwin data: `{raw_dir}`
- LeRobot repo: `{repo_root}`
- Empty prompt embedding: `{bundle_dir / 'empty_emb.pt'}`
- Model assets used for text/video encoding: `{model_path}`

## Processing Command

```bash
cd /home/zaijia001/vam/lingbot-va
/home/zaijia001/ssd/miniconda3/envs/lingbot-va/bin/python script/prepare_robotwin_posttrain.py \\
  --raw-dir {raw_dir} \\
  --bundle-dir {bundle_dir} \\
  --repo-id {repo_id} \\
  --model-path {model_path} \\
  --instruction-key seen \\
  --instruction-index 0 \\
  --overwrite
```

## Training Command

```bash
cd /home/zaijia001/vam/lingbot-va
NGPU=1 CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh \\
  --dataset-path {bundle_dir} \\
  --empty-emb-path {bundle_dir / 'empty_emb.pt'} \\
  --save-root /home/zaijia001/vam/lingbot-va/train_out/place_can_basket_demo_clean \\
  --enable-wandb false
```

## Frame Alignment Rule

- LeRobot action/state/video length is padded to the nearest upper multiple of `4`
- Latent extraction uses the first `length - 3` frames so Wan VAE sees `1 (mod 4)` frames

## Raw Camera Assumption

- This converter is intentionally strict about the raw RoboTwin camera frames.
- Expected raw frame size for `head_camera`, `left_camera`, `right_camera`: `480x640`
- If the raw HDF5 contains another size, the converter raises an error instead of silently adapting.
"""
    (bundle_dir / "README.md").write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    if args.bundle_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.bundle_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.bundle_dir)
    args.bundle_dir.mkdir(parents=True, exist_ok=True)

    repo_root = args.bundle_dir / args.repo_id
    dataset = create_dataset(repo_root=repo_root, repo_id=args.repo_id, fps=args.fps)
    episode_indices = list_episode_indices(args.raw_dir, args.episode_limit)
    specs = write_lerobot_dataset(
        dataset=dataset,
        raw_dir=args.raw_dir,
        episode_indices=episode_indices,
        instruction_key=args.instruction_key,
        instruction_index=args.instruction_index,
    )
    add_action_config(repo_root, specs)

    device = torch.device(args.device)
    dtype = get_torch_dtype(args.dtype)
    prompt_encoder = PromptEncoder(
        args.model_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    prompt_embeddings = precompute_prompt_embeddings(specs, prompt_encoder)
    save_empty_embedding(args.bundle_dir, prompt_encoder)
    del prompt_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    latent_encoder = CameraLatentEncoder(
        args.model_path,
        device=device,
        dtype=dtype,
        video_chunk_size=args.video_chunk_size,
    )
    metadata = LeRobotDatasetMetadata(args.repo_id, root=repo_root, force_cache_sync=False)
    write_latents(
        repo_root=repo_root,
        raw_dir=args.raw_dir,
        specs=specs,
        metadata=metadata,
        prompt_embeddings=prompt_embeddings,
        latent_encoder=latent_encoder,
    )
    write_bundle_readme(
        bundle_dir=args.bundle_dir,
        repo_id=args.repo_id,
        raw_dir=args.raw_dir,
        model_path=args.model_path,
    )
    summary = {
        "raw_dir": str(args.raw_dir),
        "bundle_dir": str(args.bundle_dir),
        "repo_root": str(repo_root),
        "episodes_processed": len(specs),
        "min_action_frames": int(min(item["action_frames"] for item in specs)),
        "max_action_frames": int(max(item["action_frames"] for item in specs)),
        "min_latent_video_frames": int(min(item["latent_video_frames"] for item in specs)),
        "max_latent_video_frames": int(max(item["latent_video_frames"] for item in specs)),
        "instruction_key": args.instruction_key,
        "instruction_index": args.instruction_index,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
