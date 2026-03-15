import argparse
import json
import re
import sys
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
WAN_VA_ROOT = REPO_ROOT / "wan_va"
if str(WAN_VA_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_VA_ROOT))

from configs import VA_CONFIGS
from modules.utils import load_vae


def latent_sort_key(path: Path) -> int:
    match = re.search(r"latents_(\d+)\.pt$", path.name)
    if match is None:
        raise ValueError(f"Unexpected latent filename: {path}")
    return int(match.group(1))


class LatentDecoder:

    def __init__(self, config_name: str):
        self.config = VA_CONFIGS[config_name]
        self.vae = load_vae(
            str(Path(self.config.wan22_pretrained_model_name_or_path) / "vae"),
            torch_dtype=torch.float32,
            torch_device="cpu",
        )
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )

    @torch.no_grad()
    def decode_latent_video(self, latents: torch.Tensor):
        latents = latents.to(dtype=torch.float32, device="cpu")
        latents = latents / self.latents_std.to(latents) + self.latents_mean.to(latents)
        video = self.vae.decode(latents, return_dict=False)[0]
        return self.video_processor.postprocess_video(video, output_type="np")[0]


def decode_episode(decoder: LatentDecoder, episode_record: dict, fps: int) -> dict:
    exp_save_root = Path(episode_record["server_exp_save_root"])
    latent_files = sorted(exp_save_root.glob("latents_*.pt"), key=latent_sort_key)
    if not latent_files:
        raise FileNotFoundError(f"No latents found under {exp_save_root}")

    latent_chunks = [torch.load(path, map_location="cpu") for path in latent_files]
    latents = torch.cat(latent_chunks, dim=2)
    decoded_video = decoder.decode_latent_video(latents)

    output_path = Path(episode_record["latent_decode_video_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(decoded_video, str(output_path), fps=fps)

    return {
        "episode_idx": episode_record["episode_idx"],
        "seed": episode_record["seed"],
        "success": episode_record["success"],
        "prompt": episode_record["prompt"],
        "server_exp_save_root": str(exp_save_root),
        "latent_chunk_count": len(latent_files),
        "latent_files": [str(path) for path in latent_files],
        "latent_decode_video_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--config-name", type=str, default="robotwin")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    decoder = LatentDecoder(args.config_name)
    decode_results = []
    for episode_record in manifest.get("episodes", []):
        if not episode_record.get("server_exp_save_root"):
            continue
        decode_results.append(decode_episode(decoder, episode_record, args.fps))

    output_path = manifest_path.with_name("latent_decode_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config_name": args.config_name,
                "manifest_path": str(manifest_path),
                "decoded_episode_count": len(decode_results),
                "episodes": decode_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
