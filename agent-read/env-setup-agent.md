# Lingbot-va Environment Setup For Agents

## 1. Goal

Build a stable LingBot-VA runtime and post-training environment that can interoperate with RoboTwin-lingbot.

## 2. Scope And Paths

- Repo root: `lingbot-va`
- Expected conda env name: `lingbot-va`
- Core dependency files:
  - `requirements.txt`
  - `pyproject.toml`

## 3. Python And CUDA Baseline

From repository docs and local usage:

- Python: 3.10.x (recommended 3.10.16)
- Torch baseline in README: 2.9.0 with cu126 wheels
- Local Blackwell-friendly variant used in this workspace: cu128 wheel set

Choose one torch wheel profile and keep it consistent across `torch/torchvision/torchaudio`.

## 4. Conda Environment

```bash
conda create -n lingbot-va python=3.10.16 -y
conda activate lingbot-va
```

## 5. Install Runtime Dependencies

Option A: Follow README-style explicit install (torch first, then packages)

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Option B: Install package itself after deps:

```bash
pip install .
```

## 6. flash-attn Handling

If flash-attn can compile/install:

```bash
pip install flash-attn --no-build-isolation
```

If flash-attn is unavailable or ABI-incompatible, use `attn_mode=torch` for inference/eval.

Important mode rule from project docs:

- Training: `attn_mode=flex`
- Inference/eval: `attn_mode=torch` or `attn_mode=flashattn`

## 7. Post-Training Extra Dependencies

```bash
pip install lerobot==0.3.3 scipy wandb --no-deps
```

Reason: avoid pulling conflicting transitive dependencies into the main LingBot stack.

## 8. Minimal Validation

```bash
python -c "import torch, diffusers, transformers; print(torch.__version__)"
python -c "import wan_va; print('wan_va import ok')"
```

## 9. Interop With RoboTwin

LingBot eval uses RoboTwin in a separate repository/environment:

- RoboTwin repo: `/home/zaijia001/vam/RoboTwin-lingbot`
- RoboTwin env: `RoboTwin-lingbot`

Set `ROBOTWIN_ROOT` accordingly when running eval client/server scripts.

## 10. Common Failure Points

- Torch CUDA wheel mismatch with host driver/GPU arch.
- flash-attn build/ABI issues causing runtime import failures.
- Mixing RoboTwin-side heavy deps into the wrong environment.
- Forgetting to switch `attn_mode` between train and eval.
