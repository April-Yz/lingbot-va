# Environment Change Log

## 2026-03-16

### Entry 1
- Datetime: `2026-03-16`
- Conda environment: `lingbot-va`
- Command executed: `pip install sapien==3.0.0b1 mplib==0.2.1 --no-deps`
- What changed: installed RoboTwin runtime core packages required for `sapien` scene loading and motion planning bindings.
- Why it was needed: `lingbot-va` originally lacked the minimum RoboTwin simulator imports, so the new action-only DSRL training entry could not even import `envs`.
- Whether it succeeded: `yes`
- How it was verified: reran `ActionOnlyDsrlTrainer` construction and confirmed the import chain progressed past the previous `ModuleNotFoundError: No module named 'sapien'`.

### Entry 2
- Datetime: `2026-03-16`
- Conda environment: `lingbot-va`
- Command executed: `pip install 'setuptools<81'`
- What changed: downgraded `setuptools` to restore `pkg_resources`.
- Why it was needed: the freshly installed `sapien` wheel imports `pkg_resources` on startup, and the environment was missing that runtime path after the previous package state.
- Whether it succeeded: `yes`
- How it was verified: `sapien` import advanced past `pkg_resources` and the trainer startup report completed.

### Entry 3
- Datetime: `2026-03-16`
- Conda environment: `lingbot-va`
- Command executed: `pip install transforms3d==0.4.2 --no-deps`
- What changed: installed RoboTwin viewer/math helper package.
- Why it was needed: RoboTwin import then failed in `sapien.utils.viewer` because `transforms3d` was missing.
- Whether it succeeded: `yes`
- How it was verified: the training entry advanced past the earlier `ModuleNotFoundError: No module named 'transforms3d'`.

### Entry 4
- Datetime: `2026-03-16`
- Conda environment: `lingbot-va`
- Command executed: `pip install open3d==0.18.0 trimesh==4.4.3 zarr openai moviepy azure==4.0.0 azure-ai-inference 'pyglet<2'`
- What changed: installed the missing RoboTwin-side Python packages that were still absent from the LingBot environment.
- Why it was needed: after fixing `sapien`, the action-only DSRL entry still failed while importing RoboTwin utilities because packages such as `open3d` and `trimesh` were missing.
- Whether it succeeded: `yes`
- How it was verified: the import chain advanced beyond `open3d` and reached task-level environment setup.

### Entry 5
- Datetime: `2026-03-16`
- Conda environment: `lingbot-va`
- Command executed: `pip install toppra`
- What changed: installed TOPPRA.
- Why it was needed: RoboTwin task environment import then failed in `envs/_base_task.py` because `toppra` was missing.
- Whether it succeeded: `yes`
- How it was verified: the action-only DSRL entry advanced past the previous `ModuleNotFoundError: No module named 'toppra'`.

### Entry 6
- Datetime: `2026-03-16`
- Conda environment: `lingbot-va`
- Command executed: `pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable' --no-build-isolation`
- What changed: attempted to install `pytorch3d` from the upstream stable branch.
- Why it was needed: RoboTwin setup still hit a `missing pytorch3d` blocker during seed validation and environment setup.
- Whether it succeeded: `no`
- How it was verified: the build failed in `nvcc` with `Unsupported gpu architecture 'compute_120'` because the host CUDA toolchain is `12.1` while the GPU is Blackwell `sm_120`.

### Entry 7
- Datetime: `2026-03-16`
- Conda environment: `lingbot-va`
- Command executed: `pip install lxml`
- What changed: installed `lxml`.
- Why it was needed: RoboTwin online task startup reached `sapien` URDF loading and then failed with `ModuleNotFoundError: No module named 'lxml'`.
- Whether it succeeded: `yes`
- How it was verified: reran `click_bell.setup_demo(...)` inside the LingBot action-only pipeline and confirmed the environment initialized and returned observations.

### Entry 8
- Datetime: `2026-03-16`
- Conda environment: `RoboTwin-lingbot`
- Command executed: `pip install lxml`
- What changed: no package change was needed because `lxml` was already present.
- Why it was needed: the paired RoboTwin environment was checked to keep the two approved environments aligned for follow-up debugging.
- Whether it succeeded: `yes`
- How it was verified: `pip` reported `Requirement already satisfied: lxml`.

## Notes

- No new conda environments were created.
- No system-level packages were modified.
- All environment changes in this implementation pass were limited to the already approved `lingbot-va` and `RoboTwin-lingbot` conda environments.
