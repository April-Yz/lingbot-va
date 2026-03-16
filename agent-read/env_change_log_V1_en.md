# LingBot Action-Only DSRL V1 Environment Log

## Approved Environments

- `lingbot-va`
- `RoboTwin-lingbot`

## Recorded Changes

### 2026-03-16 | `lingbot-va`

- Installed: `sapien==3.0.0b1`, `mplib==0.2.1`
- Installed: `setuptools<81`
- Installed: `transforms3d==0.4.2`
- Installed: `open3d==0.18.0`, `trimesh==4.4.3`, `zarr`, `openai`, `moviepy`, `azure==4.0.0`, `azure-ai-inference`, `pyglet<2`
- Installed: `toppra`
- Installed: `lxml`
- Attempted and failed: `pytorch3d` build from upstream stable branch

## Why

These packages were required to let the LingBot-side action-only training entry import and run RoboTwin components inside the approved environment.

## Verification

- RoboTwin setup reached observation return after `lxml`
- online action-only episode completed end-to-end

## Notes

- `pytorch3d` is still unavailable on this host because the available CUDA 12.1 toolchain cannot build for Blackwell `sm_120`
- RoboTwin camera code now has a CPU fallback, so this is no longer a hard blocker for the validated RGB-based path
