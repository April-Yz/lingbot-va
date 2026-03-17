# LingBot Frozen-Noise-DSRL V2 Change Log

## Scope

This change set only covers V2 static code integration and V2 documentation handoff. It does not include real training, real evaluation, environment installation, environment upgrades, or environment testing.

## Added Files

### `wan_va/frozen_noise_dsrl/__init__.py`

- Purpose: export entry for the V2 package
- Impact scope: V2 only
- Affects original flow: no
- Status: static integration only, not runtime-validated

### `wan_va/frozen_noise_dsrl/modules.py`

- Purpose: add the future-latent-conditioned noise-policy modules and critic wrappers
- Impact scope: V2 only
- Affects original flow: no
- Status: static integration only, not runtime-validated

### `wan_va/frozen_noise_dsrl/policy.py`

- Purpose: add the frozen LingBot backbone V2 policy / trainer
- Impact scope: V2 only
- Affects original flow: no
- Status: static integration only, not runtime-validated

### `script/run_lingbot_frozen_noise_dsrl_v2.py`

- Purpose: provide a dedicated V2 training entrypoint
- Impact scope: V2 only
- Affects original flow: no
- Status: static integration only, not runtime-validated

### `script/run_lingbot_frozen_noise_eval_v2.sh`

- Purpose: provide a dedicated static V2 evaluation entry
- Impact scope: V2 only
- Affects original flow: no
- Status: static integration only, not runtime-validated

### `examples/embodiment/config/robotwin_lingbot_frozen_noise_dsrl_v2.yaml`

- Purpose: provide a dedicated V2 config
- Impact scope: V2 only
- Affects original flow: no
- Status: static integration only, not runtime-validated

### `agent-read/V2/IMPLEMENTATION_REPORT_V2_CN.md`

- Purpose: Chinese V2 implementation report
- Impact scope: documentation
- Affects original flow: no
- Status: static documentation

### `agent-read/V2/IMPLEMENTATION_REPORT_V2_EN.md`

- Purpose: English V2 implementation report
- Impact scope: documentation
- Affects original flow: no
- Status: static documentation

### `agent-read/V2/CHANGELOG_V2_CN.md`

- Purpose: Chinese V2 file-level change log
- Impact scope: documentation
- Affects original flow: no
- Status: static documentation

### `agent-read/V2/CHANGELOG_V2_EN.md`

- Purpose: English V2 file-level change log
- Impact scope: documentation
- Affects original flow: no
- Status: static documentation

### `agent-read/V2/ENVIRONMENT_RECORD_V2_CN.md`

- Purpose: Chinese environment-range record
- Impact scope: documentation
- Affects original flow: no
- Status: static documentation

### `agent-read/V2/ENVIRONMENT_RECORD_V2_EN.md`

- Purpose: English environment-range record
- Impact scope: documentation
- Affects original flow: no
- Status: static documentation

### `agent-read/V2/RUN_GUIDE_V2_CN.md`

- Purpose: Chinese expected command guide
- Impact scope: documentation
- Affects original flow: no
- Status: static documentation, commands not executed

### `agent-read/V2/RUN_GUIDE_V2_EN.md`

- Purpose: English expected command guide
- Impact scope: documentation
- Affects original flow: no
- Status: static documentation, commands not executed

### `agent-read/V2/REPRO_CHECKLIST_V2_CN.md`

- Purpose: Chinese migration-time verification checklist
- Impact scope: documentation
- Affects original flow: no
- Status: static documentation, not executed

### `agent-read/V2/REPRO_CHECKLIST_V2_EN.md`

- Purpose: English migration-time verification checklist
- Impact scope: documentation
- Affects original flow: no
- Status: static documentation, not executed

### `agent-read/V2/ENV_PREDICTION_V2_CN.md`

- Purpose: Chinese environment-risk prediction
- Impact scope: documentation
- Affects original flow: no
- Status: static documentation

### `agent-read/V2/ENV_PREDICTION_V2_EN.md`

- Purpose: English environment-risk prediction
- Impact scope: documentation
- Affects original flow: no
- Status: static documentation

## Modified Files

### `agent-read/README.md`

- Purpose: record the V2 static integration in the project overview
- Impact scope: project documentation
- Affects original flow: no
- Status: documentation update, not runtime-validated

### `agent-read/CHANGELOG.md`

- Purpose: record the 2026-03-18 V2 static integration in the repo-level log
- Impact scope: project log
- Affects original flow: no
- Status: documentation update, not runtime-validated

## Deleted Files

- None

## Compatibility Note

- baseline flow remains untouched
- original post-train flow remains untouched
- V1 flow remains untouched
- the current result is a static integration version and has not been runtime-validated
