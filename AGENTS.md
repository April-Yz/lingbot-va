# Project Agent Rules

These rules apply to all work inside `/home/zaijia001/vam/lingbot-va`.

## Scope Protection

1. Allowed edit scope for this project line of work:
   - `/home/zaijia001/vam/lingbot-va`
   - `/home/zaijia001/vam/RoboTwin-lingbot`
   - conda envs `lingbot-va` and `RoboTwin-lingbot`
2. Do not modify `/home/zaijia001/ssd/RoboTwin` or any other conda environment unless the user explicitly approves it again.
3. Prefer project-local fixes, local worktrees, and conda-local installs over global or system-wide changes.

## Debug Record Rules

1. Every time a meaningful runtime issue, environment issue, or regression is investigated, add a debug record under `agent-read/`.
2. Each debug record should state:
   - what failed
   - the observed symptoms or traceback
   - the root cause or current hypothesis
   - the fix, workaround, or remaining blocker
3. Update `agent-read/README.md` and `agent-read/CHANGELOG.md` whenever a new debug record or workflow change is added.

## Command Documentation Rules

1. Whenever a user-facing command changes, also update the command reference docs in the same turn.
2. Keep the bilingual command index synchronized:
   - `agent-read/command-index.md`
   - `agent-read/command-index_ZH.md`
3. Keep the detailed bilingual baseline docs synchronized when baseline data, training, eval, or decoder commands change.
4. GPU-bound commands should include an explicit device selection example such as `CUDA_VISIBLE_DEVICES=...` unless the command is intentionally device-agnostic.
5. If a command emits a known benign warning, document that warning next to the command so the user does not need to re-ask whether it is fatal.

## Eval And Decoder Notes

1. If RoboTwin eval output naming changes, keep both the command docs and the relevant baseline docs updated.
2. If latent decoding depends on a manifest path or model tag, document the exact path pattern and an immediately runnable example.
