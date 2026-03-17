#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/home/zaijia001/vam/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin}
TASK_NAME=${TASK_NAME:-click_bell}
TASK_CONFIG=${TASK_CONFIG:-demo_clean_large_d435}
RESULT_ROOT=${RESULT_ROOT:-/home/zaijia001/vam/RoboTwin-lingbot/results_frozen_noise_v2}
TEST_NUM=${TEST_NUM:-1}
MODEL_TAG=${MODEL_TAG:-frozen-noise-v2}

echo "Static helper only. Do not treat this script as validated on the current server."
echo "1. Launch the LingBot server in a separate shell:"
echo "   MODEL_PATH=${MODEL_PATH} bash evaluation/robotwin/launch_server.sh"
echo
echo "2. Launch the RoboTwin client:"
echo "   python /home/zaijia001/vam/lingbot-va/evaluation/robotwin/eval_polict_client_openpi.py \\"
echo "     --task_name ${TASK_NAME} \\"
echo "     --task_config ${TASK_CONFIG} \\"
echo "     --save_dir ${RESULT_ROOT} \\"
echo "     --test_num ${TEST_NUM} \\"
echo "     --model_tag ${MODEL_TAG}"
