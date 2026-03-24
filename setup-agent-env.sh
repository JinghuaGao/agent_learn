#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash setup-agent-env.sh
#
# 说明：
# 1) 优先用 conda 完整环境文件复现
# 2) 若你只想在已有 Python 环境里安装 pip 包，可使用 requirements-agent-lock.txt

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] 未检测到 conda，请先安装 Miniconda/Anaconda。"
  exit 1
fi

ENV_NAME="agent"

echo "[INFO] 使用 environment-agent-full.yml 创建/更新 conda 环境: ${ENV_NAME}"
conda env update -n "$ENV_NAME" -f environment-agent-full.yml --prune || \
  conda env create -n "$ENV_NAME" -f environment-agent-full.yml

echo "[INFO] 使用 pip 锁定文件补齐依赖"
conda run -n "$ENV_NAME" python -m pip install -r requirements-agent-lock.txt

echo "[DONE] 环境准备完成。"
echo "激活环境：conda activate ${ENV_NAME}"
