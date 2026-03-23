# Evaluation（Ragas 实验区）

这里用于快速验证 `ragas` 的评估流程。

## 1) 本目录内容

- [evaluation/ragas_demo.py](evaluation/ragas_demo.py)：最小可运行示例（2 条样本）
- `ragas_demo_result.csv`：运行后自动生成的结果文件

## 2) 运行前准备

你当前可用 Python：
- `/Users/jiean/miniconda3/envs/agent/bin/python`

至少设置一个 API Key：
- `NV_API_KEY` 或 `NVIDIA_API_KEY` 或 `OPENAI_API_KEY`

可选参数：
- `OPENAI_BASE_URL`（默认 `https://integrate.api.nvidia.com/v1`）
- `RAGAS_EVAL_MODEL`（默认读取 `OPENAI_MODEL`，再回退 `minimaxai/minimax-m2.5`）

## 3) 运行 demo

在仓库根目录执行：

`/Users/jiean/miniconda3/envs/agent/bin/python evaluation/ragas_demo.py`

## 4) 结果说明

脚本会输出并保存以下指标：
- `faithfulness`
- `context_precision`
- `context_recall`

这些分数用于快速判断：
- 回答是否忠于检索证据
- 检索上下文是否有用
- 检索是否覆盖到回答所需信息

## 5) 参考文档

Ragas 官方文档：
https://docs.ragas.io/en/stable/getstarted/evals/#project-structure