#!/usr/bin/env bash
set -euo pipefail

modelscope download Qwen/Qwen3-8B                            --local_dir ./Qwen3-8B
modelscope download google/gemma-3-12b-it                    --local_dir ./gemma-3-12b-it
modelscope download deepseek-ai/DeepSeek-R1-Distill-Llama-8B --local_dir ./DeepSeek-R1-Distill-Llama-8B
modelscope download LLM-Research/Meta-Llama-3.1-8B-Instruct  --local_dir ./Llama-3.1-8B-Instruct
