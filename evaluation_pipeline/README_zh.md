# 文档解析任务评测流程

本目录提供 Xiaomi-OCR-0 文档解析任务的两条推理路线：

1. **端到端（e2e）**：整页图片直接推理得到 Markdown。
2. **两阶段（two-stage）**：先用 PP-DocLayoutV3(PaddleX版) 检测版面区域，再逐个区域识别并组装为页面级 Markdown。

两条路线都使用 OmniDocBench v1.6 的评测代码和相同配置进行评测。我们使用的配置已公开在 [`configs/end2end.yaml`](configs/end2end.yaml)。英文说明见 [README.md](README.md)。

## 环境版本

评测环境参考 [OmniDocBench 的 Environment Setup and Running 说明](https://github.com/opendatalab/OmniDocBench/blob/main/README.md#environment-setup-and-running)，主要组件如下：

| Component | Version |
| --- | --- |
| Python | 3.10.x |
| TeX Live | 2025 |
| pdflatex | 3.141592653-2.6-1.40.28 |
| ImageMagick | 7.1.1-47 |
| Ghostscript | 9.55.0 |

评测使用的 Docker 镜像来自：

```text
ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204
```

该镜像用于计算评测指标。GPU 推理需要单独准备支持 Qwen3.5 多模态模型的 vLLM 环境，并安装仓库根目录的 `requirements.txt`；两阶段的版面检测还需要兼容的 PaddlePaddle 和 `requirements-paddlex.txt`。具体安装见 [INSTALL.md](../INSTALL.md)。


## 输入数据格式

以下命令均在仓库根目录运行。输入为 JSONL 文件，每行包含一张页面图片的路径：

```json
{"images": ["/absolute/path/to/page.png"]}
```

图片路径需能被推理进程访问。输出按 `<图片文件名>.md` 命名，因此输入图片的文件名应唯一，并与评测标注中的页面名称对应。评测标注需使用 OmniDocBench 支持的 JSON 格式。

按实际位置设置输入文件和模型路径：

```bash
MODEL=SeerRay-Lab/Xiaomi-OCR-0
INPUT_JSONL=/absolute/path/to/images.jsonl
OUTPUT_DIR="$PWD/outputs"
GPUS=0,1,2,3,4,5,6,7
mkdir -p "$OUTPUT_DIR"
```

## vLLM 推理超参数

端到端和两阶段的主推理使用以下生成参数：

| 参数 | 数值 |
| --- | --- |
| `temperature` | `0.0` |
| `top_p` | `0.8` |
| `presence_penalty` | `0.0` |
| `repetition_penalty` | `1.0` |
| `max_tokens` | `16384` |
| `chat_template_kwargs.enable_thinking` | `false` |
| 请求超时 | `600 s` |



## 端到端推理

使用 [`e2e_img2md.py`](e2e_img2md.py) 生成逐页 Markdown：

```bash
python -m evaluation_pipeline.e2e_img2md \
  --input-jsonl "$INPUT_JSONL" \
  --output_dir "$OUTPUT_DIR/e2e" \
  --model_path "$MODEL" --served_model_name Xiaomi-OCR-0 \
  --task document --gpu_devices "$GPUS" --port 26000 \
  --gpu_mem_util 0.8 --workers 8 \
  --presence_penalty 0.0 --repetition_penalty 1.0 \
  --max_tokens 16384 --request_timeout 600 \
  --retry-repetitive --retry-presence-penalty 1.5 \
  --start_server --stop_server
```

后处理会将 OTSL 转成最终 Markdown 中的 HTML 表格，评测使用最终的逐页 `.md` 文件。

## 两阶段推理

### 1. 用 PaddleX 生成 layout

使用 [`layout_detect_paddlex.py`](layout_detect_paddlex.py)，通过 **PaddleX 版 PP-DocLayoutV3，阈值 0.25** 生成版面检测结果：

```bash
LAYOUT_JSONL="$OUTPUT_DIR/layout_paddlex-th0.25.jsonl"
python -m evaluation_pipeline.layout_detect_paddlex \
  --input_jsonl "$INPUT_JSONL" \
  --output_jsonl "$LAYOUT_JSONL" \
  --gpu_ids "$GPUS" --batch_size 8 --threshold 0.25
```

首次运行时 PaddleX 会下载 PP-DocLayoutV3；也可以通过 `--model_dir` 指定本地 **PaddleX 格式**权重。

### 2. 分区域识别并组装页面

在 vLLM 环境中运行 [`twostage_img2md.py`](twostage_img2md.py)，通过 `--layout-jsonl` 传入上一阶段生成的文件：

```bash
python -m evaluation_pipeline.twostage_img2md \
  --input-jsonl "$INPUT_JSONL" \
  --layout-jsonl "$LAYOUT_JSONL" \
  --output_dir "$OUTPUT_DIR/stage2" \
  --model_path "$MODEL" --served_model_name Xiaomi-OCR-0 \
  --vlm_gpus "$GPUS" --port 26000 --gpu_mem_util 0.8 --workers 8 \
  --temperature 0.0 --top_p 0.8 \
  --presence_penalty 0.0 --repetition_penalty 1.0 \
  --max_tokens 16384 --request_timeout 600 \
  --retry-repetitive --retry-presence-penalty 1.5 \
  --start_server --stop_server
```

两阶段复现需保持以下设置：

- **复现时传入 PaddleX 生成的 `--layout-jsonl`。** 若改用 `--layout_model_dir`，脚本会通过 HF Transformers 加载 layout 模型，该路径默认阈值为 `0.3`。
- 输入 JSONL 和 layout JSONL 中的图片路径字符串必须完全一致，脚本按该路径关联检测结果。
- **每个 PaddleX 阈值都必须独立运行。** 原生后处理使高阈值结果不一定是低阈值结果按 score 过滤后的子集，不能通过离线过滤替代重新生成。比较阈值时使用独立的 layout 文件。
- 区域表格同样使用 OTSL 提示词，在页面组装时转换为 HTML。评测对象是最终的页面级 Markdown。

## 评测

复用 Docker 镜像内的 OmniDocBench v1.6 `pdf_validation.py`，配置使用本仓库的 [`configs/end2end.yaml`](configs/end2end.yaml)。该配置包含以下指标：

| 类别 | 指标 |
| --- | --- |
| 文本 | Edit Distance |
| 公式 | Edit Distance、CDM |
| 表格 | TEDS、Edit Distance |
| 阅读顺序 | Edit Distance |

匹配方式为 `quick_match`；matching、CDM、TEDS 的 worker 数均为 `16`；`quick_match_truncated_timeout_sec` 和 `match_timeout_sec` 均为 `1200`；`timeout_fallback_max_chunk_span=200`，`timeout_fallback_order_penalty=0.05`。复现时保留这些设置。配置中的默认数据路径是示例路径，下面的命令在临时副本中替换为实际挂载路径。

将 `GT` 设置为标注 JSON，`PRED_DIR` 设置为待评测的 Markdown 目录。

```bash
GT=/absolute/path/to/ground_truth.json
PRED_DIR="$OUTPUT_DIR/e2e"  # 两阶段输出改为 "$OUTPUT_DIR/stage2"
RESULT_DIR="$PWD/evaluation_results"
mkdir -p "$RESULT_DIR"
docker pull ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204

docker run --rm --entrypoint bash \
  -v "$PWD/evaluation_pipeline/configs/end2end.yaml:/repro/end2end.yaml:ro" \
  -v "$GT:/repro/gt.json:ro" \
  -v "$PRED_DIR:/repro/predictions:ro" \
  -v "$RESULT_DIR:/workspace/result" \
  -w /workspace \
  ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204 -c '
set -euo pipefail
python - <<"PY_CONFIG"
import yaml
with open("/repro/end2end.yaml") as f:
    config = yaml.safe_load(f)
dataset = config["end2end_eval"]["dataset"]
dataset["ground_truth"]["data_path"] = "/repro/gt.json"
dataset["prediction"]["data_path"] = "/repro/predictions"
with open("/tmp/xiaomi-ocr-eval.yaml", "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY_CONFIG
python pdf_validation.py --config /tmp/xiaomi-ocr-eval.yaml
'
```

## Real5-OmniDocBench：复现结果参考

| 方法 | 参数量 | Overall ↑ | 扫描 ↑ | 弯曲 ↑ | 屏幕拍摄 ↑ | 光照 ↑ | 倾斜 ↑ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| e2e | 0.8B | 93.45 | 94.524 | 93.401 | 93.279 | 93.858 | 92.170 |
| 两阶段 | 0.8B | 95.24 | 96.41 | 95.46 | 94.47 | 95.39 | 94.46 |

Overall 为五个子集成绩的算术平均。两阶段推理使用 PaddleX 版 PP-DocLayoutV3，阈值为 0.25。

## OmniDocBench v1.6：复现结果参考

| Method | overall↑ | text_block_Edit_dist↓ | display_formula_CDM↑ | display_formula_Edit_dist↓ | table_TEDS↑ | table_TEDS_structure_only↑ | table_Edit_dist↓ | reading_order_Edit_dist↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| e2e | 94.629 | 0.0395 | 96.325 | 0.0988 | 91.5119 | 94.0208 | 0.0795 | 0.1364 |
| 两阶段 | 96.4739 | 0.0332 | 98.3444 | 0.079 | 94.3973 | 96.6835 | 0.0544 | 0.1221 |

## Wild-OmniDocBench：复现结果参考

| Method | overall↑ | text_block_Edit_dist↓ | display_formula_CDM↑ | display_formula_Edit_dist↓ | table_TEDS↑ | table_TEDS_structure_only↑ | table_Edit_dist↓ | reading_order_Edit_dist↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| e2e | 83.9104 | 0.1379 | 87.7774 | 0.1691 | 77.7438 | 81.8345 | 0.1928 | 0.2164 |
| 两阶段 | 83.9477 | 0.1262 | 78.983 | 0.1628 | 85.48 | 89.8564 | 0.1209 | 0.1967 |
