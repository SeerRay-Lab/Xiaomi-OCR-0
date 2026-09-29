# Xiaomi-OCR-0 · 代码仓库

这里提供本地 Agent Skill、MCP 服务、浏览器 Demo 和评测辅助脚本。模型权重与模型卡单独托管在 [Hugging Face](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0)。

## 获取代码

```bash
git clone https://github.com/SeerRay-Lab/Xiaomi-OCR-0.git
cd Xiaomi-OCR-0
```

## 在 Agent 中使用

> ⚠️ **注意：** 执行此 Skill 可能会让 Agent 新建 Python 环境、下载模型或版面分析权重、安装并配置 SGLang 或 vLLM，以及注册 MCP 服务。如果你不同意这些操作，请不要运行下面的命令。请先在 GitHub 查看具体安装步骤。

```text
Read and execute https://raw.githubusercontent.com/SeerRay-Lab/Xiaomi-OCR-0/main/SKILL.md
```

Skill 会先检查已有的运行环境和本地模型文件；安装推理运行时或下载权重前会征求选择与授权。安装步骤见 [INSTALL.md](INSTALL.md)，Agent 入口见 [SKILL.md](SKILL.md)。

## 运行浏览器 Demo

Demo 连接本机运行的 SGLang 或 vLLM 服务。如果已经有兼容的本地推理服务和 Xiaomi-OCR-0 权重，可以直接跳到第 2 步。

### 1. 启动本地模型服务

根据操作系统、GPU 和驱动选择一种运行时，并按其安装指南配置。版本需要支持 `Qwen3_5ForConditionalGeneration` 和图像输入。

- [SGLang 安装指南](https://docs.sglang.ai/get_started/install.html) · [SGLang 支持的模型](https://docs.sglang.io/docs/supported-models/multimodal_language_models.md)
- [vLLM 安装指南](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) · [Qwen3.5 支持情况](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/)

在本机回环地址 `127.0.0.1:8000` 启动模型：

```bash
sglang serve --model-path SeerRay-Lab/Xiaomi-OCR-0 \
  --host 127.0.0.1 --port 8000 --context-length 16384
```

或者

```bash
vllm serve SeerRay-Lab/Xiaomi-OCR-0 \
  --host 127.0.0.1 --port 8000 --max-model-len 16384
```

首次启动时，如果权重尚未缓存，运行时会从 Hugging Face 下载。使用 Demo 时请保持模型服务运行。

### 2. 启动 Demo

在另一个终端进入仓库目录并运行：

```bash
python3 -m pip install Pillow
python3 demo/server.py --port 8787
```

浏览器打开 <http://127.0.0.1:8787>。解析 PDF 还需要安装 `pypdfium2`：

```bash
python3 -m pip install pypdfium2
```

网页 Demo 支持文档解析、KIE、VQA、图片上传和 PDF 解析。Region 模式还需要 PaddlePaddle/PaddleX 和 PP-DocLayoutV3，安装说明见 [demo/README.md](demo/README.md)。规则的印刷体文档适合 Region 模式，可并发处理检测到的版面区域；场景文字、手写、书法、古籍和不规则版面建议用整页模式，避免区域边界切分错误造成上下文丢失。

## 仓库内容

| 路径 | 用途 |
|:--|:--|
| [`SKILL.md`](SKILL.md) | Agent 安装入口 |
| [`skills/xiaomi-ocr/`](skills/xiaomi-ocr/) | OCR、PDF、KIE、VQA 的 MCP 工具及共用前后处理 |
| [`demo/`](demo/) | 本地浏览器 Demo 和回放样例 |
| [`example_pics/`](example_pics/) | 示例输入图片 |
| [`workflows/evaluate_cases.py`](workflows/evaluate_cases.py) | 使用本地推理服务运行评测案例 |
| [`INSTALL.md`](INSTALL.md) | 本地运行时、MCP 和可选版面分析环境的安装说明 |

MCP 服务统一负责图像/PDF 预处理、提示词、可选版面检测、区域并发推理和结果组装。浏览器 Demo 复用同一套处理逻辑。推理运行在用户本机；此仓库不提供托管 OCR API。
