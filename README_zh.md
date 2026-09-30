<div align="center">
<h1 align="center">Xiaomi-OCR-0</h1>
<p><b>0.8B 模型，统一文档解析与 OCR 相关理解。</b></p>
<p><a href="README.md">English</a> · <b>简体中文</b></p>
<p>
  <a href="https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0"><img src="https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face Model"></a>
  <a href="https://arxiv.org/abs/2609.36136"><img src="https://img.shields.io/badge/arXiv-2609.36136-b31b1b.svg" alt="arXiv:2609.36136"></a>
  <a href="https://huggingface.co/spaces/SeerRay-Lab/Xiaomi-OCR-0"><img src="https://img.shields.io/badge/Project-Page-FF6900?logo=huggingface&amp;logoColor=FFD21E" alt="Project Page"></a>
  <a href="https://github.com/SeerRay-Lab/Xiaomi-OCR-0"><img src="https://img.shields.io/badge/GitHub-SeerRay--Lab%2FXiaomi--OCR--0-181717?logo=github" alt="GitHub"></a>
</p>
</div>

Xiaomi-OCR-0 是一个统一的 0.8B 视觉语言模型，面向文档解析与 OCR 中心理解。模型以 Qwen3.5-0.8B 为起点，在约 1.7 亿条 OCR 中心样本上训练，采用 Q-Mask 文本锚定、继续预训练（CPT）和混合任务强化学习（Mix-RL）。

## 获取代码

```bash
git clone https://github.com/SeerRay-Lab/Xiaomi-OCR-0.git
cd Xiaomi-OCR-0
```

## 能力

- **文档解析：** 识别文档页面并输出结构化 Markdown，模型输出的 OTSL 表格由后处理转换为 Markdown 中的 HTML 表格，公式使用 LaTeX。
- **关键信息抽取（KIE）：** 按指定字段提取 JSON。
- **OCR 视觉问答（VQA）：** 回答文档图像中的文字与内容问题。
- **本地 Agent 工具：** MCP 服务负责提示词、PDF/图像预处理、可选区域检测、裁剪并发推理和结果组装。
- **浏览器 Demo：** 通过本地网页体验整页和区域解析。

印刷、版式规整的文档适合使用区域解析，可以检测区域并发处理。场景文字、手写笔记、书法、古籍和版式不规则的材料建议使用整页解析，避免区域边界切错后丢失上下文。

## 在 Agent 中体验

> ⚠️ **注意：** 经你授权后，这个 Skill 可能会创建 Python 环境、安装依赖、下载模型权重、安装并配置 SGLang 或 vLLM，以及注册 MCP 服务。模型和版面分析权重可能较大。如果你不同意这种安装流程，请不要执行下面的命令。

```
Read and execute https://raw.githubusercontent.com/SeerRay-Lab/Xiaomi-OCR-0/main/SKILL.md
```

模型在你的机器上运行。

## 运行浏览器 Demo

不安装 Agent Skill 或 MCP 服务也能运行网页 Demo。Demo 会向本机的 SGLang 或 vLLM 服务 `http://127.0.0.1:8000/v1` 发送请求。
需要 Python 3.10 或更高版本。

### 1. 安装推理运行环境

如果你已经有兼容的本地模型服务，可以直接跳到第 2 步。否则，选择 **一种**运行环境，并按你的操作系统、GPU 和驱动版本查看安装指南。运行环境有硬件要求，官方指南列出了支持的平台和安装方式：

- [SGLang 安装指南](https://docs.sglang.ai/get_started/install.html) · [支持的模型](https://docs.sglang.io/docs/supported-models/multimodal_language_models.md)
- [vLLM GPU 安装指南](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) · [Qwen3.5 模型支持](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/)

选择支持 `Qwen3_5ForConditionalGeneration` 架构的版本。macOS 和非 NVIDIA 设备请先查看运行环境的平台支持说明；下面的默认启动命令适用于兼容的 GPU 环境。

### 2. 启动 Xiaomi-OCR-0

在终端中运行下面 **其中一条**命令。如果本地缓存中没有模型，首次启动时会从 Hugging Face 下载权重。使用 Demo 时请保持这个终端运行。

SGLang：

```bash
python -m sglang.launch_server --model-path SeerRay-Lab/Xiaomi-OCR-0 \
  --host 127.0.0.1 --port 8000 --context-length 16384
```

vLLM：

```bash
vllm serve SeerRay-Lab/Xiaomi-OCR-0 \
  --host 127.0.0.1 --port 8000 --max-model-len 16384
```

等待模型加载完成后再继续。

### 3. 启动浏览器 Demo

在仓库目录中另开一个终端，运行：

```bash
python3 -m pip install -r requirements.txt
python3 demo/server.py --port 8787
```

打开 <http://127.0.0.1:8787>。同一网页支持整页/区域文档解析、KIE、VQA 和 PDF 解析。上述 `requirements.txt` 已包含图像、PDF 和公共后处理所需依赖。区域模式可选依赖 PaddlePaddle/PaddleX 与 PP-DocLayoutV3 权重。依赖说明见 [demo/README.md](demo/README.md)，MCP 安装说明见 [INSTALL.md](INSTALL.md)。

## 仓库内容

| 路径 | 内容 |
|:--|:--|
| [`skills/xiaomi-ocr/`](skills/xiaomi-ocr/) | Agent Skill 与 MCP 服务，支持 OCR、PDF 解析、KIE 和 VQA |
| [`demo/`](demo/) | 本地浏览器 Demo 和样例 |
| [`example_pics/`](example_pics/) | 与 Hugging Face 模型仓库同步的示例输入、参考 Markdown 和演示动画 |
| [`pipeline/`](pipeline/) | 整页与区域批量推理 |
| [`postprocess/`](postprocess/) | 表格转换、公式/文本组装与重复内容处理 |
| [Hugging Face model](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) | 模型卡片、示例和基准图表 |
| [Hugging Face Space](https://huggingface.co/spaces/SeerRay-Lab/Xiaomi-OCR-0/tree/main) | Space 资源和项目页源码 |
| [`examples/`](examples/) | MCP 配置示例 |

## 关键性能

以下是部分对比结果。箭头表示指标方向；粗体标记 Xiaomi-OCR-0，但不一定代表该列最优。基准说明和更多结果见 [Hugging Face 模型卡片](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0)。

![OmniDocBench v1.6 参数效率](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0/resolve/main/assets/omni-parameters.png)

![OCR 导向 VQA 参数效率](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0/resolve/main/assets/vqa-parameters.png)

![Real5-OmniDocBench 参数效率](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0/resolve/main/assets/real5-parameters.png)

![Wild-OmniDocBench 参数效率](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0/resolve/main/assets/wild-parameters.png)

### 文档解析：部分模型对比

| 模型 | 参数量 | OmniDocBench v1.6 ↑ | Real5 ↑ | Wild ↑ |
|:--|--:|--:|--:|--:|
| **Xiaomi-OCR-0** | **0.8B** | **96.83** | **95.24** | **87.94** |
| TeleOCR | 1.2B | 96.87 | — | 88.53 |
| OvisOCR2 | 0.8B | 96.58 | 92.29 | 87.91 |
| PaddleOCR-VL-1.6 | 0.9B | 96.33 | 93.19 | 87.36 |
| MinerU2.5-Pro | 1.2B | 95.75 | 88.94 | 87.33 |
| GLM-OCR | 0.9B | 95.22 | 90.32 | 85.08 |

三列均为 Overall 成绩。

### OCR 中心视觉问答

| 模型 | 参数量 | DocVQA | InfoVQA | ChartQA | OCRBench | TextVQA | 平均分 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **Xiaomi-OCR-0** | **0.8B** | **93.1** | **75.1** | **84.6** | **84.6** | **78.6** | **83.2** |
| Qwen3.5-0.8B | 0.8B | 88.5 | 60.3 | 69.5 | 77.9 | 68.3 | 72.9 |
| Qwen3.5-2B | 2B | 92.4 | 72.4 | 77.0 | 85.9 | 76.9 | 80.9 |
| Qwen3.5-4B | 4B | 94.4 | 80.4 | 82.4 | 86.6 | 80.8 | 84.9 |
| MiniCPM-V-4.5 | 8B | 84.9 | 69.6 | 87.4 | 89.0 | 82.2 | 82.6 |

平均分为这五项基准在 0–100 分尺度上的算术平均。

批量推理见 [pipeline/README.md](pipeline/README.md)，公共前后处理见 [postprocess/README.md](postprocess/README.md)。

## 引用

如果你关注了我们的工作，请引用以下内容：

```bibtex
@misc{chen2026xiaomiocr0technicalreport,
      title={Xiaomi-OCR-0 Technical Report},
      author={Xin Chen and Anan Du and Feng Feng and Pei Fu and Jian Luan and Longwei Xu and Shaojie Zhang and Hang Li and Heng Qu and Cheng Tan},
      year={2026},
      eprint={2609.36136},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.36136},
}
```
