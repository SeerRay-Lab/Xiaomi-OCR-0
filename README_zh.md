<div align="center">
<h1 align="center">Xiaomi-OCR-0</h1>
<p><b>0.8B 模型，统一文档解析与 OCR 相关理解。</b></p>
<p><a href="README.md">English</a> · <b>简体中文</b></p>
<p>
  <a href="https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0"><img src="https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face Model"></a>
  <a href="https://arxiv.org/abs/2609.36136"><img src="https://img.shields.io/badge/arXiv-2609.36136-b31b1b.svg" alt="arXiv:2609.36136"></a>
  <a href="https://huggingface.co/spaces/SeerRay-Lab/Xiaomi-OCR-0"><img src="https://img.shields.io/badge/Project-Page-FF6900?logo=huggingface&amp;logoColor=FFD21E" alt="Project Page"></a>
  <a href="https://github.com/SeerRay-Lab/Xiaomi-OCR-0"><img src="https://img.shields.io/badge/GitHub-SeerRay--Lab%2FXiaomi--OCR--0-181717?logo=github" alt="GitHub"></a>
  <a href="https://ollama.com/longwayxu/xiaomi-ocr-0:bf16"><img src="https://img.shields.io/badge/Ollama-BF16%20%28experimental%29-000000?logo=ollama&amp;logoColor=white" alt="Ollama BF16 (experimental)"></a>
  <a href="https://www.modelscope.cn/models/SeerRay-Lab/Xiaomi-OCR-0"><img src="https://img.shields.io/badge/ModelScope-Model-624AFF?logo=ModelScope&amp;logoColor=white" alt="ModelScope Model"></a>
</p>
</div>

Xiaomi-OCR-0 是一个统一的 0.8B 视觉语言模型，面向文档解析与 OCR 中心理解。模型以 Qwen3.5-0.8B 为起点，在约 1.7 亿条 OCR 中心样本上训练，采用 Q-Mask 文本锚定、继续预训练（CPT）和混合任务强化学习（Mix-RL）。
## 🏆 Benchmark 成绩与排名

- 🥇 Real5-OmniDocBench 得分 **95.24**，排名 **第 1**。
- 🥈 OmniDocBench v1.6 得分 **96.83**，Wild-OmniDocBench 得分 **87.94**，两项均排名 **第 2**。
- 📐 UniMER-Test (公式识别) 得分 **97.84**，PubTabNet（表格识别） 得分 **92.27** 。
- ✍️ Chronicles-OCR Mature Scripts（隶书、草书、行书、楷书） 得分 **0.73**。
- 🧠 五项 OCR 中心 VQA 基准平均分 **83.2**，以 **0.8B** 参数超过 Qwen3.5-2B(**80.9)**。

## 📰 更新日志

- **2026-10-08：** 感谢社区对 Xiaomi-OCR-0 的支持与贡献：[prithivMLmods 的 GGUF 版本](https://huggingface.co/prithivMLmods/Xiaomi-OCR-0-GGUF)、[xzl01 的 GGUF 版本](https://github.com/xzl01/Xiaomi-OCR-0-GGUF)，以及 [ByronLeeee 的 BF16 NInfer 转换版](https://huggingface.co/ByronLeeee/Xiaomi-OCR-0-Ninfer)。

## 能力

- **文档解析：** 识别文档页面并输出结构化 Markdown，模型输出的 OTSL 表格由后处理转换为 Markdown 中的 HTML 表格，公式使用 LaTeX。
- **关键信息抽取（KIE）：** 按指定字段提取 JSON。
- **OCR 视觉问答（VQA）：** 回答文档图像中的文字与内容问题。
- **本地 Agent 工具：** MCP 服务负责提示词、PDF/图像预处理、可选区域检测、裁剪并发推理和结果组装。
- **浏览器 Demo：** 通过本地网页体验整页和区域解析。

印刷、版式规整的文档适合使用区域解析，可以检测区域并发处理。场景文字、手写笔记、书法、古籍和版式不规则的材料建议使用整页解析，避免区域边界切错后丢失上下文。

## 仓库内容

| 路径 | 内容 |
|:--|:--|
| [`skills/xiaomi-ocr/`](skills/xiaomi-ocr/) | Agent Skill 与 MCP 服务，支持 OCR、PDF 解析、KIE 和 VQA |
| [`demo/`](demo/) | 本地浏览器 Demo 和样例 |
| [`example_pics/`](example_pics/) | 与 Hugging Face 模型仓库同步的示例输入、参考 Markdown 和演示动画 |
| [`pipeline/`](pipeline/) | 整页与区域批量推理 |
| [`postprocess/`](postprocess/) | 表格转换、公式/文本组装与重复内容处理 |
| [Hugging Face model](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) | huggingface模型卡片、示例和基准图表
| [Model Scope model](https://www.modelscope.cn/models/SeerRay-Lab/Xiaomi-OCR-0) | modelscope模型卡片、示例和基准图表|
| [Hugging Face Space](https://huggingface.co/spaces/SeerRay-Lab/Xiaomi-OCR-0/tree/main) | Space 资源和项目页源码 |
| [`examples/`](examples/) | MCP 配置示例 |

## 安装

选择以下两种安装方式之一：

- [在 Agent 中体验](#在-agent-中体验)：由 Agent 安装本地模型并配置 MCP 服务。
- [手动安装](#手动安装)：自行获取代码、选择推理运行环境并启动浏览器 Demo。

### 在 Agent 中体验

> ⚠️ **注意：** 经你授权后，这个 Skill 可能会创建 Python 环境、安装依赖、下载模型权重、安装并配置所选推理运行环境（SGLang、vLLM 或 Ollama），以及注册 MCP 服务。模型和版面分析权重可能较大。如果你不同意这种安装流程，请不要执行下面的命令。

复制下面这句话发送给你的 Agent：

```
Read and execute https://raw.githubusercontent.com/SeerRay-Lab/Xiaomi-OCR-0/main/SKILL.md
```

模型将在你的机器上运行。完整安装时，Agent 会在你授权的安装范围内，一并安装区域识别所需的 PaddlePaddle、PaddleX 和 PP-DocLayoutV3，并验证区域解析。你也可以指定只安装整页模式；如果当前平台不支持区域依赖，Agent 会说明限制。

### 手动安装

需要 Python 3.10 或更高版本。先获取代码，再选择 **一种**推理运行环境：[SGLang / vLLM](#sglang--vllm) 或 [Ollama](#ollama实验性)。如果需要区域解析，还需安装[可选的版面分析依赖](#可选区域模式依赖)。

手动配置 MCP 服务见 [INSTALL.md](INSTALL.md)，Ollama 专用配置见 [OLLAMA.md](OLLAMA.md)。

#### 获取代码

```bash
git clone https://github.com/SeerRay-Lab/Xiaomi-OCR-0.git
cd Xiaomi-OCR-0
```

#### SGLang / vLLM

不安装 Agent Skill 或 MCP 服务也能运行网页 Demo。Demo 会向本机的 SGLang 或 vLLM 服务 `http://127.0.0.1:8000/v1` 发送请求。

##### 1. 安装推理运行环境

如果上述地址已经运行兼容的本地模型服务，可以直接跳到第 3 步。否则，选择 **一种**运行环境，并按你的操作系统、GPU 和驱动版本查看安装指南。运行环境有硬件要求，官方指南列出了支持的平台和安装方式：

- [SGLang 安装指南](https://docs.sglang.ai/get_started/install.html) · [支持的模型](https://docs.sglang.io/docs/supported-models/multimodal_language_models.md)
- [vLLM GPU 安装指南](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) · [Qwen3.5 模型支持](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen3_5/)

选择支持 `Qwen3_5ForConditionalGeneration` 架构的版本。macOS 和非 NVIDIA 设备请先查看运行环境的平台支持说明；下面的默认启动命令适用于兼容的 GPU 环境。

##### 2. 启动 Xiaomi-OCR-0

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

##### 3. 启动浏览器 Demo

在仓库目录中另开一个终端，运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo/server.py --port 8787
```

打开 <http://127.0.0.1:8787>。界面支持中英文切换，右上角可选择语言；英文直达链接为 <http://127.0.0.1:8787/?lang=en>。同一网页支持整页/区域文档解析、KIE、VQA 和 PDF 解析。上述 `requirements.txt` 已包含图像、PDF 和公共后处理所需依赖。区域模式可选依赖 PaddlePaddle/PaddleX 与 PP-DocLayoutV3 权重。依赖说明见 [demo/README.md](demo/README.md)，MCP 安装说明见 [INSTALL.md](INSTALL.md)。

#### Ollama（实验性）

[社区 BF16 GGUF 包](https://ollama.com/longwayxu/xiaomi-ocr-0:bf16)包含语言模型和视觉投影器，约 1.8 GB。安装 [Ollama](https://ollama.com/download) 并保持本地服务运行，然后拉取模型：

```bash
ollama pull longwayxu/xiaomi-ocr-0:bf16
```

进行 OCR 时，请使用本仓库的 **MCP 服务或浏览器 Demo**。它们会自动应用对齐检查所用的图像预处理、非思考提示词和解码设置；仅拉取模型不会配置这些步骤。使用这两个入口无需再写额外的包装脚本。

运行浏览器 Demo 时，在仓库根目录使用 Python 3.10 或更高版本执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export XIAOMI_OCR_BACKEND=ollama
export XIAOMI_OCR_MODEL=longwayxu/xiaomi-ocr-0:bf16
export XIAOMI_OCR_LOCAL_URL=http://127.0.0.1:11434/v1
python demo/server.py --port 8787
```

打开 <http://127.0.0.1:8787>。整页 OCR、PDF 解析、KIE 和 VQA 复用现有流程；区域模式还需安装下文的版面分析依赖。直接安装 MCP 和转换细节见 [Ollama 配置与验证说明](OLLAMA.md)。

#### 可选：区域模式依赖

区域识别额外需要 PaddlePaddle、PaddleX 和 PP-DocLayoutV3。请在运行 Demo 或 MCP 服务的 Python 环境中，先按 [PaddlePaddle 官方安装教程](https://paddlepaddle.github.io/PaddleX/latest/installation/paddlepaddle_install.html)选择适合操作系统、Python 和 CPU/GPU 的版本，再参考 [PaddleX 官方安装教程](https://paddlepaddle.github.io/PaddleX/latest/installation/installation.html)。在仓库根目录运行：

```bash
python -m pip install -r requirements-paddlex.txt
```

该清单包含 `paddlex[cv]` 和 `shapely`；PaddlePaddle 需按官方教程单独安装。首次使用时会下载尚未缓存的 PP-DocLayoutV3 权重。使用 CPU 做版面检测时，在 Demo 或 MCP 环境中设置 `XIAOMI_OCR_LAYOUT_DEVICE=cpu`。整页 OCR、KIE 和 VQA 无需这些额外依赖。

## 社区模型转换

感谢社区对xiaomi-ocr-0的支持, [prithivMLmods/Xiaomi-OCR-0-GGUF](https://huggingface.co/prithivMLmods/Xiaomi-OCR-0-GGUF) 和 [xzl01/Xiaomi-OCR-0-GGUF](https://github.com/xzl01/Xiaomi-OCR-0-GGUF) 提供了GGUF量化权重, [ByronLeeee/Xiaomi-OCR-0-Ninfer](https://huggingface.co/ByronLeeee/Xiaomi-OCR-0-Ninfer) 提供 BF16 NInfer 模型包。


## 性能对比

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
