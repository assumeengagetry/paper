# Qwen 模型谱系相关论文与官方技术资料

核查／归档日期：2026-11-19。用户原调研的截止日为2026-11-17；本次只核对资料身份、来源和文件，不对调研全文的人员职务或所有技术推断背书。

**15份相关PDF：新增11份，复用4份。新增PDF约76.46 MiB、291页；另转存13份官方技术博客正文HTML。全库PDF从565增加至576，原有565份均保持不变。**

- [当前完整PDF目录](CATALOG.md)
- [来源、版本、页数、SHA-256与验证机器清单](QWEN-Papers-20261119.manifest.json)

## 论文／独立技术报告

下表覆盖所提供的14个不同arXiv报告链接，并补关联精读路线明确提到的Gated Delta Networks原论文。已有版本复用，不重新下载同一篇、覆盖或升级旧文件；新下载采用本次核对的官方最新revision。

| 论文／技术报告 | 本地PDF | 官方版本 | 页数 | 状态 |
| --- | --- | --- | ---: | --- |
| Qwen3 Technical Report | [PDF](LLM-Systems/Model-Architecture/2505.09388v1.pdf) | [2505.09388v1](https://arxiv.org/abs/2505.09388v1) | 35 | 已有，复用 |
| Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models | [PDF](Natural-Language-Processing/Text-Embeddings-and-Encoders/2506.05176v3.pdf) | [2506.05176v3](https://arxiv.org/abs/2506.05176v3) | 14 | 新增 |
| Qwen-Image Technical Report | [PDF](Generative-Models/Diffusion/2508.02324v1.pdf) | [2508.02324v1](https://arxiv.org/abs/2508.02324v1) | 46 | 新增 |
| Qwen3-Omni Technical Report | [PDF](LLM-Systems/Model-Architecture/2509.17765v1.pdf) | [2509.17765v1](https://arxiv.org/abs/2509.17765v1) | 25 | 已有，复用 |
| Qwen3Guard Technical Report | [PDF](Explainable-and-Responsible-AI/Safety-and-Red-Teaming/2510.14276v1.pdf) | [2510.14276v1](https://arxiv.org/abs/2510.14276v1) | 28 | 新增 |
| Qwen3-VL Technical Report | [PDF](Computer-Vision/Vision-Language-Pretraining/2511.21631v2.pdf) | [2511.21631v2](https://arxiv.org/abs/2511.21631v2) | 42 | 已有，复用 |
| Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking | [PDF](Computer-Vision/Vision-Language-Pretraining/2601.04720v2.pdf) | [2601.04720v2](https://arxiv.org/abs/2601.04720v2) | 23 | 新增 |
| Qwen3-TTS Technical Report | [PDF](Generative-Models/Speech-and-Audio/2601.15621v1.pdf) | [2601.15621v1](https://arxiv.org/abs/2601.15621v1) | 14 | 新增 |
| Qwen3-ASR Technical Report | [PDF](Audio-and-Speech/Automatic-Speech-Recognition/2601.21337v2.pdf) | [2601.21337v2](https://arxiv.org/abs/2601.21337v2) | 16 | 新增 |
| Qwen3-Coder-Next Technical Report | [PDF](Generative-Models/Language-Model-Training-and-Alignment/2603.00729v1.pdf) | [2603.00729v1](https://arxiv.org/abs/2603.00729v1) | 23 | 新增 |
| Qwen3.5-Omni Technical Report | [PDF](LLM-Systems/Model-Architecture/2604.15804v2.pdf) | [2604.15804v2](https://arxiv.org/abs/2604.15804v2) | 28 | 新增 |
| Qwen-AgentWorld: Language World Models for General Agents | [PDF](Generative-Models/Language-Model-Training-and-Alignment/2606.24597v1.pdf) | [2606.24597v1](https://arxiv.org/abs/2606.24597v1) | 47 | 新增 |
| On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability | [PDF](LLM-Systems/Model-Architecture/2608.30320v1.pdf) | [2608.30320v1](https://arxiv.org/abs/2608.30320v1) | 28 | 新增 |
| Qwen3.8-Omni: Towards Native Omni-Modal Agents | [PDF](LLM-Systems/Model-Architecture/2609.25611v1.pdf) | [2609.25611v1](https://arxiv.org/abs/2609.25611v1) | 24 | 新增 |
| Gated Delta Networks: Improving Mamba2 with Delta Rule | [PDF](LLM-Systems/Model-Architecture/2412.06464v3.pdf) | [2412.06464v3](https://arxiv.org/abs/2412.06464v3) | 22 | 已有，复用 |

## 官方技术长文／发布资料（HTML）

以下是真实官方技术博客的静态格式转存，**不是独立论文PDF，也不是原网站逐字节离线包**。已去除登录导航、跟踪与执行脚本；媒体保留官方外链，懒加载代码／图表和互动示例可能仍需回原网页查看。来源原文中的命令没有执行。

| 主题 | 官方标题 | 本地静态转存 | 原始网页 |
| --- | --- | --- | --- |
| qwen3 | Qwen3: Think Deeper, Act Faster | [HTML](LLM-Systems/Model-Architecture/Qwen-Official-Articles/qwen3.html) | [官方来源](https://qwen.ai/blog?id=qwen3) |
| qwen3-next | Qwen3-Next: Towards Ultimate Training &amp; Inference Efficiency | [HTML](LLM-Systems/Model-Architecture/Qwen-Official-Articles/qwen3-next.html) | [官方来源](https://qwen.ai/blog?id=qwen3-next) |
| qwen3-coder | Qwen3-Coder: Agentic Coding in the World | [HTML](Generative-Models/Language-Model-Training-and-Alignment/Qwen-Official-Articles/qwen3-coder.html) | [官方来源](https://qwen.ai/blog?id=qwen3-coder) |
| qwen3-max | Qwen3-Max: Just Scale it | [HTML](LLM-Systems/Model-Architecture/Qwen-Official-Articles/qwen3-max.html) | [官方来源](https://qwen.ai/blog?id=qwen3-max) |
| qwen3.5 | Qwen3.5: Towards Native Multimodal Agents | [HTML](LLM-Systems/Model-Architecture/Qwen-Official-Articles/qwen3.5.html) | [官方来源](https://qwen.ai/blog?id=qwen3.5) |
| qwen3.6-27b | Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model | [HTML](LLM-Systems/Model-Architecture/Qwen-Official-Articles/qwen3.6-27b.html) | [官方来源](https://qwen.ai/blog?id=qwen3.6-27b) |
| qwen3.7 | Qwen3.7: The Agent Frontier | [HTML](LLM-Systems/Model-Architecture/Qwen-Official-Articles/qwen3.7.html) | [官方来源](https://qwen.ai/blog?id=qwen3.7) |
| qwen3.8 | Qwen3.8-Max: A New Bar for Coding and Cowork | [HTML](LLM-Systems/Model-Architecture/Qwen-Official-Articles/qwen3.8.html) | [官方来源](https://qwen.ai/blog?id=qwen3.8) |
| qwen3.8-flash-next | Qwen3.8-Flash-Next: A New Architecture, Towards Ultimate Cost-Efficiency | [HTML](LLM-Systems/Model-Architecture/Qwen-Official-Articles/qwen3.8-flash-next.html) | [官方来源](https://qwen.ai/blog?id=qwen3.8-flash-next) |
| qwen3-coder-next | Qwen3-Coder-Next: Pushing Small Hybrid Models on Agentic Coding | [HTML](Generative-Models/Language-Model-Training-and-Alignment/Qwen-Official-Articles/qwen3-coder-next.html) | [官方来源](https://qwen.ai/blog?id=qwen3-coder-next) |
| qwen-image-2.0 | Qwen-Image-2.0: Professional infographics, exquisite photorealism | [HTML](Generative-Models/Diffusion/Qwen-Official-Articles/qwen-image-2.0.html) | [官方来源](https://qwen.ai/blog?id=qwen-image-2.0) |
| qwen-image-2.1 | Qwen-Image-2.1: Compact, Efficient, and Unified Image Creation | [HTML](Generative-Models/Diffusion/Qwen-Official-Articles/qwen-image-2.1.html) | [官方来源](https://qwen.ai/blog?id=qwen-image-2.1) |
| qwen4-max | Qwen4-Max: Your New Teammate Has Arrived | [HTML](LLM-Systems/Model-Architecture/Qwen-Official-Articles/qwen4-max.html) | [官方来源](https://qwen.ai/blog?id=qwen4-max) |

## 文献身份与未核实的独立PDF

- Qwen3.5-Omni不是普通Qwen3.5的技术报告；普通Qwen3.8也不能拿Flash-Next或3.8-Omni报告替代。
- 已转存Qwen4-Max官方发布文章；本次检查来源未建立独立完整Qwen4技术报告的PDF身份，未伪造或用Flash-Next改名替代。
- 普通Qwen3.5／3.6／3.7／3.8、初代Coder、Qwen3-Next／Max、Qwen3-2507和Image-2.0／2.1，在本次提供并核对的来源中未确认独立同名PDF；有官方文章的已转存。这个限定不是断言它们在所有来源都绝对不存在论文。
- AgentWorld正文研究文本环境状态转移、CPT→SFT→RL和agent训练，不是视频—动作机器人世界模型，归语言模型训练；Coder-Next按agentic训练贡献同样归训练主题。
- 文本Embedding归NLP表征；VL-Embedding归视觉语言表征；Guard归内容安全；Image归扩散图像生成；ASR单设音频识别子域，避免误归TTS。Omni架构及Flash-Next沿用现有Qwen架构报告入口。

## 实际核验

- 11份新增PDF均通过`%PDF-`签名、`pdfinfo`页数、标题与自身arXiv版本标记核对；`qpdf --check`为11份清洁通过、0警告、0致命错误。
- 已检查每份新增PDF的标题、摘要、引言与实际结论，44条章节短引文逐项在源文本对应行核对。未复现论文实验。
- 归档时原有565份PDF及其他既有资料校验和不变；HTML和PDF分别记录SHA-256。各批来源manifest与验证记录保留原批次计数，当前完整目录统一见 `CATALOG.md`。
- 此批新增PDF共约76.46 MiB，最大单份约41.16 MiB，均未达到100 MiB；本次提交不会重复上传原Zotero ZIP。
- 不保存输入链接的email_hash、utm_source等跟踪参数；不下载模型权重、克隆执行代码或新增用户报告中未提供的模型谱系图片。

资料版权归作者／发布方；静态格式转存不表示版权转让或独立发表。
