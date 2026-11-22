# Paper Library

按研究主题整理的论文、技术博客和阅读笔记。PDF、离线网页与笔记保存在对应目录，网页正文中保留原始来源链接。

## 分类目录

| 目录 | 内容 |
| --- | --- |
| [Robotics-and-Embodied-AI](Robotics-and-Embodied-AI/) | 世界动作模型、VLA、机器人部署、评测与控制 |
| [Generative-Models](Generative-Models/) | 扩散模型、图像与视频生成、多模态、世界模型、生成模型训练 |
| [LLM-Systems](LLM-Systems/) | 模型架构、推测解码、Attention、KV Cache、推理服务与性能分析 |
| [ML-Systems-and-Hardware](ML-Systems-and-Hardware/) | 分布式训练、GPU 编程、编译器与硬件性能 |
| [Machine-Learning-Foundations](Machine-Learning-Foundations/) | 优化、量化、蒸馏、强化学习与概率采样 |
| [Natural-Language-Processing](Natural-Language-Processing/) | 文本编码、检索、分类、抽取、推理与语言基准 |
| [Computer-Vision](Computer-Vision/) | 目标检测、分割、视觉语言模型与高效视觉架构 |
| [Audio-and-Speech](Audio-and-Speech/) | 语音识别、对齐、音乐与音频模型 |
| [Computer-Graphics](Computer-Graphics/) | 渲染、路径追踪与神经图形学 |
| [Computer-Science-Foundations](Computer-Science-Foundations/) | 算法、数据结构、数据库、并发与数值计算 |
| [Explainable-and-Responsible-AI](Explainable-and-Responsible-AI/) | 可解释性、公平性、可靠性与安全评测 |
| [ML-Security-and-Robustness](ML-Security-and-Robustness/) | 后门、水印与模型压缩鲁棒性 |
| [Medical-AI](Medical-AI/) | 医学图像与临床语言理解 |
| [Recommender-Systems](Recommender-Systems/) | 推荐模型与生成式推荐 |
| [Research-Landscapes](Research-Landscapes/) | 团队、研究机构与信息源调研 |
| [skills](skills/) | AI 开发工作流、科研报告写作与改写技能 |

## 常用资料目录

| 主题 | 入口 |
| --- | --- |
| 扩散推理：缓存、并行、卸载与量化 | [扩散论文](Generative-Models/Diffusion/) · [社区文章与方法文档](Generative-Models/Diffusion/Community-Articles-20261122/) |
| LLM 服务与 KV Cache | [论文与笔记](LLM-Systems/Serving-and-KV-Cache/) · [社区文章](LLM-Systems/Serving-and-KV-Cache/Community-Articles-20261122/) |
| Attention 与推理内核 | [论文](LLM-Systems/Attention-and-Kernels/) · [社区文章](LLM-Systems/Attention-and-Kernels/Community-Articles-20261122/) |
| GPU 编程与 FlashAttention | [论文与教材](ML-Systems-and-Hardware/GPU-Programming/) · [技术文章](ML-Systems-and-Hardware/GPU-Programming/Community-Articles-20261122/) |
| 推理性能分析 | [笔记与论文](LLM-Systems/Performance-and-Profiling/) |
| 分布式训练 | [论文与演示资料](ML-Systems-and-Hardware/Distributed-Training/) |
| Qwen 架构与官方文章 | [模型架构](LLM-Systems/Model-Architecture/) · [官方文章](LLM-Systems/Model-Architecture/Qwen-Official-Articles/) |
| VLA 与世界动作模型 | [VLA 部署](Robotics-and-Embodied-AI/VLA-Inference/) · [世界动作模型](Robotics-and-Embodied-AI/World-Action-Models/) |
| 目标检测与分割 | [目标检测](Computer-Vision/Object-Detection/) · [图像与视频分割](Computer-Vision/Image-and-Video-Segmentation/) |

## 扩散推理先读这五篇

1. [SGLang-Diffusion: Advanced Optimizations](Generative-Models/Diffusion/Community-Articles-20261122/lmsys/2026-02-16-sglang-diffusion-advanced-optimizations.html)
2. [xDiT 论文](Generative-Models/Diffusion/2411.01738v1.pdf)
3. [PipeFusion 论文](Generative-Models/Diffusion/2405.14430v4.pdf)
4. [TensorRT-LLM: Disaggregated Serving](LLM-Systems/Serving-and-KV-Cache/Community-Articles-20261122/tensorrt-llm/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
5. [NVIDIA Dynamo 介绍](LLM-Systems/Serving-and-KV-Cache/Community-Articles-20261122/nvidia/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models.html)

离线 HTML 可直接用浏览器打开；视频和交互图表通过文中的原始链接查看。`MinerU_markdown_*` 是 PDF 全文提取，与对应论文放在一起。
