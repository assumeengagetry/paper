# VLA 模型与机器人 Benchmark 论文下载清单

更新日期：2026-10-19。

## 下载范围与结果

按《VLA 模型 Benchmark 调研：π、GR00T 与 Hy-VLA》中明确讨论的模型和 benchmark 补齐对应论文，并补充 Language Table 官方引用的来源论文。

- **共覆盖 25 篇论文：新增下载 20 篇，复用原库已有 5 篇。**
- 新增 PDF 共 **269.93 MiB、466 页**；已有 5 篇共 **73.40 MiB、123 页**。
- 沿用现有主题分类，保留 arXiv 原文件名；已有文件不重命名、不覆盖、不删除。
- 模型论文优先采用调研引用的明确 revision；新下载的 benchmark 论文采用下载时可取得的最新 arXiv revision。原库已有论文保留原版本。
- 每份 PDF 的来源、版本、页数、大小、SHA-256 与校验状态见 [机器可读清单](VLA-Benchmark-Papers-20261019.manifest.json)。

## 1. 模型论文

| 条目 | 论文标题 | 本地 PDF | 来源版本 | 页数 | 状态 |
| --- | --- | --- | --- | ---: | --- |
| π₀ | π0: A Vision-Language-Action Flow Model for General Robot Control | [pi0.pdf](Vision-Language-Action-Models/pi0.pdf) | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)，既有官方 PDF 未标注具体 revision | 17 | 已有，保留 |
| FAST / π₀-FAST | FAST: Efficient Action Tokenization for Vision-Language-Action Models | [2501.09747v1.pdf](Vision-Language-Action-Models/2501.09747v1.pdf) | [2501.09747v1](https://arxiv.org/abs/2501.09747v1) | 19 | 已有，保留 |
| π₀.₅ | π0.5: a Vision-Language-Action Model with Open-World Generalization | [2504.16054v1.pdf](Vision-Language-Action-Models/2504.16054v1.pdf) | [2504.16054v1](https://arxiv.org/abs/2504.16054v1) | 19 | 已有，保留 |
| π₀.₅ + KI | Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better | [2505.23705v1.pdf](Vision-Language-Action-Models/2505.23705v1.pdf) | [2505.23705v1](https://arxiv.org/abs/2505.23705v1) | 18 | 新增 |
| π₀.₇ | π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities | [2604.15483v1.pdf](Vision-Language-Action-Models/2604.15483v1.pdf) | [2604.15483v1](https://arxiv.org/abs/2604.15483v1) | 25 | 新增 |
| GR00T N1 | GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | [2503.14734v2.pdf](Vision-Language-Action-Models/2503.14734v2.pdf) | [2503.14734v2](https://arxiv.org/abs/2503.14734v2) | 36 | 新增 |
| Hy-VLA | Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack | [2606.14409v2.pdf](Vision-Language-Action-Models/2606.14409v2.pdf) | [2606.14409v2](https://arxiv.org/abs/2606.14409v2) | 26 | 新增 |

## 2. Benchmark 与相关数据生成论文

| 条目 | 论文标题 | 本地 PDF | 来源版本 | 页数 | 状态 |
| --- | --- | --- | --- | ---: | --- |
| LIBERO | LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning | [2306.03310v2.pdf](Benchmarks-and-Datasets/2306.03310v2.pdf) | [2306.03310v2](https://arxiv.org/abs/2306.03310v2) | 44 | 已有，保留 |
| RoboTwin 2.0 | RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation | [2506.18088v2.pdf](Benchmarks-and-Datasets/2506.18088v2.pdf) | [2506.18088v2](https://arxiv.org/abs/2506.18088v2) | 24 | 已有，保留 |
| CALVIN | CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks | [2112.03227v4.pdf](Benchmarks-and-Datasets/2112.03227v4.pdf) | [2112.03227v4](https://arxiv.org/abs/2112.03227v4) | 10 | 新增 |
| RoboCasa | RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots | [2406.02523v1.pdf](Benchmarks-and-Datasets/2406.02523v1.pdf) | [2406.02523v1](https://arxiv.org/abs/2406.02523v1) | 16 | 新增 |
| SIMPLER / SimplerEnv | Evaluating Real-World Robot Manipulation Policies in Simulation | [2405.05941v1.pdf](Benchmarks-and-Datasets/2405.05941v1.pdf) | [2405.05941v1](https://arxiv.org/abs/2405.05941v1) | 22 | 新增 |
| RLBench | RLBench: The Robot Learning Benchmark & Learning Environment | [1909.12271v1.pdf](Benchmarks-and-Datasets/1909.12271v1.pdf) | [1909.12271v1](https://arxiv.org/abs/1909.12271v1) | 8 | 新增 |
| DexMimicGen | DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning | [2410.24185v2.pdf](Benchmarks-and-Datasets/2410.24185v2.pdf) | [2410.24185v2](https://arxiv.org/abs/2410.24185v2) | 10 | 新增 |
| VLABench | VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks | [2412.18194v1.pdf](Benchmarks-and-Datasets/2412.18194v1.pdf) | [2412.18194v1](https://arxiv.org/abs/2412.18194v1) | 34 | 新增 |
| LIBERO-Plus | LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models | [2510.13626v3.pdf](Benchmarks-and-Datasets/2510.13626v3.pdf) | [2510.13626v3](https://arxiv.org/abs/2510.13626v3) | 32 | 新增 |
| LIBERO-PRO | LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization | [2510.03827v2.pdf](Benchmarks-and-Datasets/2510.03827v2.pdf) | [2510.03827v2](https://arxiv.org/abs/2510.03827v2) | 10 | 新增 |
| The Colosseum | THE COLOSSEUM: A Benchmark for Evaluating Generalization for Robotic Manipulation | [2402.08191v2.pdf](Benchmarks-and-Datasets/2402.08191v2.pdf) | [2402.08191v2](https://arxiv.org/abs/2402.08191v2) | 33 | 新增 |
| RoboCasa365 | RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots | [2603.04356v1.pdf](Benchmarks-and-Datasets/2603.04356v1.pdf) | [2603.04356v1](https://arxiv.org/abs/2603.04356v1) | 25 | 新增 |
| RoboLab | RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies | [2604.09860v4.pdf](Benchmarks-and-Datasets/2604.09860v4.pdf) | [2604.09860v4](https://arxiv.org/abs/2604.09860v4) | 20 | 新增 |
| RoboDojo | RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies | [2607.04434v3.pdf](Benchmarks-and-Datasets/2607.04434v3.pdf) | [2607.04434v3](https://arxiv.org/abs/2607.04434v3) | 46 | 新增 |
| BEHAVIOR-1K | BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation | [2403.09227v1.pdf](Benchmarks-and-Datasets/2403.09227v1.pdf) | [2403.09227v1](https://arxiv.org/abs/2403.09227v1) | 43 | 新增 |
| RoboArena | RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies | [2506.18123v2.pdf](Benchmarks-and-Datasets/2506.18123v2.pdf) | [2506.18123v2](https://arxiv.org/abs/2506.18123v2) | 29 | 新增 |
| RoboChallenge / Table30 | RoboChallenge: Large-scale Real-robot Evaluation of Embodied Policies | [2510.17950v1.pdf](Benchmarks-and-Datasets/2510.17950v1.pdf) | [2510.17950v1](https://arxiv.org/abs/2510.17950v1) | 12 | 新增 |
| Language Table | Interactive Language: Talking to Robots in Real Time | [2210.06407v1.pdf](Vision-Language-Action-Models/2210.06407v1.pdf) | [2210.06407v1](https://arxiv.org/abs/2210.06407v1) | 11 | 新增 |

Interactive Language 是 [Language Table 官方仓库](https://github.com/google-research/language-table) 指向的来源论文。其主贡献是 visuolinguomotor learning framework，因此 PDF 归入 `Vision-Language-Action-Models/`，而非仅因调研中以 benchmark 名称提到就归入 benchmark 目录。DexMimicGen 的主贡献是 automated data generation，归入 `Benchmarks-and-Datasets/`。

## 3. 不单独计为论文的项目

| 项目 | 处理方式 | 官方材料 |
| --- | --- | --- |
| GR00T N1.5 | 本次调研未建立独立论文来源；已下载 N1 white paper，不用发布研究页面冒充另一篇 PDF | [发布研究](https://research.nvidia.com/labs/gear/gr00t-n1_5/) |
| GR00T N1.6 | 后续 benchmark results 与发布材料不是独立论文；参考 N1 white paper、发布研究及对应版本仓库 | [发布研究](https://research.nvidia.com/labs/gear/gr00t-n1_6/)／[官方代码](https://github.com/NVIDIA/Isaac-GR00T/tree/n1d6) |
| GR00T N1.7 | 已有 release、model card 和 evaluation results，但本次未确认独立 N1.7 paper；不能把 N1 white paper 改名当作 N1.7 论文 | [Model card](https://huggingface.co/nvidia/GR00T-N1.7-3B)／[官方 examples](https://github.com/NVIDIA/Isaac-GR00T/tree/dac54b313160c201c450f97df0fd74ebd379c892/examples) |
| GR-1 Tabletop / Sim GR-1 Language | 作为 N1 论文／后续发布研究中的 evaluation suite 处理，不虚构独立论文 | [N1 论文](https://arxiv.org/abs/2503.14734v2)／[N1.5 发布研究](https://research.nvidia.com/labs/gear/gr00t-n1_5/) |
| 2026 BEHAVIOR Challenge | 已下载 BEHAVIOR-1K 基础论文；2026 的 100-task protocol 仍需参考 challenge 文档，不将基础论文等同于 2026 新论文 | [Challenge](https://behavior.stanford.edu/challenge/index.html) |
| OpenPI | 按 π₀、FAST、π₀.₅ 等模型原论文处理，不重复下载仓库说明为“OpenPI paper” | [官方仓库](https://github.com/Physical-Intelligence/openpi) |

本清单不把仅作为背景说明出现的 DROID、OXE、BridgeData、Isaac Sim / Lab、PerAct、RVT 等再展开为额外下载范围。

## 4. 文件验证与保留说明

- 已检查 PDF 文件头、`pdfinfo` 页数、`pdftotext` 正文可读性，并核对标题、abstract、introduction 与 conclusion/discussion 的内容和归类。
- **新增 20 份 PDF 的 `qpdf --check` 均返回 0。**
- 已有 5 份中，4 份 `qpdf --check` 返回 0；原有 `pi0.pdf` 返回 3，原因是 linearization hint table 警告。它可由 `pdfinfo` 和 `pdftotext` 正常读取，本次未修复、重写或覆盖；具体警告保存在 JSON 清单中。
- 新 PDF 归档后的 SHA-256 与下载文件一致；所有原有 PDF 的 SHA-256 保持不变。
- 论文库入口见 [README](../README.md)，全部PDF见 [完整目录](../CATALOG.md)。
