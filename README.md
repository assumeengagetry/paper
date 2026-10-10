# Paper Library

这是按研究主题归档的本地论文、书籍和研究笔记库。完整文件列表见 [INDEX.md](INDEX.md)，整理规则见 [AGENTS.md](AGENTS.md)。

更新日期：2026-10-10。

## 主题入口

| 目录 | 主要内容 | PDF 数量 |
| --- | --- | ---: |
| [Robotics-and-Embodied-AI](Robotics-and-Embodied-AI/) | 世界动作模型、VLA 模型、扩散策略、推理系统、机器人评测与学习框架 | 33 |
| [Generative-Models](Generative-Models/) | 图像扩散、扩散语言模型、语音生成、科学应用、变分推断及语言模型训练 | 15 |
| [LLM-Systems](LLM-Systems/) | 模型架构、推理性能分析、服务与 KV Cache | 9 |
| [ML-Systems-and-Hardware](ML-Systems-and-Hardware/) | 分布式训练、GPU 编程、加速器性能与 Roofline | 6 |
| [Computer-Vision](Computer-Vision/) | 视觉语言预训练、实时与开放词汇目标检测、图像与视频分割 | 27 |
| [Explainable-and-Responsible-AI](Explainable-and-Responsible-AI/) | 可解释性、公平性与偏差 | 14 |
| [ML-Security-and-Robustness](ML-Security-and-Robustness/) | 后门、水印与模型压缩鲁棒性 | 3 |
| [Medical-AI](Medical-AI/) | 内窥镜与息肉分析 | 5 |
| [Research-Landscapes](Research-Landscapes/) | 团队、研究机构与研究信息源笔记 | — |

PDF 总数为 112，包含已有书籍、学位论文和保留的重复副本。

## 实时目标检测论文（2026-10-10 新增）

新增 10 篇 arXiv 论文 PDF：RT-DETR、RT-DETRv2／v3／v4、DEIM、DEIMv2、D-FINE、YOLOv10／v12／v13，统一归档至 [Computer-Vision/Object-Detection](Computer-Vision/Object-Detection/)。完整标题、本地文件和来源版本见 [下载清单](Computer-Vision/Object-Detection/README.md)。

全部使用下载时的最新版本，总大小约 32.00 MiB，已核对标题、版本、页数和归档前后的校验和。截图中的 `ultralytics/ultralytics` 是 GitHub 仓库，不作为论文下载。

## Ultralytics 相关模型原论文补齐（2026-10-10）

按 Ultralytics 官方模型列表、原论文引用及 YOLO 基础谱系另新增 16 篇 PDF，共约 109.29 MiB：

- **检测相关 11 篇**：YOLOv1／v2／v3／v4、YOLOv6 初版与 v3.0、YOLOv7、YOLOv9、Ultralytics YOLO26、YOLO-World、YOLOE，放在 `Computer-Vision/Object-Detection/`。
- **分割相关 5 篇**：SAM、SAM 2／3、MobileSAM、FastSAM，放在 `Computer-Vision/Image-and-Video-Segmentation/`。

完整标题、本地文件、版本及排除项见 [Ultralytics 原论文清单](Computer-Vision/Ultralytics-Papers.md)。YOLOv5／v8／11 的官方页面明确说明没有正式研究论文；YOLO27 仍为预告，YOLO-NAS 的官方引用指向软件项目，未用第三方论文替代。YOLO26 的官方原论文已确认并下载。原有 96 份 PDF 均未改变。

## 本次所需项目与论文

| 项目 | 对应论文／资料 | 本地文件 | 官方来源 |
| --- | --- | --- | --- |
| FastWAM | Fast-WAM: Do World Action Models Need Test-time Future Imagination? | [2603.16666v2.pdf](Robotics-and-Embodied-AI/World-Action-Models/2603.16666v2.pdf)（原库已有） | [arXiv](https://arxiv.org/abs/2603.16666v2)、[项目页](https://yuantianyuan01.github.io/FastWAM/) |
| Flash-WAM | Flash-WAM: Modality-Aware Distillation for World Action Models | [2606.05254v2.pdf](Robotics-and-Embodied-AI/World-Action-Models/2606.05254v2.pdf)（本次下载，2026-10-03 更新版） | [arXiv](https://arxiv.org/abs/2606.05254v2)、[项目页](https://flashwam.github.io/) |
| LeRobot | LeRobot: An Open-Source Library for End-to-End Robot Learning，ICLR 2026 | [2602.22818v1.pdf](Robotics-and-Embodied-AI/Robot-Learning-Frameworks/2602.22818v1.pdf)（本次下载） | [arXiv](https://arxiv.org/abs/2602.22818v1)、[官方仓库与引用](https://github.com/huggingface/lerobot#citation) |
| RoboTwin 1.0 | RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins | [2504.13059v1.pdf](Robotics-and-Embodied-AI/Benchmarks-and-Datasets/2504.13059v1.pdf)（本次下载） | [arXiv](https://arxiv.org/abs/2504.13059v1)、[官方仓库](https://github.com/RoboTwin-Platform/RoboTwin) |
| RoboTwin 2.0 | RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation | [2506.18088v2.pdf](Robotics-and-Embodied-AI/Benchmarks-and-Datasets/2506.18088v2.pdf)（本次下载） | [arXiv](https://arxiv.org/abs/2506.18088v2)、[项目页](https://robotwin-platform.github.io/) |
| LIBERO | LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning | [2306.03310v2.pdf](Robotics-and-Embodied-AI/Benchmarks-and-Datasets/2306.03310v2.pdf)（本次下载） | [arXiv](https://arxiv.org/abs/2306.03310v2)、[官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| OpenPI／π0 | π0: A Vision-Language-Action Flow Model for General Robot Control | [pi0.pdf](Robotics-and-Embodied-AI/Vision-Language-Action-Models/pi0.pdf)（原库已有） | [arXiv](https://arxiv.org/abs/2410.24164)、[OpenPI](https://github.com/Physical-Intelligence/openpi) |
| OpenPI／π0.5 | π0.5: a Vision-Language-Action Model with Open-World Generalization | [2504.16054v1.pdf](Robotics-and-Embodied-AI/Vision-Language-Action-Models/2504.16054v1.pdf)（本次下载） | [arXiv](https://arxiv.org/abs/2504.16054v1)、[项目页](https://www.pi.website/blog/pi05) |
| OpenPI／π0-FAST | FAST: Efficient Action Tokenization for Vision-Language-Action Models | [2501.09747v1.pdf](Robotics-and-Embodied-AI/Vision-Language-Action-Models/2501.09747v1.pdf)（本次下载） | [arXiv](https://arxiv.org/abs/2501.09747v1)、[项目页](https://www.pi.website/research/fast) |
| Spec-VLA | VLA 模型在昇腾算力上的适配与优化（集群项目内现有学位论文） | [main.pdf](Robotics-and-Embodied-AI/VLA-Inference/Spec-VLA/main.pdf)（本次从集群取回） | [来源说明](Robotics-and-Embodied-AI/VLA-Inference/Spec-VLA/README.md) |
| APXinf-robo | 项目说明，以及官方引用的 STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction | [APXinf-robo 说明](Robotics-and-Embodied-AI/VLA-Inference/APXinf-robo.md)、[STEP PDF](Robotics-and-Embodied-AI/Diffusion-Policies/2602.08245v2.pdf)（STEP 本次下载） | [官方仓库](https://github.com/RLinf/APXinf-robo)、[STEP arXiv](https://arxiv.org/abs/2602.08245v2) |

OpenPI 是包含 π0、π0.5、π0-FAST 等模型的代码库，因此按模型原论文归档。APXinf／APXinf-robo 的独立公开论文尚未确认；已保存项目说明和它明确引用的 STEP 论文。Spec-VLA 的 PDF 来自集群中的现有学位论文稿，本次没有确认其公开预印本或正式发表版本。

## 机器人主题分类

```text
Robotics-and-Embodied-AI/
├── World-Action-Models/             视频／世界模型与动作联合建模
│   └── Additional-Copies/           原根目录已有的三个相同副本
├── Vision-Language-Action-Models/   π0、π0.5、FAST、LLaDA-VLA 等模型与表示方法
├── Diffusion-Policies/              Diffuser、STEP、机器人扩散综述
├── VLA-Inference/                  部署、推理加速、性能表征与项目说明
│   └── Spec-VLA/                   现有学位论文及来源说明
├── Benchmarks-and-Datasets/        LIBERO、RoboTwin、评测协议与基准审计
└── Robot-Learning-Frameworks/      LeRobot 等机器人学习框架
```

STEP 的主贡献是扩散策略的 warm-start 少步推理，因此 PDF 放在 `Diffusion-Policies`；APXinf-robo 作为部署引擎放在 `VLA-Inference`。Flash-WAM 虽然加速推理，其研究对象和方法是联合视频—动作世界模型的蒸馏，因此归入 `World-Action-Models`。

## 本次整理记录

- 新增 8 份公开论文 PDF，另从集群取回 1 份 Spec-VLA 学位论文 PDF；合计新增 9 份。
- 对根目录原有的 22 份 PDF、`rss.md` 和 `post/20260924.pptx` 进行主题归档；共移动 24 个既有文件。
- 扩散语言模型集中在 `Generative-Models/Diffusion-Language-Models`；F5-TTS 放在 `Speech-and-Audio`；流体与孔隙图像扩散放在 `Diffusion/Scientific-Applications`。
- TPU v4 放在加速器性能目录；GPU 编程教材放在 `GPU-Programming`；以 DDP/FSDP 实验为主的演示文件放在 `Distributed-Training`；研究信息源笔记 `rss.md` 放在 `Research-Landscapes`。
- 原库 77 份 PDF 均通过 SHA-256 校验，原文件名及内容完整保留。
- LingBot-VA、DreamZero、GigaWorld-Policy 此前各有一个根目录副本和一个已分类副本。根目录副本已归入 `World-Action-Models/Additional-Copies`。
- `Language Ranker` 的两个既有 PDF 副本及各自 MinerU 提取文件均保留在原主题目录。
- 本次下载的 PDF 通过 PDF 格式、页数和正文标题检查；来源与版本在上表中记录。

整理后根目录作为导航入口。需要找文件时，从项目对应表或完整 [INDEX.md](INDEX.md) 进入。
