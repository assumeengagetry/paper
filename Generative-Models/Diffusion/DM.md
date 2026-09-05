![](/home/assumeengage/.config/marktext/images/2026-09-01-08-47-32-image.png)

![](/home/assumeengage/.config/marktext/images/2026-09-01-08-47-47-image.png)

![](/home/assumeengage/.config/marktext/images/2026-09-01-09-11-20-image.png)

![](/home/assumeengage/.config/marktext/images/2026-09-01-09-11-31-image.png)

比较火的几个生成模型：

![在这里插入图片描述](https://p3-xtjj-sign.byteimg.com/tos-cn-i-73owjymdk6/2ec562e7689149358767abd3d6ca14b4~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgREZDRUQ=:q75.awebp?rk3s=f64ab15b&x-expires=1788691452&x-signature=XVUQSzdxA%2FU1sgHF1aiR%2FvWuK%2Bg%3D)

截至 **2026 年 9 月 1 日**，我按“正式同行评审 + 计算机公认旗舰刊/领域顶刊/顶会 Survey Track”筛选，排除了纯 arXiv、IEEE Access、Artificial Intelligence Review 等条目。

先给结论：如果你想了解最新业内格局，不能只读一篇通用综述，最好用“全景 + 效率部署 + 可控/对齐 + 视频/机器人”组合。

## 最值得优先读的 6 篇

| 顺序  | 论文                                                                                                                            | 顶刊                         | 主要价值                                                                      |
| --- | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------- |
| 1   | [Efficient Diffusion Models: A Comprehensive Survey From Principles to Practices](https://doi.org/10.1109/TPAMI.2025.3569700) | TPAMI 2025                 | 最贴近“业内与系统”：U-Net、DiT、U-ViT、Mamba，PEFT、蒸馏、少步采样、缓存、量化、移动/云端、多 GPU 部署。**首推** |
| 2   | [Diffusion Models: A Comprehensive Survey of Methods and Applications](https://doi.org/10.1145/3626235)                       | ACM Computing Surveys 2024 | 最标准的基础地图：DDPM、Score Matching、SDE，以及高效采样、likelihood、结构化数据和跨领域应用            |
| 3   | [Controllable Generation With Text-to-Image Diffusion Models](https://doi.org/10.1109/TPAMI.2025.3646548)                     | TPAMI 2026                 | 单条件、多条件、统一控制、编辑、3D；还纳入 Flow Matching、Rectified Flow，是理解当前生成模型路线迁移的重要综述    |
| 4   | [Alignment of Diffusion Models: Fundamentals, Challenges, and Future](https://doi.org/10.1145/3796982)                        | ACM Computing Surveys 2026 | Diffusion 后训练：人类偏好、奖励模型、RL、DPO 类方法、评测、安全与 reward hacking                  |
| 5   | [A Survey on Video Diffusion Models](https://doi.org/10.1145/3696415)                                                         | ACM Computing Surveys 2025 | 视频生成、视频编辑、视频理解的完整 taxonomy；目前正式顶刊里最直接的视频 Diffusion 综述                     |
| 6   | [Diffusion Models in Robotics: A Survey](https://doi.org/10.1007/s11263-026-02893-1)                                          | IJCV 2026                  | Diffusion Policy、轨迹/动作生成、规划、视频生成辅助机器人学习、数据集与 benchmark；和你的 VLA/具身方向最相关    |

注意：发表年份不等于文献覆盖年份。比如 TPAMI 2025 的效率综述和 CSUR 2025 的视频综述，主体仍主要覆盖到 2024。因此没有任何单篇综述能够完整包含 2025–2026 的全部产业模型。

## 按细分方向选择

- 图像编辑：[Diffusion Model-Based Image Editing: A Survey](https://doi.org/10.1109/TPAMI.2025.3541625)，TPAMI 2025。覆盖 training-based、test-time tuning、training-free、inpainting/outpainting 和 EditEval。

- 低层视觉与逆问题：[Diffusion Models in Low-Level Vision](https://doi.org/10.1109/TPAMI.2025.3545047)，TPAMI 2025。覆盖超分、去模糊、去噪、医疗、遥感等二十余类任务。

- 表征学习：[Diffusion Models and Representation Learning](https://doi.org/10.1109/TPAMI.2026.3658965)，TPAMI 2026。研究如何从预训练 Diffusion 中提取识别特征，以及如何用自监督表征改善生成。

- 攻击与防御：[Attacks and Defenses for Generative Diffusion Models](https://doi.org/10.1145/3721479)，ACM Computing Surveys 2025。涵盖对抗攻击、后门、成员推断和相应防御。

- 记忆、版权与训练数据复制：[Replication in Visual Diffusion Models](https://doi.org/10.1109/TPAMI.2026.3713990)，TPAMI 2026 Early Access。目前最新，但尚未编入正式卷期。

- 时间序列与时空数据：[A Survey on Diffusion Models for Time Series and Spatio-Temporal Data](https://doi.org/10.1145/3783986)，ACM Computing Surveys 2026。覆盖预测、插补、异常检测、生成，以及交通、气候、能源、医疗和音频。

- 图生成、分子与蛋白：[Diffusion-Based Graph Generative Methods](https://doi.org/10.1109/TKDE.2024.3466301)，TKDE 2024。

- 生物信息与药物设计：[Diffusion Models in Bioinformatics and Computational Biology](https://doi.org/10.1038/s44222-023-00114-9)，Nature Reviews Bioengineering 2024。

- 医学影像最新入口：[Physics-Inspired Generative Models in Medical Imaging](https://doi.org/10.1146/annurev-bioeng-102723-013922)，Annual Review of Biomedical Engineering 2025。

## 顶会中的综述

真正发表于顶会 proceedings 的 Diffusion 综述很少。目前最明确的是：

- [Diffusion Models for Non-autoregressive Text Generation: A Survey](https://doi.org/10.24963/ijcai.2023/750)，IJCAI 2023 Survey Track。适合了解早期连续/离散文本扩散，但不覆盖 2025–2026 的 Diffusion LLM 浪潮。

Diffusion LLM 最新的正式综述是 [Discrete Diffusion in Large Language and Multimodal Models](https://openreview.net/forum?id=0DsqnkP8Cp)，TMLR 2026。TMLR 是正式同行评审的一线 ML venue，但没有传统 CCF-A/影响因子口径，所以我没有把它混入上面的“严格顶刊表”。不过要了解 LLaDA、Dream、Seed Diffusion、并行解码、cache 与 serving，它目前最有用。

## 这些综述反映出的 2026 业内主线

这是我根据上述顶刊综述作出的综合判断：

1. **架构路线正在从 U-Net/LDM 转向 DiT、MMDiT 与 Flow Matching/Rectified Flow。**  
   2026 年“Diffusion”已经更像一个扩散—流模型家族，而不只是传统 DDPM。

2. **竞争重点从单纯图像质量转向系统效率。**  
   Few-step/one-step、蒸馏、缓存、并行、量化、多 GPU 和边缘部署，已成为核心工程方向。

3. **产品能力从 T2I 向多条件控制、编辑、视频、3D/4D 和 World Simulator 迁移。**  
   可进一步参考非 Diffusion-only 的 TPAMI 2026 综述 [Simulating the Real World](https://doi.org/10.1109/TPAMI.2026.3690925)。

4. **后训练与治理成为独立研究主线。**  
   包括偏好对齐、RL、版权复制、训练数据记忆、成员推断、后门和内容安全。

5. **Diffusion 正在离开纯视觉生成。**  
   一条路线进入机器人动作与轨迹生成；另一条进入非自回归语言模型，但 dLLM 的 KV Cache、推理步数和 serving 生态仍明显落后于自回归 LLM。

结合你目前做的 VLA、推理性能和 runtime，我建议阅读顺序是：

> **TPAMI Efficiency → IJCV Robotics → CSUR Video → TPAMI Controllable Generation → CSUR Alignment/Security**

这条路线比单纯从数学综述一路读下来，更容易建立“模型—workload—runtime—部署”的完整联系。

如果有用，我可以设置“每月跟踪 Diffusion 顶刊顶会新综述”，方便持续补上正式发表的新论文。

![](/home/assumeengage/.config/marktext/images/2026-09-02-00-07-20-image.png)

![](/home/assumeengage/.config/marktext/images/2026-09-02-00-27-10-image.png)

![](/home/assumeengage/.config/marktext/images/2026-09-02-02-42-28-image.png)

![](/home/assumeengage/.config/marktext/images/2026-09-02-02-50-43-image.png)

![](/home/assumeengage/.config/marktext/images/2026-09-02-04-26-32-image.png)

The main content review of this survey about diffusion models in robotics
