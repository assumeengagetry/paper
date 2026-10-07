# APXinf-robo：项目与论文对应说明

核对日期：2026-10-07。

## 官方来源

- [APXinf-robo 仓库](https://github.com/RLinf/APXinf-robo)
- [APXinf-robo README](https://github.com/RLinf/APXinf-robo/blob/main/README.md)
- [ApxInf 核心引擎](https://github.com/infinigence/ApxInf)
- 本次核对时，Robo 仓库记录的引擎子模块提交为 `7f37642dd6c5896bca66d7640d9944c2a3fe04b4`。

## 项目定位

APXinf-robo 将 ApxInf 推理引擎接入机器人、仿真环境和评测基准。核心引擎使用 Rust、Python 绑定以及 CUDA 内核，项目包含 π0.5 推理、精度配置、延迟测速、OpenPI 兼容服务和 LIBERO 评测等功能。官方说明还列出 π0-FAST、GR00T N1.7 与 Qwen-Drive 的性能结果及对应文档。

因此本项目归入 `VLA-Inference`，重点是机器人模型的推理系统与部署。

## 独立论文的核对结果

本次检查官方仓库 README、项目引用信息和论文索引，尚未确认一篇以 APXinf／APXinf-robo 为题的独立公开论文。此文件是项目资料说明。

官方 README 明确说明引擎集成了 STEP 的 warm-start 少步动作生成方法，并给出该论文的引用。相关 PDF 已归档：

- [STEP: Warm-Started Visuomotor Policies with Spatiotemporal Consistency Prediction](../Diffusion-Policies/2602.08245v2.pdf)
- [arXiv:2602.08245](https://arxiv.org/abs/2602.08245)

STEP 的贡献是扩散策略的时空一致性 warm-start 与少步动作生成，因此 PDF 放在 `Diffusion-Policies`。它是 APXinf 所引用的相关方法论文。

## 关联的模型与评测论文

- [π0](../Vision-Language-Action-Models/pi0.pdf)
- [π0.5](../Vision-Language-Action-Models/2504.16054v1.pdf)
- [FAST／π0-FAST](../Vision-Language-Action-Models/2501.09747v1.pdf)
- [LIBERO](../Benchmarks-and-Datasets/2306.03310v2.pdf)

阅读顺序可以先看模型与评测论文，再读 STEP，最后结合官方源码理解引擎实现。README 中给出的设备延迟是项目报告的数据，不能直接当作当前 A800 集群的实测结果。
