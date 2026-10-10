# Ultralytics 相关模型原论文补齐清单

核查与下载日期：2026-10-10。

## 范围与核查依据

本次按 [Ultralytics 官方模型列表](https://docs.ultralytics.com/models/)及各模型页面的原论文引用查缺补漏，并补齐 YOLOv1／v2 基础谱系和 YOLOv6 官方仓库列出的两篇论文。“Ultralytics 相关”指该生态支持或介绍的模型，并不表示这些论文都由 Ultralytics 团队撰写。

- [Ultralytics 官方 CITATION.cff](https://github.com/ultralytics/ultralytics/blob/main/CITATION.cff) 将 YOLO26 的 `2606.03748` 列为首选论文引用。
- YOLOv3／v4／v6／v7／v9、YOLO-World、YOLOE，以及 SAM／SAM 2／SAM 3／MobileSAM／FastSAM 的官方模型页面均给出对应原论文。
- [YOLOv6 作者仓库](https://github.com/meituan/YOLOv6)明确列出初版报告和 v3.0 报告；它们是两篇独立论文，不是同一 arXiv 条目的重复版本。
- YOLOv1／v2 是 YOLO 系列的基础论文，来源可由 [YOLO 作者项目页](https://pjreddie.com/darknet/yolo/)及论文作者、标题核对。

下载前检查了库中 96 份 PDF 的文件名及前两页识别信息。下列 16 篇此前均未收录，均已下载，采用下载时的 arXiv 最新版本，共约 **109.29 MiB**。已检查标题、摘要、引言和结论（YOLOv3 为末尾总结章节），并核对 PDF 格式、版本、页数和归档前后的 SHA-256。原有 96 份 PDF 的校验和均未变化。

本清单覆盖模型原论文及上述基础谱系，不展开所有骨干网络、优化器、跟踪算法等参考文献，也不把第三方综述或应用改进论文当成某版本的官方原论文。

## 新增：检测与开放词汇模型（11 篇）

归档目录：`Computer-Vision/Object-Detection/`。YOLOE 同时支持检测和分割，其主要贡献是统一开放提示的 YOLO 模型，因此与检测谱系一起归档。

| 模型／方法 | 论文标题 | 本地 PDF | arXiv 来源／版本 | 页数 |
| --- | --- | --- | --- | ---: |
| YOLOv1 | You Only Look Once: Unified, Real-Time Object Detection | [1506.02640v5.pdf](Object-Detection/1506.02640v5.pdf) | [1506.02640v5](https://arxiv.org/abs/1506.02640v5) | 10 |
| YOLOv2／YOLO9000 | YOLO9000: Better, Faster, Stronger | [1612.08242v1.pdf](Object-Detection/1612.08242v1.pdf) | [1612.08242v1](https://arxiv.org/abs/1612.08242v1) | 9 |
| YOLOv3 | YOLOv3: An Incremental Improvement | [1804.02767v1.pdf](Object-Detection/1804.02767v1.pdf) | [1804.02767v1](https://arxiv.org/abs/1804.02767v1) | 6 |
| YOLOv4 | YOLOv4: Optimal Speed and Accuracy of Object Detection | [2004.10934v1.pdf](Object-Detection/2004.10934v1.pdf) | [2004.10934v1](https://arxiv.org/abs/2004.10934v1) | 17 |
| YOLOv6 | YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications | [2209.02976v1.pdf](Object-Detection/2209.02976v1.pdf) | [2209.02976v1](https://arxiv.org/abs/2209.02976v1) | 17 |
| YOLOv6 v3.0 | YOLOv6 v3.0: A Full-Scale Reloading | [2301.05586v1.pdf](Object-Detection/2301.05586v1.pdf) | [2301.05586v1](https://arxiv.org/abs/2301.05586v1) | 7 |
| YOLOv7 | YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors | [2207.02696v1.pdf](Object-Detection/2207.02696v1.pdf) | [2207.02696v1](https://arxiv.org/abs/2207.02696v1) | 15 |
| YOLOv9 | YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information | [2402.13616v2.pdf](Object-Detection/2402.13616v2.pdf) | [2402.13616v2](https://arxiv.org/abs/2402.13616v2) | 18 |
| Ultralytics YOLO26／YOLOE-26 | Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models | [2606.03748v1.pdf](Object-Detection/2606.03748v1.pdf) | [2606.03748v1](https://arxiv.org/abs/2606.03748v1) | 31 |
| YOLO-World | YOLO-World: Real-Time Open-Vocabulary Object Detection | [2401.17270v3.pdf](Object-Detection/2401.17270v3.pdf) | [2401.17270v3](https://arxiv.org/abs/2401.17270v3) | 15 |
| YOLOE | YOLOE: Real-Time Seeing Anything | [2503.07465v2.pdf](Object-Detection/2503.07465v2.pdf) | [2503.07465v2](https://arxiv.org/abs/2503.07465v2) | 15 |

## 新增：图像与视频分割模型（5 篇）

归档目录：`Computer-Vision/Image-and-Video-Segmentation/`。SAM 2／3 的主要贡献包含视频分割，因此没有混放在实时目标检测目录。

| 模型／方法 | 论文标题 | 本地 PDF | arXiv 来源／版本 | 页数 |
| --- | --- | --- | --- | ---: |
| SAM | Segment Anything | [2304.02643v1.pdf](Image-and-Video-Segmentation/2304.02643v1.pdf) | [2304.02643v1](https://arxiv.org/abs/2304.02643v1) | 30 |
| SAM 2 | SAM 2: Segment Anything in Images and Videos | [2408.00714v2.pdf](Image-and-Video-Segmentation/2408.00714v2.pdf) | [2408.00714v2](https://arxiv.org/abs/2408.00714v2) | 42 |
| SAM 3 | SAM 3: Segment Anything with Concepts | [2511.16719v2.pdf](Image-and-Video-Segmentation/2511.16719v2.pdf) | [2511.16719v2](https://arxiv.org/abs/2511.16719v2) | 78 |
| MobileSAM | Faster Segment Anything: Towards Lightweight SAM for Mobile Applications | [2306.14289v2.pdf](Image-and-Video-Segmentation/2306.14289v2.pdf) | [2306.14289v2](https://arxiv.org/abs/2306.14289v2) | 10 |
| FastSAM | Fast Segment Anything | [2306.12156v1.pdf](Image-and-Video-Segmentation/2306.12156v1.pdf) | [2306.12156v1](https://arxiv.org/abs/2306.12156v1) | 11 |

## 已有原论文：不重复下载

- [YOLOv10](Object-Detection/2405.14458v2.pdf)、[YOLOv12](Object-Detection/2502.12524v1.pdf)。
- [RT-DETR](Object-Detection/2304.08069v3.pdf)、[RT-DETRv2](Object-Detection/2407.17140v1.pdf)。
- 同目录还保留首批下载的 YOLOv13、RT-DETRv3／v4、D-FINE、DEIM 和 DEIMv2；详见 [首批截图清单](Object-Detection/README.md)及 [完整索引](../INDEX.md)。

## 没有确认可下载的官方原论文

| 模型／版本 | 核查结果 | 官方依据 |
| --- | --- | --- |
| YOLOv5 | 官方明确表示未发表正式研究论文，提供软件引用；未下载第三方替代论文 | [Citations and Acknowledgments](https://docs.ultralytics.com/models/yolov5/#citations-and-acknowledgments) |
| YOLOv8 | 官方明确表示未发表正式研究论文，提供软件引用；未下载第三方替代论文 | [Citations and Acknowledgments](https://docs.ultralytics.com/models/yolov8/#citations-and-acknowledgments) |
| YOLO11 | 官方明确表示未发表正式研究论文，提供软件引用；未下载第三方替代论文 | [Citations and Acknowledgments](https://docs.ultralytics.com/models/yolo11/#citations-and-acknowledgments) |
| YOLO27 | 官方页面为 Coming Soon 预告，模型未发布，未给出独立原论文下载或引用 | [官方模型页](https://docs.ultralytics.com/models/yolo27/) |
| YOLO-NAS | 官方引用指向 SuperGradients 软件项目及 Zenodo 软件记录，未确认独立原论文 | [官方引用](https://docs.ultralytics.com/models/yolo-nas/#citations-and-acknowledgments)、[Zenodo](https://zenodo.org/records/7789328) |

YOLOE-26 的技术说明已包含在 YOLO26 原论文中，没有按模型名字另存同一篇 PDF。Ultralytics 的 LLM 接口是软件接口，不作为单独视觉模型原论文处理。
