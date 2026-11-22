# Attention 与推理系统社区长文离线阅读清单

归档日期：2026-11-22。合并两轮清单，重复URL只存一份；有标题但无直链的3篇已从vLLM官方索引与原文确认。

**47个来源全部取得：45个博客页面、2个文档页面；343处有效正文图片引用已本地化，保存343个不同图片内容，约70.39 MiB。另新增4份论文PDF（37.76 MiB、72页），复用9篇已有论文。全库PDF从576增至580，既有资料不覆盖。**

- **用浏览器打开 [离线HTML目录](ATTENTION-Systems-20261122.html)**，可进入各篇正文和本地图片。
- [来源、校验和、媒体与检查结果](ATTENTION-Systems-20261122.manifest.json)
- [全库580份PDF目录](CATALOG-20261122.md)

## 先读这六篇

1. [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](LLM-Systems/Serving-and-KV-Cache/Community-Articles-20261122/vllm/2023-06-20-vllm.html)
2. [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](LLM-Systems/Performance-and-Profiling/Community-Articles-20261122/vllm/2025-09-05-anatomy-of-vllm.html)
3. [Fast and Expressive LLM Inference with RadixAttention and SGLang](LLM-Systems/Serving-and-KV-Cache/Community-Articles-20261122/lmsys/2024-01-17-sglang.html)
4. [Introduction to torch.compile and How It Works with vLLM](LLM-Systems/Performance-and-Profiling/Community-Articles-20261122/vllm/2025-08-20-torch-compile.html)
5. [Accelerating Self-Attentions for LLM Serving with FlashInfer](LLM-Systems/Attention-and-Kernels/Community-Articles-20261122/flashinfer/introduce-flashinfer.html)
6. [Mastering LLM Techniques: Inference Optimization](LLM-Systems/Performance-and-Profiling/Community-Articles-20261122/nvidia/mastering-llm-techniques-inference-optimization.html)

## 全部博客／文档

| 主题 | 正文／实际标题 | 来源 | 本地图片引用 |
| --- | --- | --- | ---: |
| 缓存、调度与阶段分离 | [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](LLM-Systems/Serving-and-KV-Cache/Community-Articles-20261122/vllm/2023-06-20-vllm.html) | [原网页](https://vllm.ai/blog/2023-06-20-vllm) | 13 |
| 推理系统、编译与性能 | [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](LLM-Systems/Performance-and-Profiling/Community-Articles-20261122/vllm/2025-09-05-anatomy-of-vllm.html) | [原网页](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) | 18 |
| 缓存、调度与阶段分离 | [Fast and Expressive LLM Inference with RadixAttention and SGLang](LLM-Systems/Serving-and-KV-Cache/Community-Articles-20261122/lmsys/2024-01-17-sglang.html) | [原网页](https://www.lmsys.org/blog/2024-01-17-sglang) | 5 |
| 推理系统、编译与性能 | [Introduction to torch.compile and How It Works with vLLM](LLM-Systems/Performance-and-Profiling/Community-Articles-20261122/vllm/2025-08-20-torch-compile.html) | [原网页](https://vllm.ai/blog/2025-08-20-torch-compile) | 9 |
| 推理Kernel与后端 | [Accelerating Self-Attentions for LLM Serving with FlashInfer](LLM-Systems/Attention-and-Kernels/Community-Articles-20261122/flashinfer/introduce-flashinfer.html) | [原网页](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html) | 13 |
| 推理系统、编译与性能 | [Mastering LLM Techniques: Inference Optimization](LLM-Systems/Performance-and-Profiling/Community-Articles-20261122/nvidia/mastering-llm-techniques-inference-optimization.html) | [原网页](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) | 12 |
| 可编程Attention | [FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention](ML-Systems-and-Hardware/GPU-Programming/Community-Articles-20261122/pytorch/flexattention.html) | [原网页](https://pytorch.org/blog/flexattention/) | 16 |
| 推理Kernel与后端 | [FlexAttention Part II: FlexAttention for Inference](LLM-Systems/Attention-and-Kernels/Community-Articles-20261122/pytorch/flexattention-for-inference.html) | [原网页](https://pytorch.org/blog/flexattention-for-inference/) | 10 |
| 可编程Attention | [FlexAttention + FlashAttention-4: Fast and Flexible](ML-Systems-and-Hardware/GPU-Programming/Community-Articles-20261122/pytorch/flexattention-flashattention-4-fast-and-flexible.html) | [原网页](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) | 11 |
| 量化Attention与扩散后端 | [SageAttention：即插即用的8-bit Attention 最佳实践](ML-Systems-and-Hardware/GPU-Programming/Community-Articles-20261122/qingkeai/SageAttention.html) | [原网页](https://qingkeai.online/blog/SageAttention) | 6 |
| FlashAttention／GPU原理 | [FlashAttention: Fast Transformer Training with Long Sequences](ML-Systems-and-Hardware/GPU-Programming/Community-Articles-20261122/hazy-research/2023-01-12-flashattention-long-sequences.html) | [原网页](https://hazyresearch.stanford.edu/blog/2023-01-12-flashattention-long-sequences) | 5 |
| FlashAttention／GPU原理 | [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](ML-Systems-and-Hardware/GPU-Programming/Community-Articles-20261122/hazy-research/2023-07-17-flash2.html) | [原网页](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2) | 5 |
| FlashAttention／GPU原理 | [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](ML-Systems-and-Hardware/GPU-Programming/Community-Articles-20261122/pytorch/flashattention-3.html) | [原网页](https://pytorch.org/blog/flashattention-3/) | 10 |
| FlashAttention／GPU原理 | [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](ML-Systems-and-Hardware/GPU-Programming/Community-Articles-20261122/together-ai/flashattention-4.html) | [原网页](https://www.together.ai/blog/flashattention-4) | 12 |
| 推理Kernel与后端 | [vLLM Triton Attention Backend Deep Dive](LLM-Systems/Attention-and-Kernels/Community-Articles-20261122/vllm/2026-03-04-vllm-triton-backend-deep-dive.html) | [原网页](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) | 10 |
| 缓存、调度与阶段分离 | [The State of FP8 KV-Cache and Attention Quantization in vLLM](LLM-Systems/Serving-and-KV-Cache/Community-Articles-20261122/vllm/2026-04-22-fp8-kvcache.html) | [原网页](https://vllm.ai/blog/2026-04-22-fp8-kvcache) | 14 |
| 缓存、调度与阶段分离 | [DeepSeek V4 in vLLM: Efficient Long-context Attention](LLM-Systems/Serving-and-KV-Cache/Community-Articles-20261122/vllm/2026-04-24-deepseek-v4.html) | [原网页](https://vllm.ai/blog/2026-04-24-deepseek-v4) | 3 |
| 推理Kernel与后端 | [Portable Paged Attention in Helion](LLM-Systems/Attention-and-Kernels/Community-Articles-20261122/pytorch/portable-paged-attention-in-helion.html) | [原网页](https://pytorch.org/blog/portable-paged-attention-in-helion/) | 8 |
| 推理Kernel与后端 | [Attention Backend](LLM-Systems/Attention-and-Kernels/Community-Articles-20261122/sglang-docs/attention_backend.html) | [原网页](https://docs.sglang.io/docs/advanced_features/attention_backend) | 0 |
| 量化Attention与扩散后端 | [Attention Backends](Generative-Models/Diffusion/Community-Articles-20261122/sglang-docs/attention_backends.html) | [原网页](https://docs.sglang.io/docs/sglang-diffusion/attention_backends) | 0 |
| 缓存、调度与阶段分离 | [SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs](LLM-Systems/Serving-and-KV-Cache/Community-Articles-20261122/lmsys/2024-12-04-sglang-v0-4.html) | [原网页](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) | 6 |
| 推理系统、编译与性能 | [SGLang Low-Latency Mode: Kernels and Runtime](LLM-Systems/Performance-and-Profiling/Community-Articles-20261122/lmsys/2026-10-18-sglang-low-latency-mode.html) | [原网页](https://www.lmsys.org/blog/2026-10-18-sglang-low-latency-mode) | 6 |
| 推理Kernel与后端 | [HPC-Ops × SGLang: High-Performance Attention, Router GEMM, and MoE Kernels from Tencent Hunyuan](LLM-Systems/Attention-and-Kernels/Community-Articles-20261122/lmsys/2026-08-07-hpc-ops-sglang.html) | [原网页](https://www.lmsys.org/blog/2026-08-07-hpc-ops-sglang) | 5 |
| 推理Kernel与后端 | [Run High-Performance LLM Inference Kernels from NVIDIA Using FlashInfer​​](LLM-Systems/Attention-and-Kernels/Community-Articles-20261122/nvidia/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer.html) | [原网页](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/) | 2 |
| FlashAttention／GPU原理 | [Tuning Flash Attention for Peak Performance in NVIDIA CUDA Tile](ML-Systems-and-Hardware/GPU-Programming/Community-Articles-20261122/nvidia/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile.html) | [原网页](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/) | 3 |
| FlashAttention／GPU原理 | [We reverse-engineered Flash Attention 4](ML-Systems-and-Hardware/GPU-Programming/Community-Articles-20261122/modal/reverse-engineer-flash-attention-4.html) | [原网页](https://modal.com/blog/reverse-engineer-flash-attention-4) | 5 |
| 推理Kernel与后端 | [Making FlashAttention-4 faster for inference](LLM-Systems/Attention-and-Kernels/Community-Articles-20261122/modal/flash-attention-4-faster.html) | [原网页](https://modal.com/blog/flash-attention-4-faster) | 10 |
| 推理Kernel与后端 | [Announcing Humming: Fast, Flexible Kernels for Quantized Inference](LLM-Systems/Model-Compression-and-Quantization/Community-Articles-20261122/vllm/2026-10-21-humming.html) | [原网页](https://vllm.ai/blog/2026-10-21-humming) | 11 |
| 推理Kernel与后端 | [Keeping Agents Fast: High-Interactivity Decode with FlyDSL MonoKernels in vLLM](LLM-Systems/Attention-and-Kernels/Community-Articles-20261122/vllm/2026-11-18-flydsl-monokernels.html) | [原网页](https://vllm.ai/blog/2026-11-18-flydsl-monokernels) | 5 |
| 推测解码 | [Diving into speculative decoding training support for vLLM with Speculators v0.3.0](LLM-Systems/Speculative-Decoding/Community-Articles-20261122/vllm/2025-12-13-speculators-v030.html) | [原网页](https://vllm.ai/blog/2025-12-13-speculators-v030) | 4 |
| 缓存、调度与阶段分离 | [Unified Radix Cache in SGLang: one tree for hybrid model prefix caching](LLM-Systems/Serving-and-KV-Cache/Community-Articles-20261122/sglang/unified-radix-cache.html) | [原网页](https://www.sglang.io/blog/unified-radix-cache) | 0 |
| 推理系统、编译与性能 | [Advanced CUDA Graph Techniques in SGLang](LLM-Systems/Performance-and-Profiling/Community-Articles-20261122/lmsys/2026-08-17-advanced-cuda-graph.html) | [原网页](https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph) | 6 |
| 推理Kernel与后端 | [Humming in SGLang: Flexible Low-Bit Kernels for Efficient MoE Serving](LLM-Systems/Model-Compression-and-Quantization/Community-Articles-20261122/lmsys/2026-11-18-humming-sglang.html) | [原网页](https://www.lmsys.org/blog/2026-11-18-humming-sglang) | 2 |
| 缓存、调度与阶段分离 | [Accelerating GLM-5.3-Flash on H200: High-Throughput Disaggregated Serving](LLM-Systems/Serving-and-KV-Cache/Community-Articles-20261122/lmsys/2026-11-01-glm53-h200-pd.html) | [原网页](https://www.lmsys.org/blog/2026-11-01-glm53-h200-pd) | 9 |
| 分布式训练 | [Introducing PyTorch Fully Sharded Data Parallel (FSDP) API](ML-Systems-and-Hardware/Distributed-Training/Community-Articles-20261122/pytorch/introducing-pytorch-fully-sharded-data-parallel-api.html) | [原网页](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) | 3 |
| 分布式训练 | [Maximizing Training Throughput Using PyTorch FSDP and Torch.compile](ML-Systems-and-Hardware/Distributed-Training/Community-Articles-20261122/pytorch/maximizing-training-throughput.html) | [原网页](https://pytorch.org/blog/maximizing-training-throughput/) | 2 |
| 分布式训练 | [Supercharging Training using float8 and FSDP2](ML-Systems-and-Hardware/Distributed-Training/Community-Articles-20261122/pytorch/training-using-float8-fsdp2.html) | [原网页](https://pytorch.org/blog/training-using-float8-fsdp2/) | 2 |
| 量化基础 | [Introduction to Quantization on PyTorch](Machine-Learning-Foundations/Quantization-and-Compression/Community-Articles-20261122/pytorch/introduction-to-quantization-on-pytorch.html) | [原网页](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) | 1 |
| 优化工具箱 | [PyTorch Native Architecture Optimization: torchao](ML-Systems-and-Hardware/ML-Frameworks/Community-Articles-20261122/pytorch/pytorch-native-architecture-optimization.html) | [原网页](https://pytorch.org/blog/pytorch-native-architecture-optimization/) | 5 |
| 推理Kernel与后端 | [Sorting-Free GPU Kernels for LLM Sampling](LLM-Systems/Attention-and-Kernels/Community-Articles-20261122/flashinfer/sampling.html) | [原网页](https://flashinfer.ai/2025/03/10/sampling.html) | 6 |
| 推理系统、编译与性能 | [LLM inference: How it works and how to optimize it](LLM-Systems/Performance-and-Profiling/Community-Articles-20261122/together-ai/llm-inference.html) | [原网页](https://www.together.ai/blog/llm-inference) | 0 |
| FlashAttention／GPU原理 | [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](ML-Systems-and-Hardware/GPU-Programming/Community-Articles-20261122/together-ai/flashattention-3.html) | [原网页](https://www.together.ai/blog/flashattention-3) | 10 |
| 推理系统、编译与性能 | [Last year we built megakernels. This year we deleted them: The road to Together Compiler](ML-Systems-and-Hardware/GPU-Programming/Community-Articles-20261122/together-ai/road-to-together-compiler.html) | [原网页](https://www.together.ai/blog/road-to-together-compiler) | 20 |
| 推理系统、编译与性能 | [Optimizing your LLM in production](LLM-Systems/Performance-and-Profiling/Community-Articles-20261122/hugging-face/optimize-llm.html) | [原网页](https://huggingface.co/blog/optimize-llm) | 10 |
| 缓存、调度与阶段分离 | [A First Comprehensive Study of TurboQuant: Accuracy and Performance](LLM-Systems/Serving-and-KV-Cache/Community-Articles-20261122/vllm/2026-05-11-turboquant.html) | [原网页](https://vllm.ai/blog/2026-05-11-turboquant) | 14 |
| 推测解码 | [How Speculative Decoding Boosts vLLM Performance by up to 2.8x](LLM-Systems/Speculative-Decoding/Community-Articles-20261122/vllm/2024-10-17-spec-decode.html) | [原网页](https://vllm.ai/blog/2024-10-17-spec-decode) | 10 |
| 推测解码 | [P-EAGLE: Faster LLM inference with Parallel Speculative Decoding in vLLM](LLM-Systems/Speculative-Decoding/Community-Articles-20261122/vllm/2026-03-13-p-eagle.html) | [原网页](https://vllm.ai/blog/2026-03-13-p-eagle) | 6 |

## 对应原论文

只补清单具名方法的原论文，不递归展开这些论文或网页的全部参考文献。TurboAttention与TurboQuant是不同论文；现有FA1／2／3、FlexAttention、PagedAttention等复用，不重复下载。

| 论文 | 本地PDF | 来源版本 | 页数 | 状态 |
| --- | --- | --- | ---: | --- |
| SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration | [PDF](ML-Systems-and-Hardware/GPU-Programming/2410.02367v9.pdf) | [2410.02367v9](https://arxiv.org/abs/2410.02367v9) | 21 | 新增 |
| TurboAttention: Efficient Attention Approximation For High Throughputs LLMs | [PDF](LLM-Systems/Attention-and-Kernels/2412.08585v3.pdf) | [2412.08585v3](https://arxiv.org/abs/2412.08585v3) | 18 | 新增 |
| FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling | [PDF](ML-Systems-and-Hardware/GPU-Programming/2603.05451v1.pdf) | [2603.05451v1](https://arxiv.org/abs/2603.05451v1) | 20 | 新增 |
| Flex Attention: A Programming Model for Generating Optimized Attention Kernels | [PDF](ML-Systems-and-Hardware/GPU-Programming/Dong%20%E7%AD%89%20-%202024%20-%20Flex%20Attention%20A%20Programming%20Model%20for%20Generating%20Optimized%20Attention%20Kernels.pdf) | [2412.05496v1](https://arxiv.org/abs/2412.05496v1) | 13 | 复用 |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | [PDF](ML-Systems-and-Hardware/GPU-Programming/Dao%20%E7%AD%89%20-%202022%20-%20FlashAttention%20Fast%20and%20Memory-Efficient%20Exact%20Attention%20with%20IO-Awareness.pdf) | [2205.14135v2](https://arxiv.org/abs/2205.14135v2) | 34 | 复用 |
| FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | [PDF](ML-Systems-and-Hardware/GPU-Programming/Dao%20-%202023%20-%20FlashAttention-2%20Faster%20Attention%20with%20Better%20Parallelism%20and%20Work%20Partitioning.pdf) | [2307.08691v1](https://arxiv.org/abs/2307.08691v1) | 14 | 复用 |
| FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | [PDF](ML-Systems-and-Hardware/GPU-Programming/Shah%20%E7%AD%89%20-%202024%20-%20FlashAttention-3%20Fast%20and%20Accurate%20Attention%20with%20Asynchrony%20and%20Low-precision.pdf) | [2407.08608v2](https://arxiv.org/abs/2407.08608v2) | 22 | 复用 |
| FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving | [PDF](LLM-Systems/Attention-and-Kernels/Ye%20%E7%AD%89%20-%202025%20-%20FlashInfer%20Efficient%20and%20Customizable%20Attention%20Engine%20for%20LLM%20Inference%20Serving.pdf) | [2501.01005v2](https://arxiv.org/abs/2501.01005v2) | 21 | 复用 |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | [PDF](LLM-Systems/Serving-and-KV-Cache/2309.06180v1.pdf) | [2309.06180v1](https://arxiv.org/abs/2309.06180v1) | 16 | 复用 |
| Online normalizer calculation for softmax | [PDF](ML-Systems-and-Hardware/GPU-Programming/1805.02867v2.pdf) | [1805.02867v2](https://arxiv.org/abs/1805.02867v2) | 9 | 复用 |
| SGLang: Efficient Execution of Structured Language Model Programs | [PDF](LLM-Systems/Serving-and-KV-Cache/Zheng%20%E7%AD%89%20-%202024%20-%20SGLang%20Efficient%20Execution%20of%20Structured%20Language%20Model%20Programs.pdf) | [2312.07104v2](https://arxiv.org/abs/2312.07104v2) | 20 | 复用 |
| TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate | [PDF](Machine-Learning-Foundations/Quantization-and-Compression/Zandieh%20%E7%AD%89%20-%202025%20-%20TurboQuant%20Online%20Vector%20Quantization%20with%20Near-optimal%20Distortion%20Rate.pdf) | [2504.19874v1](https://arxiv.org/abs/2504.19874v1) | 25 | 复用 |
| P-EAGLE: Parallel-Drafting EAGLE with Scalable Training | [PDF](LLM-Systems/Speculative-Decoding/2602.01469v1.pdf) | [2602.01469v1](https://arxiv.org/abs/2602.01469v1) | 13 | 新增 |

## 保存范围与实际验证

- 这是正文静态快照，不是网站运行时打包。正文图片、GIF、SVG、代码、表格和公式保留；视频／音频、互动Demo、任意外链页面或仓库仍需联网。没有执行文中的安装、模型、训练或CUDA命令。
- 231个代码块的文本与缩进逐项和来源核对；保留262个MathML公式及62个来源已渲染的SVG公式。不依赖远程MathJax脚本。
- 全47页在全新隔离浏览器中禁用来源脚本、阻断外网后通过：343处图片无坏图，12个文档提示图标可见。Unified Radix Cache的2张命中率图、6条曲线、图例、坐标轴及16个静态条形均可显示，曲线点和条宽与原始快照核对一致；所需静态样式来源记录在机器清单中。
- 全部1774处HTML本地引用（含1302处片段引用）、885处Markdown本地链接均通过；2段来源脚注正文与回链保留。除README／INDEX两份导航外，其余639个起点文件（含576份原有PDF）SHA-256不变。
- 47个选定正文都不是登录／404空壳，343个图片URL全部成功；图片内容按SHA-256去重。七份SVG仅移除指向外部DTD的声明，绘图数据不改，源内容哈希与转换记录在机器清单中。
- 4份新PDF已检查标题、摘要、引言及真实结论，16条短引文逐项核对；文件签名、页数、版本、SHA-256及qpdf检查通过。没有复现实验或验证历史教程在当前框架版本的兼容性。
- PyTorch和Together的FA3是两个托管版本，按两个明确来源保留，不把它们当两篇独立原论文。SageAttention中文长文按qingkeai页面原署名和链接保存，不据此认定博客作者就是论文作者。
- 旧的2026-11-05／11-19目录、manifest、Qwen资料及原ZIP保留。本轮仅更新当前导航和增加新资料；资料版权归原作者／发布方。
- 本轮没有未取得的正文或正文图片；未打包的视频、互动组件及外部页面属于上述离线范围边界。
