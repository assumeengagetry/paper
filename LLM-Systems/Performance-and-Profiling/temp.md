# VLA 大小 架构 MoE

自回归 0.3B 大小 TPOP TFTP 推理范式 只拿推理范式 real-time 机械控制频率 30ms 50ms 推理600700-300ms 调度优化 



# 四、目前学术界如何解决 VLA 实时推理问题

## 4.1 轻量模型和更高效的动作表示

代表方向包括：

- TinyVLA、SmolVLA、RoboMamba；
- FAST/VQ 类动作 tokenization；
- 更小的视觉 backbone；
- 更小的 Action Expert；
- 直接减少 action token 数量。

**优点**

- 每次推理都更便宜；
- 显存占用更低；
- 不依赖场景连续性；
- 对所有 timestep 都生效。

**缺点**

- 通常需要重新训练或蒸馏；
- 容量降低可能损失开放世界泛化和长程推理；
- 一种动作 tokenizer 可能不适合所有机器人和控制频率；
- 参数少不保证速度快，仍可能被大量 diffusion steps 拖慢。

OpenVLA 和 π0 分别体现了离散 action token 与连续 flow-matching Action Expert 两条主要路线。

---

## 4.2 量化、剪枝、蒸馏和 consistency training

**优点**

- 量化可同时减少显存和带宽；
- 蒸馏/consistency 可以直接把几十步压缩为几步；
- 理论上能从根源上减少迭代次数。

**缺点**

- 量化并不一定自动加速，小 batch、小矩阵时低比特 kernel 未必高效；
- 动作控制对数值误差可能比语言输出更敏感；
- 蒸馏需要额外训练、数据和 teacher 推理；
- 低步数策略可能损失动作分布的多模态性；
- 量化还会改变 speculative decoding 的 acceptance rate。

论文自己的 OpenVLA 实验就是典型例子：

- 4-bit 量化有一定加速；
- 但平均成功率下降明显；
- speculative + cache 虽然更快，误差却会叠加；
- 量化产生的数值偏差还可能降低 draft acceptance。

这说明：

> **两个方法在代码路径上正交，不等于它们在误差空间中正交。**

---

## 4.3 时间缓存和动态计算

例如 VLA-Cache、adaptive caching、ActionCache 等方法会检测：

- 相邻帧视觉 token 是否变化；
- 某些 layer feature 是否稳定；
- 当前 action chunk 是否仍可复用；
- diffusion 中间状态是否需要重新计算。

**优点**

- 往往 training-free；
- 对连续机器人视频非常自然；
- 不需要改变基础模型；
- 可与量化、编译叠加。

**缺点**

- 接触发生、物体被遮挡、相机快速移动时，时间连续性会突然失效；
- 阈值需要按模型和任务调；
- 错误可能跨多个控制周期累积；
- 平均 benchmark 相似度不能代替安全性保证。

VLA-Cache 和后续 adaptive cache 工作都在探索如何将固定缓存变为输入相关的动态缓存。

---

## 4.4 Speculative inference

思路是：

1. 用更便宜的模型、较浅的网络或历史动作提出 candidate；
2. 用目标模型验证；
3. 成功时一次接受多个 action token 或 diffusion steps。

**优点**

- 可以跳过完整大模型调用；
- 不一定需要修改 target model；
- 当 draft 很便宜且 acceptance rate 高时收益大。

**缺点**

- 需要额外 draft；
- draft 占显存；
- acceptance rate 很依赖任务、精度和数值实现；
- 对只有 4 个 Action Expert steps 的模型，验证成本可能抵消收益；
- 在控制任务中，“数值接近”不一定意味着“动力学和安全性等价”。

Spec-VLA、HeiSD 和一些更近期的 VLA speculative/action-head 方法都属于这条路线。

---

## 4.5 异步控制和流水线

Real-Time Chunking、Running VLAs、V-AEFusion 等方向不一定减少 FLOPs，而是：

- 推理下一段动作时继续执行上一段；
- 平滑拼接 action chunks；
- 将视觉推理与动作生成重叠；
- 使用 delayed/stale features 隐藏延迟。

**优点**

- 对大模型非常有效；
- 不必牺牲模型规模；
- 可以改善机器人停顿和动作抖动。

**缺点**

- 推理与执行之间产生 delay；
- 当前动作可能基于过时观测；
- 需要处理 chunk 边界和轨迹连续性；
- 极端情况下会把“模型延迟”变成“控制延迟”；
- 资源不够时并行反而造成竞争。

---

## 4.6 编译器、kernel 和专用 runtime

例如：

- `torch.compile`；
- vendor graph compiler；
- TensorRT/ATC；
- 专用 VLA runtime；
- fused attention、fused MLP；
- 量化 GEMM；
- 跨 CPU/GPU/NPU 调度。

**优点**

- 通常是接近无损的优化；
- 可与算法级方法叠加；
- 能消除大量框架开销；
- 对 edge device 尤其重要。

**缺点**

- 高度依赖硬件和软件版本；
- 动态 shape、控制流和不支持算子会导致 graph break；
- 不能消除 diffusion 本身的串行依赖；
- benchmark 可能更多反映编译器成熟度，而非硬件本身。

VLA-Perf 和 vla.cpp 一类工作正在把 VLA 当成独立 workload 做系统化 profiling 和 runtime 优化。

```markdown
推演 3：用 profiler 和 Roofline 将现象分解为两个阶段

结果发现两种串行化：

pipeline-level serialization：VLM 必须先于 Action Expert；
component-level serialization：Action Expert 内部要逐步 denoise。
推演 4：分别解决两种串行化
对内部 denoising：借鉴视频 diffusion 的 PAB、TeaCache，做 DP-Cache；
对 VLM→AE 串行：借鉴时间缓存和 pipeline execution，让 AE 使用旧 feature 提前启动，形成 V-AEFusion。

PAB、TeaCache 和 VLA temporal caching 是最明显的背景来源。
```

# 十三、目前学术界如何解决大型 diffusion 推理问题

## 13.1 更好的 numerical solver

例如 DPM-Solver、DPM-Solver++。

核心是用更高阶 ODE/SDE solver，在更少的 neural function evaluations 下求解 diffusion trajectory。

**优点**

- training-free；
- 不需要额外模型；
- 实现相对简单；
- 通常可直接从 50–100 步降到 10–20 步。

**缺点**

- 极低步数下质量会下降；
- classifier-free guidance 较大时可能不稳定；
- 不同 noise schedule 和模型需要调整；
- 对已经采用少步 flow solver 的模型，空间有限。

---

## 13.2 Distillation 和 Consistency Models

把多步 teacher 蒸馏成：

- 1-step；
- 2-step；
- 4-step；

student。

**优点**

- 减少的 target calls 最多；
- 部署时不需要 draft 和 verification；
- 对实时生成最有潜力。

**缺点**

- 需要昂贵训练；
- 要访问 teacher 和大量数据；
- 可能降低 diversity、prompt adherence 或细节；
- 新模型、新分辨率通常需要重新蒸馏。

---

## 13.3 Cache 和 step skipping

代表包括 DeepCache、PAB、TeaCache。

- PAB 固定或分层复用 attention；
- TeaCache 用 timestep embedding 或输入变化预测何时可以复用；
- DeepCache 复用网络中间 feature。

**优点**

- training-free；
- 不需要额外模型；
- 内存开销较小；
- 很适合相邻 diffusion steps 高度相似的情况。

**缺点**

- 使用的是旧 target 输出，而不是更精确的 draft 预测；
- 错误可能逐步累积；
- threshold 和 interval 依赖模型；
- 激进缓存容易改变对象位置、颜色、身份和视频运动。

---

## 13.4 稀疏 attention、token pruning 和量化

**优点**

- 减少每一步 FLOPs；
- 可与减少步数的方法叠加；
- 视频 token 数量巨大，因此理论收益高。

**缺点**

- 需要模型结构或 kernel 配合；
- 稀疏模式未必适用于所有 prompt；
- GPU 上理论 FLOPs 减少不一定转化为 wall-time；
- 量化 draft 在 A800 等硬件上未必有高效低比特 kernel，ASDSV 图像实验就出现 FLOPs 很低、实际延迟却不最低的情况。

---

## 13.5 多 GPU 并行

PipeFusion、DistriFusion、xDiT 等将：

- spatial patches；
- temporal segments；
- diffusion stages；
- classifier-free guidance branches；

分布到多 GPU。

**优点**

- 可以接近无损；
- 对超高分辨率和长视频有效；
- 与 cache/speculation 正交。

**缺点**

- 需要多卡；
- 通信和同步昂贵；
- 总计算量通常没有减少；
- 不适合单卡和边缘部署。

---

## 13.6 Exact speculative diffusion

De Bortoli 等工作尝试将具有分布修正的 speculative sampling 扩展到 diffusion，使最终样本仍服从 target process。后续还有 exchangeability、block verification 等方向。

**优点**

- 理论上保持 target distribution；
- 不需要用 FID 或 VBench 为 approximation 辩护；
- 概念上与 LLM speculative decoding 更一致。

**缺点**

- 连续高维 residual sampling 难；
- 验证中间 states 很贵；
- 高分辨率 DiT 上显存和 batch 成本高；
- 当前实际加速通常不如近似方法激进。

---

## 13.7 Approximate speculative diffusion

ASDSV、SpeCa 和部分 feature-level relaxed verification 工作属于这一方向。

**优点**

- 更容易在 FLUX/Wan 等大模型上落地；
- verification 可以极度简化；
- 对视频可获得明显 speedup；
- 不需要重新训练 target。

**缺点**

- 不保持精确 target distribution；
- 需要 draft model；
- 需要阈值和 stage 参数；
- 可能 false accept；
- draft/target pair 换掉后需要重新 profiling。





- Behavior Cloning；
- ACT；
- Diffusion Policy；
- Transformer Policy；















## 2. 多模态推理优化

多模态模型比 LLM 多出视觉/音频路径：

```
图像/视频→ Decode/Resize→ Vision Encoder→ Visual Tokens→ Projector/Cross-Attention→ LLM→ 输出
```

除了 LLM 的问题，还会出现：

- 图像分辨率动态变化；
- 视频帧数差异很大；
- visual token 数量多；
- 视觉 encoder 和 LLM 性能不平衡；
- CPU 图像预处理可能成为瓶颈；
- 视频相邻帧存在大量重复；
- 不同请求的模态组成不同，难以 batching；
- image features、visual KV 是否缓存；
- vision encoder、LLM 是否分设备部署。

典型优化包括：

### 算法层

- visual token pruning；
- frame sampling；
- temporal caching；
- 跳过低重要度视觉层；
- 动态分辨率；
- 视觉特征复用。

### Kernel层

- variable-length attention；
- vision attention；
- patch embedding；
- image resize/preprocess fusion；
- multimodal projector fusion。

### Runtime/Serving层

- 图像预处理流水线；
- vision encoder 独立 batching；
- modality-aware scheduler；
- vision feature cache；
- encoder和LLM分离部署；
- CPU/GPU preprocessing overlap。

因此，多模态推理优化不是简单“在 LLM 前面加一个 ViT”，而是产生了新的异构 pipeline 和动态 workload。







# 四、Workload Profiling 和 Hardware Profiling 分别是什么？

这是“做过算子优化”和“能不能做 Kernel Agent”之间最关键的桥梁。

## 1. Workload Profiling：决定优化什么

它关注真实应用中的计算分布：

- 哪些operator占用时间最多；
- 每个operator调用多少次；
- shape分布；
- batch size；
- sequence length；
- dtype；
- layout和stride；
- 动态shape比例；
- fusion机会；
- launch overhead；
- 数据搬运；
- 端到端Amdahl收益。

例如：

```
一个GEMM加速3倍但只占模型时间2%
```

整体加速只有：

S=(1−0.02)+0.02/31​≈1.014

只有约1.4%。

因此 workload profiling 解决的是：

> 哪个 kernel 值得优化，以及应该为哪些 shape 优化。

在不同领域，它观察的内容也不一样：

| Workload | 关键分布                                   |
| -------- | -------------------------------------- |
| LLM      | prompt length、decode length、batch、KV长度 |
| 多模态      | 图像数、分辨率、视觉token、视频帧数                   |
| VLA      | 相机数、action chunk、denoising steps、控制频率  |
| RL训练     | environment数量、rollout长度、sim/render比例   |

## 2. Hardware Profiling：解释为什么慢

典型工具：

- Nsight Systems；
- Nsight Compute；
- rocprof；
- CUPTI；
- 编译器ptxas报告；
- Roofline分析。

关注指标：

- SM active；
- Tensor Core utilization；
- DRAM bandwidth；
- L2 hit rate；
- warp stall reason；
- achieved occupancy；
- register数量；
- shared-memory占用；
- bank conflict；
- instruction mix；
- kernel launch间隙；
- memory coalescing；
- PCIe/NVLink传输。

关键不是看到“occupancy低”，而是建立因果链：

| Profile信号             | 可能诊断                 | 可能优化                  |
| --------------------- | -------------------- | --------------------- |
| kernel很短、SM空洞多        | launch-bound         | fusion、CUDA Graph     |
| DRAM接近峰值、计算低          | memory-bound         | fusion、缓存、减少写回        |
| Tensor Core利用率低       | tile/layout/shape不合适 | 调tile、padding、MMA     |
| register过多、occupancy低 | tile或pipeline过深      | 减tile、减少stage         |
| block数不足              | grid并行度不够            | split-K、persistent策略  |
| 单kernel变快但端到端没变       | 优化了错误热点              | 重新做workload profiling |

## 3. 真正的优化经验

不是“看过很多 profile”，而是形成了：

(workload shape,hardware signal)→瓶颈假设→代码变换→性能验证

所以别人问：

> “你做过算子优化，这跟现在 Kernel Agent 有什么不同？是不是有很多 workload/hardware profiling 经验？”

实际是在判断：

- 你是只优化过一个固定kernel；
- 还是能跨workload识别性能模式；
- 能否把profile转换成优化策略；
- 这些策略能否整理成Agent可调用的知识。















# 七、在 vLLM 上做多级 KV Cache 属于什么？

它主要属于：

> LLM inference runtime/framework + memory system。

不是纯算子优化。

## 典型多级结构

```
L1：GPU HBML2：CPU DRAML3：本地NVMeL4：远端DRAM/SSD Cache Pool
```

热的 KV block 放在 HBM，冷的逐级下沉。

## Prefetch

在请求真正进入 GPU 计算之前预测它需要的 KV blocks：

```
Scheduler准备执行请求→ 查Prefix/Block Metadata→ 从DRAM/NVMe/Remote预取→ Copy Stream异步搬到HBM→ 与其他Kernel重叠→ 请求开始执行
```

困难在于：

- 预取太早会占用HBM；
- 太晚会阻塞decode；
- 错误预取浪费PCIe/NVLink带宽；
- 必须和request scheduler协同。

## Eviction

HBM空间不足时选择哪些KV block移出。

简单策略：

- LRU；
- LFU。

更好的策略会考虑：

value≈block size+transfer costP(reuse)×recompute cost​

还要考虑：

- prefix共享次数；
- request是否仍在decode；
- block是否被多个请求引用；
- 重新prefill的代价；
- 远端cache传输时间；
- SLO优先级。

## 系统实现需要做什么？

- KV block allocator；
- block table；
- prefix tree/hash；
- reference counting；
- tier metadata；
- asynchronous DMA/RDMA；
- prefetch queue；
- eviction policy；
- scheduler integration；
- cache consistency；
- admission control；
- failure recovery；
- metrics和profile。

关键评测指标：

- cache hit rate；
- TTFT；
- TPOT；
- P99 latency；
- HBM占用；
- PCIe/NVLink流量；
- recompute减少量；
- overall request throughput。

如果只是给 vLLM 接入一个新模型，这是框架支持。

如果设计了新的：

- cache abstraction；
- scheduling policy；
- tier placement；
- prefetch/eviction算法；
- hardware-aware data path；

并证明其在真实trace上改善SLO和吞吐，那就是推理系统研究。

Mooncake就是更大规模的例子：它把KV Cache扩展到分布式CPU DRAM/SSD池，并与prefill-decode分离和全局调度结合。[Mooncake](https://arxiv.org/abs/2407.00079?utm_source=chatgpt.com)

---

# 八、“框架搭建”和“系统研究”怎么区分？

二者不是看代码是不是写在 vLLM 里，而是看有没有新的系统问题和机制。

## 偏工程接入

- 加模型类；
- 注册runner；
- 实现已有接口；
- 修兼容性；
- 跟随现有scheduler路径；
- 没有新的性能机制。

## 偏系统研究

- 识别真实workload中的新瓶颈；
- 提出新的资源抽象；
- 修改scheduler/cache/runtime；
- 与硬件特征结合；
- 给出可解释的design；
- 在不同负载、硬件、SLO下验证；
- 有明确的trade-off和ablation。

例如：

> “在 vLLM 上实现多级 KV Cache”

本身只能说明框架工作。

如果进一步提出：

> “根据prefix未来复用概率和PCIe传输代价做cost-aware prefetch/eviction，并让scheduler在KV未就绪时调度其他请求”

这就成为一个完整的推理系统问题。

vLLM不会帮你训练这个模型，它接管训练完成后的推理执行：

- Continuous Batching；
- PagedAttention；
- KV Block Manager；
- Prefix Cache；
- Chunked Prefill；
- CUDA Graph；
- 多GPU推理。

Mooncake又在vLLM之上或旁边处理更大的集群级问题：

- Prefill/Decode分离；
- 跨GPU、跨节点传输KV；
- DRAM/SSD多级KV Cache；
- 跨推理实例共享KV；
- KV-aware scheduling。
