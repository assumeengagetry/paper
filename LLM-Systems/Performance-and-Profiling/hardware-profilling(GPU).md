## 一

1 Men Access 

- On-chip

- Off-chip

2 Irregularity -> how irregular algorithms can
be efficiently mapped to GPUs

     6.2 Irregularity
    │
    │  核心矛盾：
    │  GPU = highly regular architecture
    │  算法 = often irregular
    │
    └── 目标：
        把 irregular algorithm
              ↓
        映射成 GPU 更容易高效执行的形式
        │
        ├── 6.2.1 Loop Unrolling
        │      │
        │      ├── 做什么？
        │      │   展开循环，减少循环控制开销
        │      │
        │      ├── 为什么有效？
        │      │   ├── 减少 branch / address calculation
        │      │   ├── 增加 ILP
        │      │   └── 暴露更多编译器优化机会
        │      │
        │      └── 风险
        │          └── unroll factor 太大 → 性能下降
        │
        ├── 6.2.2 Reduce Branch-Divergence
        │      │
        │      ├── 问题
        │      │   Warp lock-step execution
        │      │          ↓
        │      │   不同线程走不同分支
        │      │          ↓
        │      │   两个分支都执行
        │      │
        │      ├── 直接减少 Branch
        │      │   ├── 删除 branch
        │      │   ├── Arithmetic replacement
        │      │   ├── Branch refactoring
        │      │   └── Algorithm flattening
        │      │
        │      ├── 让 Branch 更规则
        │      │   ├── Lookup Table
        │      │   ├── Loop unrolling
        │      │   ├── Iteration delaying
        │      │   └── Kernel fission
        │      │
        │      ├── 用其他代价换 Branch
        │      │   ├── Redundant work
        │      │   ├── Approximation / errors
        │      │   └── Serial execution
        │      │
        │      └── 改变线程/数据组织
        │          ├── Thread/data remapping
        │          ├── Stream compaction
        │          ├── Sorting
        │          ├── Padding
        │          └── Sparse formats
        │
        ├── 6.2.3 Sparse Matrix Format
        │      │
        │      ├── 问题
        │      │   Sparse Matrix
        │      │       ↓
        │      │   每行非零元素数量不同
        │      │       ↓
        │      │   GPU 不规则
        │      │
        │      ├── 核心思想
        │      │   改变数据表示方式
        │      │       ↓
        │      │   把 irregular data
        │      │   变得 regular
        │      │
        │      ├── ELL / ELLPACK
        │      │   └── Padding → 等长
        │      │
        │      ├── 其他 GPU-friendly formats
        │      │   ├── ELL-R
        │      │   ├── SELL
        │      │   ├── CRSD
        │      │   ├── ASCR
        │      │   ├── SIC-CSR
        │      │   └── ...
        │      │
        │      ├── Hybrid Formats
        │      │   ├── HYB
        │      │   ├── ELL + CSR
        │      │   └── 不同区域使用不同格式
        │      │
        │      ├── Compression
        │      │   ├── Index compression
        │      │   ├── Delta encoding
        │      │   └── Bit flags
        │      │
        │      └── Dynamic formats
        │          └── 运行时允许增删数据
        │
        ├── 6.2.4 Kernel Fission
        │      │
        │      ├── 做什么？
        │      │   一个复杂 Kernel
        │      │       ↓
        │      │   拆成多个简单 Kernel
        │      │
        │      ├── 与 Kernel Fusion 相反
        │      │
        │      ├── 为什么有效？
        │      │   ├── Kernel 更简单
        │      │   ├── Kernel 更规则
        │      │   ├── 资源利用更好
        │      │   ├── 减少 branch divergence
        │      │   └── 帮助 autotuning
        │      │
        │      └── 应用
        │          ├── Sparse Matrix
        │          ├── Dense Matrix
        │          ├── Complex Kernel
        │          └── Autotuning
        │
        └── 6.2.5 Synchronization-related Balancing
               │
               ├── 核心思想
               │   减少 redundant work
               │
               ├── 为什么？
               │   不规则数据
               │       ↓
               │   不同线程工作量不同
               │       ↓
               │   同步 / 等待
               │       ↓
               │   一部分线程已经完成
               │       ↓
               │   继续做无意义工作
               │
               └── 优化目标
                   └── 避免不必要的重复计算

3 Balancing

1. Balancing thr instrution stream

2. parallelism related Balancing

3. Synchronization realated Balancing

     6.3 Balancing
    │
    │ 核心问题：
    │ GPU 有很多相互关联的硬件资源
    │        ↓
    │ 如何把并行计算映射到硬件
    │ 如何平衡资源使用
    │
    ├── A. Balancing the Instruction Stream
    │
    │   ├── 6.3.1 Vectorization
    │   │      ↓
    │   │   用 Vector 指令替代 Scalar 指令
    │   │      ↓
    │   │   一次处理多个数据
    │   │
    │   ├── 6.3.2 Fast Math Functions
    │   │      ↓
    │   │   用近似数学函数换取速度
    │   │      ↓
    │   │   Special Function Units
    │   │
    │   └── 6.3.3 Warp-Centric Programming
    │          ↓
    │       把 Warp 当成基本计算单位
    │          ↓
    │       减少同步开销 / 隐藏延迟 / 负载均衡
    │
    │
    ├── B. Parallelism-related Balancing
    │
    │   ├── 6.3.4 Varying Work per Thread
    │   │      ↓
    │   │   改变“一个线程干多少活”
    │   │      ↓
    │   │   增加数据复用
    │   │      ↕
    │   │   增加 Register / Shared Memory 消耗
    │   │
    │   ├── 6.3.5 Resize Thread Blocks
    │   │      ↓
    │   │   改变一个 Thread Block 有多少线程
    │   │      ↓
    │   │   影响：
    │   │      ├── Register usage
    │   │      ├── Shared Memory
    │   │      ├── Blocks / SM
    │   │      └── Occupancy / 并行度
    │   │
    │   ├── 6.3.6 Auto-tuning
    │   │      ↓
    │   │   自动搜索参数空间
    │   │      ↓
    │   │   找到性能最好的配置
    │   │
    │   └── 6.3.7 Load Balancing
    │          ↓
    │       让不同线程 / Warp / Block
    │       尽可能做相近数量的有效工作
    │
    │
    └── C. Synchronization-related Balancing
        │
        ├── 6.3.8 Reduce Synchronization
        │      ↓
        │   减少 Barrier / Synchronization
        │      ↓
        │   降低等待开销
        │
        ├── 6.3.9 Reduce Atomics
        │      ↓
        │   减少 Atomic Operations
        │      ↓
        │   降低同步带来的开销
        │
        └── 6.3.10 Inter-Block Synchronization
               ↓
            解决 Thread Block 之间如何同步
               ↓
            Cooperative Groups / Global Synchronization

4 Host Interaction->interplay between GPU as an accelerator and the host

     6.4 Host Interaction
    │
    │ 核心问题：
    │ GPU Kernel 本身优化得很好
    │        ↓
    │ 但整个 Application 仍可能很慢
    │        ↓
    │ 原因：
    │   ├── CPU ↔ GPU 通信慢
    │   └── CPU / GPU 计算分配不合理
    │
    ├── 6.4.1 Host Communication
    │      │
    │      │ 核心问题：
    │      │ CPU ↔ GPU 数据传输
    │      │        ↓
    │      │ PCIe 通信成为瓶颈
    │      │
    │      ├── ① 消除通信
    │      │      ├── 尽可能全部在 GPU 上计算
    │      │      └── 尽可能让数据留在 GPU
    │      │
    │      ├── ② 减少通信量
    │      │      └── 压缩传输数据
    │      │
    │      ├── ③ 把 Host 控制逻辑移到 GPU
    │      │      └── Dynamic Parallelism
    │      │
    │      ├── ④ 简化 CPU/GPU 内存视图
    │      │      └── Unified Memory View
    │      │
    │      ├── ⑤ 优化内存
    │      │      └── Pinned Memory
    │      │
    │      ├── ⑥ 通信与计算重叠
    │      │      ├── Pinned / Mapped Memory
    │      │      ├── Streams
    │      │      └── Command Queues
    │      │
    │      ├── ⑦ Pipeline
    │      │      └── 把通信和计算流水化
    │      │
    │      └── ⑧ Buffer Management
    │             ├── Double Buffering
    │             └── Triple Buffering
    │
    │
    └── 6.4.2 CPU/GPU Computation
           │
           │ 核心问题：
           │ CPU 和 GPU 谁来干什么？
           │
           ├── CPU + GPU 同时计算
           │
           ├── 把计算拆分
           │      ├── CPU
           │      └── GPU
           │
           ├── 适合场景
           │      └── Independent / Semi-independent Tasks
           │
           ├── 如果存在依赖
           │      ↓
           │   数据传输优化变得重要
           │
           └── Load Balancing
                  ↓
               CPU / GPU 工作量如何平衡

## 二

1. analysis of the Optimization based on application specifices

![](/home/assumeengage/.config/marktext/images/2026-08-29-22-15-36-image.png)

![](/home/assumeengage/.config/marktext/images/2026-08-29-22-15-45-image.png)

![](/home/assumeengage/.config/marktext/images/2026-08-29-22-16-13-image.png)

![](/home/assumeengage/.config/marktext/images/2026-08-29-22-18-19-image.png)

![](/home/assumeengage/.config/marktext/images/2026-08-29-22-16-31-image.png)

![](/home/assumeengage/.config/marktext/images/2026-08-29-22-16-37-image.png)

2.analysis of the Optimization based on bottlenecks



![](/home/assumeengage/.config/marktext/images/2026-08-29-22-21-15-image.png)

![](/home/assumeengage/.config/marktext/images/2026-08-29-22-21-26-image.png)

![](/home/assumeengage/.config/marktext/images/2026-08-29-22-21-38-image.png)

![](/home/assumeengage/.config/marktext/images/2026-08-29-22-21-48-image.png)
