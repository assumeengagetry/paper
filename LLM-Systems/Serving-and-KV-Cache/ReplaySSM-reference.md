# ReplaySSM: Cache SSM Inputs, Not State

资料类型：公开技术文章与代码项目，**不计入论文 PDF 数量**。

- 作者站点：[Tri Dao — ReplaySSM](https://tridao.me/blog/2026/replayssm/)
- 代码：[Johnny-Liou/ReplaySSM](https://github.com/Johnny-Liou/ReplaySSM)
- 核查日期：2026-11-05。
- 本地来源：`storage.zip` 中 `storage/P8R55566/replayssm.html` 的网页快照；原始压缩包保留，不提交 Zotero 阅读状态或缓存。

## 内容与归类

该方法针对混合注意力／状态空间模型的解码。常规实现逐 token 更新并写回完整 recurrent state；ReplaySSM 改为缓存近期输入，在需要时重建或刷新状态，降低状态内存流量，并让投机解码中的回滚变为缓冲区指针操作。文章还讨论 output-only 解码、验证阶段并行化以及 vLLM 中的 kernel 实现。

其主要贡献是模型推理时的状态管理与解码执行，因此放在 `LLM-Systems/Serving-and-KV-Cache/`。本次核对的文章提供技术正文和代码链接，没有将其网页打印或改名为独立研究论文 PDF；后续如有明确原论文，应另行核对标题、作者和公开来源。
