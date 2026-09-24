# 双 RTX 2080 Ti 22GB — vLLM 运行配方

本文档记录我如何在**两张魔改 22GB RTX 2080 Ti + NVLink** 上，用自编译的
`sm_75` 版 vLLM 跑起来一个 Qwen 3.8 27B 级别的模型。

这里公开的是**运行配置本身**：完整的启动参数、背后的 profile 文件、以及配套的
运维脚本（GPU 锁频/功耗调节）和基准测试方法。目的是给同样想在图灵架构老卡上跑
27B 级模型的人一个可复现的参照。

> **本仓库不包含任何模型权重**，不含量化文件、不含合并后的 checkpoint、不含
> HF token。权重请自备，见[模型](#模型)。

---

## 致谢，以及我做了什么

**这套运行环境建立在 [weicj](https://github.com/weicj) 的
[`vLLM-2080Ti-Definitive`](https://github.com/weicj/vLLM-2080Ti-Definitive)
项目之上。** 那个项目是地基：`sm_75` 构建工具链、launcher、profile 体系，
以及「在魔改显存的图灵卡上跑 vLLM」这条整体路线，都来自它。如果你想自己动手，
请从那里开始。

**我在此基础上叠加的**是一个更新的模型/运行时版本点，以及围绕它的配置级调优：

- **把技术栈往前推** —— `vLLM 0.2.1rc2`（基线 `0.27.1`）、PyTorch
  `2.13.0+cu130`、CUDA 13.0、transformers `5.15.1`，驱动 595.84。上游在撰写时
  指向的是基线 `0.21.0` / torch `2.11.0+cu128` / CUDA 12.8。
- **针对 230K 上下文 NVFP4 模型的新 profile 组** —— `profiles/` 下的
  `qwen3.8-27b` 系列是我加的；上游只有 `qwen27b` 和 `qwen35b`。
- **调优工作** —— 显式指定 `--kv-cache-memory-bytes`、FP8 KV cache、MTP `k=5`，
  以及把 chunked-prefill 预算一并纳入的 piecewise cudagraph 捕获列表
  `[6, 2048]`（而不是只捕获解码形状）。
- **运维配套** —— GPU 锁频/功耗调节脚本，以及本仓库的基准测试方法说明。

我机器上的 `vLLM-2080Ti-Definitive` 检出是一个追踪上游
`weicj/vLLM-2080Ti-Definitive` 的 git worktree；里面提交的作者是 weicj，不是我。
我不是那个项目的作者，也不做此声明。**本仓库中的一切都是我在其之上叠加的
自有配置与笔记。**

### 许可层面的后果

`vLLM-2080Ti-Definitive` 采用 **Apache-2.0**（vLLM 本身也是）。Apache-2.0 要求
衍生作品保留署名与许可声明。因此：

- 本仓库的 MIT 许可覆盖的是**我自己原创的内容**（为本仓库编写的 profile、
  笔记、governor 与基准脚本）。
- 该 MIT 授权**不会**重新许可 weicj 的项目或 vLLM。如果你复用他们的代码，
  仍然适用其 Apache-2.0 条款。
- 具体署名声明见 [NOTICE](NOTICE)。

---

## 硬件

| 项目 | 参数 |
| --- | --- |
| GPU | 2 × NVIDIA GeForce RTX 2080 Ti，**每张 22528 MiB（22GB）** — 魔改显存 |
| 互联 | NVLink 双路（`NV2`），单链路 25.781 GB/s |
| 拓扑 | 单 NUMA 节点，两张卡 CPU 亲和性均为 `0-7` |
| CPU | Intel Core i7-7740X @ 4.30GHz（4 核 8 线程） |
| 内存 | 15 GB |
| 系统 | Ubuntu 24.04.4 LTS，内核 7.0.0-31-generic |
| 驱动 | `nvidia-driver-595-open` 595.84，CUDA 运行时能力 13.2 |

几个关键点：

- **22GB 单卡显存是整个方案的前提。** 原厂 2080 Ti 只有 11GB。正是显存魔改让
  27B 级模型能跨两张卡装下，并留出可用的上下文长度。
- **NVLink 比额外的 PCIe 带宽重要得多。** TP=2 时每一层都要做 all-reduce，
  没有 NVLink 的话它会直接成为生成阶段的瓶颈。
- **i7-7740X 只有 4 个物理核心。** 这在长上下文下会实打实限制 tokenizer/
  detokenizer 吞吐，也直接影响了下面 `--max-num-batched-tokens` 和调度参数的选择。
- **15GB 系统内存**意味着模型不能以一种吃内存的方式在 host 侧被加载或 reshape，
  host 侧开销要压到最低。

## 版本组合

| 组件 | 版本 |
| --- | --- |
| vLLM | `0.2.1rc2`（自编译 `sm_75` 版本，来自 `vLLM-2080Ti-Definitive`） |
| 迁移到的 vLLM 基线 | `0.27.1` |
| PyTorch | `2.13.0+cu130` |
| CUDA | `13.0`（torch 构建版），驱动 595.84 |
| transformers | `5.15.1` |
| Python | `3.12.3` |

> 上游 `vLLM-2080Ti-Definitive` 在撰写时指向的是 vLLM 基线 `0.21.0`、
> torch `2.11.0+cu128` / CUDA 12.8、驱动 590.48.01。上表是本仓库记录的新版栈。

## 模型

本仓库的配置是针对以下权重调出来的，`--served-model-name` 里也写明了：

- **权重：** `orcarouter-Qwen3.8-27B-Uncensored-NVFP4`（27B 级别，
  NVFP4 / `compressed-tensors` 量化）

仅标注模型标识以便复现。**权重不在本仓库分发**，请从原作者的上游来源获取，
并把 `--model` 路径改成你自己的位置。

---

## 启动命令

这是线上实际使用的完整调用：

```bash
vllm serve /path/to/your/orcarouter-Qwen3.8-27B-Uncensored-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen38-27b-orcarouter-nvfp4-fp8kv-230K-mtp5-text-only-cu130 \
  --dtype half \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --generation-config vllm \
  --max-model-len 230000 \
  --enable-chunked-prefill \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  --kv-cache-memory-bytes 5115441742 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --language-model-only \
  --skip-mm-profiling \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[6,2048],"max_cudagraph_capture_size":2048}'
```

等价的脚本版本见 [`launch/serve.sh`](launch/serve.sh)。

### 每个参数为什么这么设

| 参数 | 理由 |
| --- | --- |
| `--tensor-parallel-size 2` | 把模型切到两张卡上。有 NVLink 才负担得起逐层 all-reduce。 |
| `--max-model-len 230000` | 230K 上下文。这是调出来的上限，再往上 KV cache + 激活值就装不进 2×22GB 了。 |
| `--kv-cache-dtype fp8` | KV cache 占用相对 `fp16` 减半，这是长上下文在这套硬件上成立的关键。 |
| `--kv-cache-memory-bytes 5115441742` | **显式**指定 KV cache 预算，不依赖 `--gpu-memory-utilization` 的启发式估算。把它钉死可以让每次重启的内存行为可复现，也避免过度预留。 |
| `--max-num-seqs 1` | 单并发序列。22GB/卡 + 230K 上下文的预算下，无法支撑有意义的 batch；优先保证一条长序列是刻意的取舍。 |
| `--max-num-batched-tokens 2048` | chunked prefill 的预算，与 cudagraph 捕获尺寸对齐，也照顾了孱弱的 4 核 CPU。 |
| `--enable-chunked-prefill` | 避免长 prefill 阻塞 decode，同时约束激活值内存峰值。 |
| `--quantization compressed-tensors` | 与 NVFP4 checkpoint 的量化方案匹配。 |
| `--mamba-cache-mode align` | 该模型家族的混合 Mamba/GDN 注意力结构要求此项。 |
| `--speculative-config … "mtp" … 5` | 多 token 预测（MTP），5 个投机 token。这是解码阶段最大的收益点，整个配置也是围绕 MTP 调的。 |
| `--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'` | 选择 legacy FlashQLA GDN prefill 后端，这是打完补丁后在 `sm_75` 上能跑通的那个。 |
| `--compilation-config … PIECEWISE` | `PIECEWISE` cudagraph 模式，捕获尺寸 `[6, 2048]`。捕获列表**必须同时包含 chunked-prefill 预算**（2048）和解码形状（6 = `MTP_K + 1`）——只列解码形状会让每次 prefill 都退回 eager，白白增加开销。 |
| `--language-model-only` + `--skip-mm-profiling` | 纯文本服务；跳过为用不到的多模态塔做的 profiling，省内存也省启动时间。 |
| `--enable-prefix-caching` | 复用共享前缀的 KV。 |
| `--dtype half` | 计算精度；权重本身仍是量化存储的。 |

`--served-model-name` 刻意把整套配置编码进去
（`…nvfp4-fp8kv-230K-mtp5-text-only-cu130`），这样客户端一眼就能看出这次请求是
哪个 profile 在服务。

---

## Profile 文件

[`profiles/`](profiles/) 存放驱动启动器的 profile 文件。每个 profile 就是一个
普通的 `KEY=VALUE` 环境变量文件，启动器负责把它翻译成 CLI 参数。把它们落成文件
（而不是散落在 shell history 里）是调参能复现、能 diff 的前提。

| Profile | 上下文 | MTP | KV 精度 | 量化 | 说明 |
| --- | --- | --- | --- | --- | --- |
| [`orcarouter-fp8kv-230K-mtp5-text-only.env`](profiles/orcarouter-fp8kv-230K-mtp5-text-only.env) | 230K | 5 | fp8 | compressed-tensors | **当前在跑的配置。** 纯文本。 |
| [`orcarouter-fp8kv-250K-mtp5-text-only.env`](profiles/orcarouter-fp8kv-250K-mtp5-text-only.env) | 250K | 5 | fp8 | compressed-tensors | 更长上下文，`gpu_util` 更低。 |
| [`orcarouter-fp8kv-200K-mtp5-text-only.env`](profiles/orcarouter-fp8kv-200K-mtp5-text-only.env) | 200K | 5 | fp8 | compressed-tensors | 余量更大的变体。 |
| [`fp8kv-240K-mtp3-text-image.env`](profiles/fp8kv-240K-mtp3-text-image.env) | 240K | 3 | fp8 | compressed-tensors | **文本 + 图像**，开启多模态。 |
| [`fp8kv-240K-nomtp-text-only.env`](profiles/fp8kv-240K-nomtp-text-only.env) | 240K | 0 | fp8 | compressed-tensors | 关闭 MTP —— 对照组。 |
| [`modelopt-w4a16-fp8kv-128K-mtp3-text-only.env`](profiles/modelopt-w4a16-fp8kv-128K-mtp3-text-only.env) | 128K | 3 | fp8 | `modelopt_mixed`（W4A16） | 另一条量化路线。 |

除了 CLI 参数之外，profile 里几个值得注意的环境变量：

- `VLLM_QWOPUS_MTP_BF16_DRAFT=1` —— MTP draft head 保持 BF16，换取稳定性。
- `VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=0` —— 刻意关闭；Mamba 投机的 full
  cudagraph 路径在这套环境上不稳定。
- `VLLM_TURBOQUANT_CONTINUATION_PREFIX_COMBINE=auto`，配合
  `…_MIN_TOKENS=20480` 和 `VLLM_TURBOQUANT_DECODE_BLOCK_KV=2`（230K profile）
  —— 针对「长前缀续写」类负载的调优。
- `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8` —— 降低长时间运行
  后的显存碎片。

### Profile 到参数的映射

启动器这段逻辑很短，核心是：

```bash
# 上下文 + KV 预算
VLLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
VLLM_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
VLLM_ARGS+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")

# 投机解码
if (( MTP_K > 0 )); then
  VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}")
fi

# 同时捕获解码形状（MTP_K+1）和 prefill 预算
capture=$((MTP_K + 1))
prefill_capture=${MAX_BATCHED_TOKENS:-2048}
(( prefill_capture < capture )) && prefill_capture=$capture
# -> cudagraph_capture_sizes: [capture, prefill_capture]  == [6, 2048]
```

---

## GPU 锁频与功耗调节

[`launch/vllm-clock-governor.sh`](launch/vllm-clock-governor.sh) 是一个把 GPU 状态
和推理活动绑定起来的小守护进程：

- 启动时给两张卡设 **200W 功耗墙**（卡本身允许到 280W；限功耗让魔改卡的温度和
  风扇噪音都保持在可控范围）。
- 通过轮询 metrics 里的 `vllm:num_requests_running`，**只在有请求时把 GPU0 的
  图形时钟锁在 1800 MHz**。
- 空闲 120 秒后释放锁，恢复动态调频。

这样做的目的是：生成期间时钟钉死（延迟可预期），其余时间让卡正常降频。脚本需要
sudoers 授权，因为它要调 `sudo -n nvidia-smi`：

```
<youruser> ALL=(root) NOPASSWD: /usr/bin/nvidia-smi
```

`POWER_LIMIT_WATTS`、`LOCK_RANGE`、`GPU_INDEX` 都可自行调整 —— 注意脚本按现在的
写法**只锁 GPU0**。

---

## 基准测试

[`benchmarks/`](benchmarks/) 存放调参期间用的测试脚本。它们作为**方法论**提供，
而不是已发布的结果 —— 请在自己的硬件上跑。路径已脱敏为 `<HOME>`。

| 脚本 | 测什么 |
| --- | --- |
| [`p6_bench.sh`](benchmarks/p6_bench.sh) | 流式跑 ~4K token 的 prompt；在首个 token 处切分，分别给出 **prefill tok/s**、**decode tok/s** 和 **TTFT**。 |
| [`p7_bench.py`](benchmarks/p7_bench.py) | 顺序、贪心、固定的一组互不相同 prompt，带 `ignore_eos`；输出单条与汇总 tok/s（含 prefill），结果追加到 summary 文件用于 A/B 对比。 |
| [`p12_bench.py`](benchmarks/p12_bench.py) | 非流式同域测试。跑两次（短生成 / 长生成）做差，抵消 prefill 从而单独测出 decode。 |
| [`bench_p4.py`](benchmarks/bench_p4.py) / [`bench_p4b.py`](benchmarks/bench_p4b.py) | 早期短 prompt 吞吐检查。 |

### 方法论要点

- **把 prefill 和 decode 分开看。** 笼统的「tok/s」会掩盖你究竟优化了哪一侧。
  `p6_bench.sh` 在 TTFT 处切分；`p12_bench.py` 用两次做差。
- **贪心 + `ignore_eos`**，保证不同配置之间 token 数可比。
- **固定但不重复的 prompt。** 反复用同一个 prompt 会让前缀缓存美化数据；p7 的
  prompt 集合刻意互不相同，以避免缓存命中。
- **单请求测量。** 在 `--max-num-seqs 1` 下这些是偏延迟的数字，不是服务端聚合
  吞吐。不要拿它们和批处理 serving benchmark 比较。

调用示例：

```bash
python3 benchmarks/p7_bench.py <label>
```

---

## 仓库结构

```
.
├── README.md                     # 英文说明
├── README.zh-CN.md               # 本文件
├── NOTICE                        # 上游署名声明
├── LICENSE                       # MIT（仅覆盖本仓库原创内容）
├── launch/
│   ├── serve.sh                  # 启动命令脚本版
│   └── vllm-clock-governor.sh    # GPU 锁频/功耗守护进程
├── profiles/                     # KEY=VALUE profile 文件
└── benchmarks/                   # 测试脚本 + 方法论
```

## 本仓库**不包含**什么

以下内容已刻意排除：

- **模型权重**及任何量化产物。
- **HF token、API key、SSH 私钥、密码** —— 一个都没有。路径和主机名已统一脱敏为
  `<HOME>` 或通用占位符。
- **性能宣传数字** —— 相比引用没有出处的数字，我更愿意之后给出受控测量下的结果。
- 内部 dashboard 与 watchdog 工具。

## 许可

本仓库**自有原创内容**（profile 文件、笔记、governor、基准脚本）按 MIT 许可提供，
仅作参考（见 [LICENSE](LICENSE)）。

该 MIT 授权**不覆盖**上游依赖：

- [`weicj/vLLM-2080Ti-Definitive`](https://github.com/weicj/vLLM-2080Ti-Definitive)
  —— **Apache-2.0**，本仓库的运行配置正建立在其之上。
- vLLM 本身 —— **Apache-2.0**。
- 模型权重 —— 遵循其上游作者自己的许可。

复用上述任一方的代码时，请遵守其各自的条款；完整署名见 [NOTICE](NOTICE)。
