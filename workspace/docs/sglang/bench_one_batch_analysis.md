# SGLang bench_one_batch 内部执行流程详细分析

## 概述

`python -m sglang.bench_one_batch` 是 SGLang 的一个**低级别性能测试工具**，它**不启动服务器**，而是直接使用底层 API 来测试单个静态批次的延迟性能。这个测试完全在本地进行，没有网络通信。

## 问题 1：测试输入数据是什么？

### 数据生成代码
在 `/iopsstor/scratch/cscs/xjin/repos/sglang/python/sglang/bench_one_batch.py` 第 211-229 行：

```python
def prepare_synthetic_inputs_for_latency_test(batch_size, input_len):
    input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.prefix_indices = []
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)

    return reqs
```

### 输入数据特点
- **完全合成数据**：使用 `np.random.randint(0, 10000, (batch_size, input_len))` 生成
- **batch_size = 2**：创建 2 个请求
- **input_len = 2048**：每个请求包含 2048 个随机 token ID
- **token ID 范围**：0-9999 之间的随机整数
- **采样参数**：温度为 0（确定性生成），最大新 token 数为 2048

## 问题 2：Prefill 和 Decode 日志详细解析

### Prefill 阶段输出
```
Prefill. latency: 0.17758 s, throughput:  23065.25 token/s
```

**打印位置**：第 382-386 行
```python
# Prefill
synchronize(device)
tic = time.perf_counter()
next_token_ids, _, batch = extend(reqs, model_runner)
synchronize(device)
prefill_latency = time.perf_counter() - tic
tot_latency += prefill_latency
throughput = input_len * batch_size / prefill_latency
rank_print(
    f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
)
```

**数值含义**：
- `latency: 0.17758 s`：处理所有输入 token（2048 × 2 = 4096 个）的总时间
- `throughput: 23065.25 token/s`：= 4096 tokens ÷ 0.17758 秒
- **Prefill 是并行处理**：所有输入 token 在一次前向传播中同时处理

### Decode 阶段输出
```
Decode 0. Batch size: 2, latency: 5.70161 s, throughput:      0.35 token/s
Decode 1. Batch size: 2, latency: 0.00244 s, throughput:    818.76 token/s
```

**打印位置**：第 389-401 行
```python
# Decode
decode_latencies = []
for i in range(output_len - 1):
    synchronize(device)
    tic = time.perf_counter()
    next_token_ids, _ = decode(next_token_ids, batch, model_runner)
    synchronize(device)
    latency = time.perf_counter() - tic
    tot_latency += latency
    throughput = batch_size / latency
    decode_latencies.append(latency)
    if i < 5 or (log_decode_step > 0 and i % log_decode_step == 0):
        rank_print(
            f"Decode {i}. Batch size: {batch_size}, latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
        )
```

**数值含义**：
- `Decode 0`：第 1 次生成新 token
  - `latency: 5.70161 s`：第一次 decode 包含 CUDA graph 编译时间，所以特别慢
  - `throughput: 0.35 token/s`：= 2 tokens ÷ 5.70161 秒
- `Decode 1`：第 2 次生成新 token  
  - `latency: 0.00244 s`：CUDA graph 已编译，使用缓存的计算图
  - `throughput: 818.76 token/s`：= 2 tokens ÷ 0.00244 秒

**Decode i 的含义**：
- `i` 是生成 token 的序号（从 0 开始）
- 总共会有 `output_len - 1 = 2047` 次 decode 步骤
- **打印条件**：只有前 5 次（i < 5）会被打印，之后不再打印单步延迟

## 问题 3：Warmup 机制详解

### Warmup 代码位置
第 456-467 行：
```python
# Warm up
rank_print("Warmup ...")
latency_test_run_once(
    bench_args.run_name,
    model_runner,
    rank_print,
    reqs,
    bench_args.batch_size[0],      # batch_size = 2
    bench_args.input_len[0],       # input_len = 2048
    min(32, bench_args.output_len[0]),  # output_len = min(32, 2048) = 32
    server_args.device,
    log_decode_step=0,
    profile=False,
    profile_filename_prefix="",
)
```

### Warmup 目的和作用
1. **CUDA 内核预热**：第一次运行时 CUDA 需要编译和优化内核
2. **CUDA Graph 编译**：SGLang 使用 CUDA Graph 优化，需要预先捕获计算图
3. **内存分配**：预先分配和初始化 GPU 内存池
4. **缓存预热**：初始化各种缓存结构

### Warmup 参数说明
- **使用相同的 batch_size 和 input_len**：确保 warmup 和实际测试使用相同的内存布局
- **较短的 output_len (32)**：Warmup 只需要激活所有代码路径，不需要完整生成
- **log_decode_step=0**：Warmup 期间不打印详细的 decode 步骤

## 问题 4：数据处理流程（非服务器模式）

### 重要说明
**bench_one_batch 不使用服务器架构**，它直接调用底层的模型执行引擎。

### 数据流程代码证据

#### 4.1 数据准备（第 451-453 行）
```python
# Prepare inputs for warm up
reqs = prepare_synthetic_inputs_for_latency_test(
    bench_args.batch_size[0], bench_args.input_len[0]
)
```

#### 4.2 批次创建和处理（第 230-243 行）
```python
@torch.no_grad
def extend(reqs, model_runner):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=None,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        enable_custom_logit_processor=False,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch
```

#### 4.3 解码处理（第 246-253 行）
```python
@torch.no_grad
def decode(input_token_ids, batch, model_runner):
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits
```

### 与服务器模式的区别

| 特性 | bench_one_batch | 正常服务器模式 |
|------|----------------|---------------|
| 网络通信 | ❌ 无 | ✅ HTTP/WebSocket |
| 数据输入 | 合成随机数据 | 真实用户请求 |
| 批处理 | 静态批次 | 动态批次调度 |
| 并发处理 | 同步执行 | 异步队列 |
| 内存管理 | 简化的池管理 | 完整的动态分配 |

### 真实的 Batch Size
**是的，真实的 batch size 就是你指定的 2**。代码证据：

1. **数据生成时**（第 212 行）：
   ```python
   input_ids = np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
   ```

2. **批次创建时**（第 218-225 行）：
   ```python
   for i in range(len(input_ids)):  # len(input_ids) = batch_size = 2
       req = Req(...)
       reqs.append(req)
   ```

3. **吞吐量计算时**（第 396 行）：
   ```python
   throughput = batch_size / latency  # 直接使用指定的 batch_size
   ```

## 总结

`bench_one_batch` 是一个**单机性能测试工具**，它：
1. 生成合成的随机输入数据
2. 直接调用 SGLang 的底层模型执行引擎
3. 测量 Prefill（并行处理输入）和 Decode（逐步生成输出）的延迟
4. 第一次 Decode 会因为 CUDA Graph 编译而很慢，后续会很快
5. 通过 Warmup 预先编译和优化各种 CUDA 内核和计算图

这个工具主要用于测试模型推理的原始性能，不涉及网络 I/O、请求调度等服务器层面的开销。
