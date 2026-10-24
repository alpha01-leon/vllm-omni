# MiniMax-H3 三个待决项 — 事实核实与推荐方案

> **2026-10-24 更新：三个待决项已全部处置完毕**，落地结果见每节的「实施结果」小节。
> 本文档的事实核实部分保持原样（它是实施依据），结论未变。

**日期**: 2026-10-21（实施于 2026-10-24）
**核实方式**: 全部对着代码与本机 checkpoint 实测，不依赖历史文档的转述
**相关文档**（仓库外的项目工作区 `/data/projects/vllm/`）：
`minimax-h3-ref2va-int8-status-and-plan-2026-10-18.md` §七 未决问题清单、
`minimax-h3-int8-usage-guide.md`、
`vllm-omni-minimax-h3-comfyui-compat-design.md` §10。
仓库内的落地位置：`tools/minimax_h3/convert_comfy_lora.py`、
`tests/tools/test_convert_comfy_lora.py`、
`tests/diffusion/models/minimax_h3/test_minimax_h3_sampler.py`、
`recipes/MiniMaxAI/MiniMax-H3.md`、

> 这三个项此前一直以「📋 待决」的形式挂着。本文给出核实到的事实和推荐处置。
> **其中两个可以直接关掉**，第三个的结论是：这是**文件转换任务**，不是运行时改动。

---

## 决策点 1：NVFP4 文本编码器 rel_err 0.17–0.29

### 核实到的事实

**① 当前部署根本走不到这条路径。**
服务加载的是 bf16 分区的 `text_encoder/`，NVFP4 checkpoint 从未被加载。
且 `resolve_encoder_quant_config`（`vllm_omni/quantization/component_config.py:51-74`）
对预量化方法返回 `None`，保证 FP8 kernel 不会被误套到 BF16 权重上：

```python
if getattr(quant_config, "is_checkpoint_int8_serialized", False):
    return None
if quant_config.get_name() in PRE_QUANTIZED_METHODS:
    return None
```

**② NVFP4 对本任务的价值只有磁盘体积。**
设计文档 §3.7 记录：checkpoint 里 350 个 nvfp4 层**全部**带
`full_precision_matrix_mult: true`，即 ComfyUI 运行时本就反量化回 BF16
再做普通 GEMM，**没有 NVFP4 tensor-core 路径**。
所以 NVFP4 权重的价值仅在于 63 GiB → 15 GiB。

**③ 真正会让误差变得致命的两个因素恰好都不在：**
- 100/350 个层带 `pre_quant_scale`（ModelOpt AWQ 激活平滑），
  作用于**激活侧**：`input = input * pre_quant_scale`
  （`comfy/ops.py:1376-1378`）。checkpoint 头部 `__metadata__` **为空**，
  没有任何文字标记提示它们的存在 —— 忽略它们会**静默**产生错误输出。
- 只有 LM 塔被量化；`visual.*` 整个 ViT 塔是 BF16。

**④ 0.17–0.29 这个数字的口径本身可能不对。**
它测的是「忠实复现 ComfyUI 运行时语义」的误差 —— 而 ComfyUI 运行时压根不跑
NVFP4 GEMM。真正该测的指标是「转换后的权重在 **BF16 GEMM** 下与源权重的差异」，
这个**从未测过**，而它才是决定要不要转换的判据。

### 推荐：**关掉，不修**

理由不是「修不动」，而是**修它没有收益，且当前无风险敞口**。

**具体动作（零代码）：**
1. 待决表里这一行从「未定位」改为「**已判定不影响本任务**」，附三条理由
   （无运行时路径 / 仅体积价值 / `pre_quant_scale` 是转换时的硬要求）。
2. 不投入排查。

**如果将来真要转换**（例如磁盘紧张、或要支持 ComfyUI 侧的 NVFP4 权重对齐）：
先测 BF16 GEMM 口径的 rel_err（这才是判据），再写转换器，且必须处理
`pre_quant_scale` 与 `lm_head` 映射。这与 `convert_comfy_checkpoint.py`
是同一类任务，可复用其结构与测试方式。

**触发重新评估的条件**：
- 需要把 ComfyUI 的 NVFP4 权重转成本仓库格式以做逐层对拍；
- 磁盘压力变成实际约束。


### 实施结果（2026-10-24）

按推荐「关掉，不修」执行，**未做任何代码改动**：

- 设计文档 §10 的待决项 4 改写为「已判定不影响本任务」，并附三条判据。
- 状态文档 §七 的第 3 行改为「✅ 已判定不影响本任务」。
- **没有**写 NVFP4 转换器，也没有去追那个 rel_err 的成因 —— 按 §1.4 的理由，
  当前没有使用场景，转换器本身才是风险（`pre_quant_scale` 静默出错）。

---

## 决策点 2：`res_multistep` 守卫是否放宽

### 核实到的事实

**① 守卫只在 `base_schedule is not None` 时触发**（两处：
`pipeline_minimax_h3.py:1814` 与 `:2898`），而这只可能出现在**蒸馏版** release
—— 自带固定的 sigma 位置（`DMD2SigmaSchedule.from_metadata`，
`vllm_omni/diffusion/sched/sigma_schedule.py:53-68`；键缺失 = 非蒸馏，
走遗留均匀调度）。

**② 我们本机四个 checkpoint 全都不触发。** 实测 `_minimax_h3` 元数据：

| checkpoint | 蒸馏调度？|
|---|---|
| `/data/models/MiniMax-H3/FL2VA` | ❌ 无 |
| `/data/models/MiniMax-H3/Ref2VA` | ❌ 无 |
| `FL2VA_comfyint8/FL2VA` | ❌ 无 |
| `Ref2VA_comfyint8/Ref2VA` | ❌ 无 |

四者的 `_minimax_h3` 块只有 `schema_version` / `partition` / `tasks` /
`task_aliases` / `sigma_shift_scales`，**没有 `base_schedule` 键**。

**③ 数值上两种调度都稳定，守卫不是防崩溃的。** 实测 `res_multistep`
在均匀调度与蒸馏调度下每步输出都有限、系数行为良好
（最坏情况是病态 4 点梯度的 `(1.805, -1.36)`）：

```
uniform  (9 点): coeffs = [(0.227,-0.085) ... (0.976,-0.476)]
distilled(5 点): coeffs = [(0.584,-0.251), (0.976,-0.476)]
```

所以守卫的含义是「**这个组合没被验证过**」，不是「会算出 NaN」。

**④ 蒸馏路径为什么值得保守。** 蒸馏 student 只在训练时见过的那些噪声级别上
被优化过。Euler 一步到位的轨迹与二阶 solver 走过的中间状态不同，
而蒸馏调度通常只有 4–8 个点，**二阶 solver 的历史几乎没有施展空间**
（`c2 = log(curr/prev)/h`，相邻点间隔不均时该系数会显著变大）。
在「没验证过」和「有空间但未验证」之间保守是对的。

**⑤ 均匀调度下的 `res_multistep` 目前没有测试覆盖。**
`tests/` 里唯一一处 `res_multistep` 出现在
`tests/diffusion/models/minimax_h3/test_vdnh3.py:181`，
而它断言的正是**蒸馏路径会抛错**：

```python
with pytest.raises(OmniClientError, match="Euler"):
    ckpt.check_request(
        OmniDiffusionSamplingParams(num_inference_steps=8,
                                    extra_args={"sampler": "res_multistep"}), "t2va")
```

### 推荐：**保留守卫，把范围收窄到「非蒸馏 + 补测试」**

不推荐的方案是「全面放宽」—— 那会让未验证的组合进入生产路径，
而蒸馏路径的守卫恰恰是有道理的。

**推荐动作（三件事，都很小）：**

1. **保留现有守卫**（`base_schedule is not None` 的两处）不动。
2. **补均匀调度下的 `res_multistep` 测试**。这是当前唯一的真实缺口：
   功能已实现、数值已实测正常、但没有回归保护。补一个单测即可
   （均匀 sigma 序列上跑一遍 solver，断言每步有限、且与 Euler 在
   收敛行为上可比），成本很低。
3. **把守卫文案改得更准**。现在写的是
   "has not been validated; use sampler='euler'"，但读者容易以为
   `res_multistep` 整体不可用。实际上它在均匀调度下可用。
   建议改为明确区分两种情况：
   - 蒸馏调度 + `res_multistep` → 拒绝（现状）
   - 均匀调度 + `res_multistep` → 允许，但文档标注「未与参考实现对拍」

**触发重新评估的条件**：
- 拿到一个**非蒸馏**但需要二阶 solver 的场景；
- 或有了 ComfyUI `res_multistep` 的参考实现可用于对拍验证。


### 实施结果（2026-10-24）

按推荐「保留守卫 + 收窄文案 + 补测试」执行：

- **守卫文案改写**（`pipeline_minimax_h3.py:1813-1818` 与 `:2896-2902`，两处同步）：
  旧文案「...has not been validated」只说「没验证过」，容易被读成「所有
  res_multistep 都没验证过」；新文案明确区分两件事 —— 蒸馏调度下**不���许**用，
  均匀调度下**允许**但仍未与参考实现比对过。守卫的**范围没有放宽**。
- **补了均匀调度下的 `res_multistep` 单元测试**（新增
  `tests/diffusion/models/minimax_h3/test_minimax_h3_sampler.py`，8 个用例）：
  此前全仓库只有一处 `res_multistep` 测试，且断言的是**蒸馏路径会抛错**，
  均匀调度这条真正在用的路径**完全没有覆盖**。新测试覆盖系数结构
  （首末为 `None` → 退化为 Euler）、中间步为真二阶项、状态机契约
  （必须按序消费）、以及数值输出有限。
- **recipe 同步**（`recipes/MiniMaxAI/MiniMax-H3.md`）：§请求参数与 §采样器表
  两处描述都改成与新文案一致。

实测：`res_multistep` 在均匀调度下**本来就是通的**（8 步全部有限，rms 单调递减），
所以这条是「补证据」而不是「修 bug」。

---

## 决策点 3：int8 + 动态 LoRA 兼容性

### 核实到的事实

**① 运行时机制与量化无关 —— int8 路径本身已经兼容。**
这是最反直觉也最关键的一点。`DiffusionBaseLinearLayerWithLoRA.apply`
（`vllm_omni/diffusion/lora/layers/base_linear.py`）这样写：

```python
output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
...
buffer = x_flat @ A.t()
torch.addmm(y_slice, buffer, B.t(), out=y_slice)   # y += (x @ A.T) @ B.T
```

LoRA 是在**输出空间**叠加的，**从不触碰被量化的权重**。
int8 kernel 只负责 `quant_method.apply` 那一段，LoRA 是它之后的独立加法。
所以「int8 权重 + LoRA」在数学上没有冲突。

**② 层的包装按类型进行，量化不可见。**
`from_layer_diffusion`（`vllm_omni/diffusion/lora/utils.py:58-90`）按
层类型分派，替换表是：

```
DiffusionMergedQKVParallelLinearWithLoRA
DiffusionQKVParallelLinearWithLoRA
DiffusionMergedColumnParallelLinearWithLoRA
DiffusionColumnParallelLinearWithLoRA
DiffusionRowParallelLinearWithLoRA
DiffusionReplicatedLinearWithLoRA
```

H3 的 DiT 用的正是标准类型（`minimax_h3_transformer.py`）：
`ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` /
`MergedColumnParallelLinear` —— **全部在替换表里**。

**③ 真正的阻塞是「产物」，不是「运行时」。** 实测本机两个 Turbo LoRA
**都解析失败**：

```
minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors -> None
minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors  -> None
```

两个**互相独立**的原因：

- **文件名**：`_TURBO_NAME_RE` 不接受 `_comfyui_` 中缀。
- **布局**：ComfyUI 通用拼写（融合的 `qkv_proj` / `fc1` / `fc2`，实测 624 keys）
  vs 加载器要求的 Diffusers PEFT 拼写
  （`attn.to_q` / `to_k` / `to_v` / `to_out.0`、`ff.net.0.proj` / `net.2`）。

加载器期望的目标集合（`lora.py:123-130`）：

```
{transformer_blocks × 0..49, token_refiner.refiner_blocks × 0..1}
  × {attn.to_q, attn.to_k, attn.to_v, attn.to_out.0, ff.net.0.proj, ff.net.2}
  = 52 × 6 = 312 对
```

文件 metadata 明确记录了转换规则：

```json
{
  "source_format": "Diffusers PEFT LoRA",
  "target_format": "ComfyUI generic LoRA",
  "qkv_fusion": "block diagonal B; concat A; alpha multiplied by 3",
  "swi_glu_mapping": "Diffusers [value;gate] -> ComfyUI [gate;value]",
  "training_rank": "128", "training_alpha": "8.0", "training_scale": "0.0625"
}
```

**④ 没有任何 int8 + LoRA 的测试覆盖。**
`tests/diffusion/lora/test_images_generations_lora.py` 里的 "int8" 是
`uint8` 的误匹配，与量化无关。

### 推荐：**写 LoRA 文件转换器，运行时不动**

**为什么不是改运行时**：①② 已经证明运行时是通的。改运行时是在解决
一个不存在的问题。

**为什么不是「当前无场景，先放着」**：产物侧是**确凿的失败**
（两个文件都解析不了），不是假想。而且转换器是
`convert_comfy_checkpoint.py` 的同构任务，复用度高。

**具体动作：**

1. **写 `tools/minimax_h3/convert_comfy_lora.py`**，把 ComfyUI 通用 LoRA
   转成加载器要的 Diffusers PEFT 布局：
   - `qkv_proj` 拆回 `to_q` / `to_k` / `to_v`（B 的 block diagonal 要拆，
     A 的 concat 要按 metadata 记录的 `alpha multiplied by 3` 反向还原）
   - `fc1` 按 `swi_glu_mapping` 拆回 `ff.net.0.proj`（注意 `[gate;value]`
     与 Diffusers `[value;gate]` 的顺序差）
   - `fc2` → `ff.net.2`；`out_proj` → `attn.to_out.0`
   - 输出文件名去掉 `_comfyui_` 中缀
   - 补 `lora_A.default.weight` / `lora_B.default.weight` 后缀
2. **加单测**：转换后与源 LoRA 的**等效性**要能证明 —— 最直接的判据是
   「同一 base 模型上，转换前后前向输出一致」。
3. **先在 BF16 base 上验证**（Turbo LoRA 本来就是给 BF16 base 蒸馏的）。

**⚠️ 重要的额外前提**：Turbo LoRA 是为**特定步数与分辨率**蒸馏的
（`minimax_h3_fl2v_turbo_8step`、`minimax_h3_ref2v_turbo_4step`，
以及 544p/768p 两档）。把它用在 **int8 分区**上会改变数值，
**需要它自己的一套精度验证** —— 不能拿 FL2VA/Ref2VA 的对拍结论来背书。
这一条不做，转换器做完也只是「能加载」，不是「能信」。

**触发重新评估的条件**：
- 有明确的「要在 int8 分区上用 Turbo 少步加速」的需求；
- 或需要支持用户自带 LoRA（那时覆盖面比 Turbo 更广，转换器要更通用）。


### 实施结果（2026-10-24）

按推荐「写 LoRA 文件转换器，运行时不动」执行。

**新增 `tools/minimax_h3/convert_comfy_lora.py`**（自包含：stdlib + torch + safetensors，
不 import ComfyUI、不 import vllm_omni —— 与同目录的 `convert_comfy_checkpoint.py` 同风格）。

转换规则（与源文件 `__metadata__` 记录的融合规则一一对应，并且**校验**这些 metadata，
缺失即报错，不静默假设）：

| # | 变换 | 说明 |
|---|---|---|
| 1 | 文件名 | 去掉 `_comfyui_` 中缀，保留 `_bf16` 后缀 → `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors`，命中 loader 的 `_TURBO_NAME_RE` |
| 2 | QKV 拆分 | fused `qkv_proj` → `to_q`/`to_k`/`to_v`：**B 按输出块切**（块对角，实测块外非零元素 **0**）、**A 按行切**（3×128 行） |
| 3 | alpha | 全局 alpha 取**未融合投影**的值 **8.0**（只有 QKV 的 24.0 是融合后的 3 倍）；产出 metadata 写 `alpha: 8.0` 与 `key_format: minimax-h3-diffusers` |
| 4 | FFN | `mlp.fc1` → `ff.net.0.proj`、`mlp.fc2` → `ff.net.2`；**B 的两半对调**成 `[value;gate]`（loader 自己会再换回 `[gate;value]`） |
| 5 | 命名空间 | `diffusion_model.blocks.N` → `transformer_blocks.N`；`token_refiner.blocks.N` → `token_refiner.refiner_blocks.N`；`.lora_A.weight` → `.lora_A.default.weight` |
| 6 | alpha 张量 | **丢弃**每个 tensor 的 `.alpha`（loader 只接受一个全局 alpha，带 per-tensor alpha 会触发 "Unconsumed"）|

**实测结果**（本机两个真实 Turbo LoRA）：

- 转换统计：`{'blocks': 52, 'pairs': 312, 'split_qkv': 156, 'swapped_ffn': 52, 'passthrough': 208}`
- loader 完整接受：`parse_turbo_filename` 解析出正确的 `TurboSpec`
  （family / step 数 / 两个 shift / rank / alpha），`_validate_and_convert_tensors` 通过全部 624 个张量。
- **前向等价性**：同一随机输入下，比较 ComfyUI 源文件蕴含的 LoRA 增量与转换后文件蕴含的增量，
  最差相对误差 **QKV 3.5e-07 / FFN 0.0**（fp32 累加噪声量级）。这是「转换前后算同一个函数」的证据。

**新增 `tests/tools/test_convert_comfy_lora.py`**（20 个用例，`core_model and cpu`）：
逐层数值等价（QKV 拆分后与 fused 前向等价、FFN 对调后与源等价）、alpha 处理、
命名空间映射、metadata 校验与拒绝路径、loader 接受性、文件名规则。
其中「loader 接受性」用例用**完整 52 块** fixture，因为 loader 的目标集合校验
（`lora.py:218`）要求 50 个 transformer block + 2 个 token refiner block 全部到位。

**运行时未改动**：int8 kernel、LoRA 运行时、`_TURBO_NAME_RE` 都没动。
int8 + LoRA 的兼容性是**架构上成立**的 —— LoRA 增量在**输出空间**叠加
（`diffusion/lora/layers/base_linear.py:123` 先调 `quant_method.apply`，
再叠加 `x @ lora_a.T @ lora_b.T`），从不触碰被量化的权重。
这一点写进了 recipe 的「Combining an adapter with the int8 partition」小节。

**未做**：int8 分区 + Turbo LoRA 的**精度**验证。转换器做完只是「能加载」，
不是「能信」；需要「用 Turbo 少步加速」这个需求真的存在时再单独验。

---

## 汇总

| # | 事项 | 推荐 | 状态 | 成本 | 风险 |
|---|---|---|---|---|---|
| 1 | NVFP4 文本编码器 | **关掉**，不修；仅改文档口径 | ✅ 已处置 | 零 | 无（无运行时路径）|
| 2 | `res_multistep` 守卫 | **保留**，补均匀调度测试 + 改文案 | ✅ 已处置 | 一个小单测 | 无 |
| 3 | int8 + 动态 LoRA | **写文件转换器**，运行时不动 | ✅ 已处置 | 一个转换器 + 单测 | 中（int8 上的数值需另验）|

**三者在 2026-10-24 全部落地**，各自的「实施结果」小节记录了具体改了什么、实测到什么。

**三者的共同点**：都不是「跑不起来」的问题。
#1 是「没有使用场景」，#2 是「已经够用但缺测试」，#3 是「文件不对而非代码不对」。

---

## 附：本文引用的代码位置

| 事实 | 位置 |
|---|---|
| 预量化方法不套 FP8 | `vllm_omni/quantization/component_config.py:51-74` |
| `pre_quant_scale` 作用于激活侧 | ComfyUI `comfy/ops.py:1376-1378` |
| 蒸馏调度读取 | `vllm_omni/diffusion/sched/sigma_schedule.py:53-68` |
| `res_multistep` 守卫（两处）| `vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py:1814`、`:2898` |
| `res_multistep` 求解器 | `vllm_omni/diffusion/models/minimax_h3/sampling.py:28-80` |
| LoRA 在输出空间叠加 | `vllm_omni/diffusion/lora/layers/base_linear.py`（`apply`）|
| LoRA 层替换分派表 | `vllm_omni/diffusion/lora/utils.py:58-90` |
| Turbo 文件名正则与目标集合 | `vllm_omni/diffusion/models/minimax_h3/lora.py:38-41`、`:123-130` |
| 唯一的 `res_multistep` 测试 | `tests/diffusion/models/minimax_h3/test_vdnh3.py:181` |
