> Open-Pi-Zero (Pi0) 的核心妙计不是“欺骗”LLM，而是给LLM配了一个**“动作专家”保镖**。
> 
> 它不再强迫LLM去“说”出动作 Token，而是采用 混合专家 (MoT/MoE) 架构：
> 
> LLM (PaliGemma) 负责“看和理解”，生成的特征被缓存；
> 
> 一个专门的 Action Expert 负责“动”，它通过 Flow Matching（流匹配） 算法，
> 
> 在 LLM 的指导下，将随机噪声一步步“流”向正确的连续物理动作。
> 
> 这样既保留了 VLM 的语义理解，又利用了扩散类模型生成连续动作的高精度优势。

# 架构图

- **输入层 (Inputs)**
    
    - **图像**：相机实时画面 (经过 Resize 到 224x224)
        
    - **文本**：指令（如 "put the carrot on the plate"）
        
    - **本体状态 (Proprio)**：机器人的关节角度、夹爪状态等
        
    - **噪声动作 (Noisy Action)**：(训练时) 混合了高斯噪声的动作；(推理时) 纯高斯噪声
        
- **Step 1: 视觉与文本编码 (VLM Expert)**
    
    - **Vision Encoder**: 使用 **SigLIP** (来自 PaliGemma) 将图像切片并编码为 Visual Tokens。
        
    - **Text Embedding**: 使用 **Gemma** 的 Embedding 层将文本转为 Text Tokens。
        
    - **VLM处理**: Visual + Text Tokens 进入 **PaliGemma (3B)** 的 Transformer 层。
        
        - _关键点_：这里的 VLM 主要是“只读”的（或者用 LoRA 微调），它只负责产出高质量的**上下文特征 (Context)**，**不直接输出动作**。
            
- **Step 2: 状态与动作编码 (Action/Proprio Expert)**
    
    - **Proprio Encoder**: 一个简单的 MLP (Linear) 将本体状态映射为 Token。
        
    - **Action Encoder**: 另一个 MLP 将当前的“噪声动作” (Noisy Action) 和 **时间步 (Time Step t)** 映射为 Token。
        
- **Step 3: 块状因果注意力 (Block-wise Causal Attention)**
    
    - 这是 Pi0 的灵魂。三个专家 (VLM, Proprio, Action) 在同一个 `JointModel` 中交互，但彼此可见性受到严格限制：
        
        1. **VLM**：只看自己 (双向/因果)。
            
        2. **Proprio**：看 VLM + 自己。
            
        3. **Action**：看 VLM + Proprio + 自己。
            
    - _效果_：Action Expert 能够“抄” VLM 和 Proprio 的作业，但 VLM 不需要关心动作是怎么生成的。
        
- **Step 4: 动作解码与流匹配 (Flow Matching)**
    
    - **Action Decoder**: 将 Action Expert 处理后的 Token 解码回物理向量。
        
    - **输出**: **速度向量 (Velocity, v)**。
        
    - _含义_：模型预测的是“当前动作应该往哪个方向变、变多快”才能接近真实动作。
        

# 如何训练

_`src/agent/train.py` 和 `src/model/vla/pizero.py` 揭示了训练的核心逻辑_

1. **Flow Matching Loss (流匹配损失)**
    
    - 不同于 OpenVLA 的“下一个词预测 (Cross Entropy)”，Pi0 使用 **MSE Loss**。
        
    - **过程**：
        
        1. 采样一个时间 $t \in [0, 1]$ (训练时倾向于采样早期的 $t$，即 Beta 分布)。
            
        2. 构造输入：$x_t = (1-t)x_{noise} + t x_{data}$ (在噪声和真实动作之间插值)。
            
        3. 模型预测速度 $v_{pred}$。
            
        4. 计算 Loss：$||v_{pred} - (x_{data} - (1-\sigma_{min})x_{noise})||^2$。
            
    - _本质_：训练模型学会把“乱码动作”推向“正确动作”的向量场。
        
2. **系统优化 (System Optimizations)**
    
    - **`torch.compile`**: 默认开启。将 PyTorch 代码编译成优化的 Kernel，极大提升训练和推理速度。
        
    - **`bfloat16`**: 全程使用 `bfloat16` 精度，显存占用减半，且配合 FlashAttention 加速。
        
    - **8-bit Optimizer**: 使用 `bitsandbytes` 的 8位 AdamW 优化器，大幅降低优化器状态的显存占用。
        
3. **冻结与微调策略**
    
    - **VLM 部分**: 加载预训练 PaliGemma 权重。可以选择完全冻结，或者只训练 **LoRA** 层 (rank=32)。
        
    - **Action Expert**: **从零训练 (Scratch)**。这部分参数量很小 (约 0.3B)，专门负责动作控制逻辑。
        

# 如何执行 (推理)

_`src/agent/eval.py` 和 `pizero.py` 中的 `infer_action` 展示了独特的推理流程_

不同于 LLM 的“一个词一个词崩出来”，Pi0 是 **“思考一次，推演十步”**。

- **Step 1: 预填充 (Prefill)与缓存**
    
    - **输入**：图像 + 文本 + Proprio。
        
    - **VLM 前向**：图像和文本通过 VLM Expert。
        
    - **KV Cache**：**关键优化！** VLM 的 Key 和 Value 被**缓存 (Cache)** 下来。
        
    - _原因_：在接下来的生成动作过程中，图像和文本是不变的，不需要重复计算 VLM，只需反复查询缓存。
        
- **Step 2: 动作生成循环 (Denoising Loop)**
    
    - **初始化**：随机生成一个纯高斯噪声动作 $x_0$。
        
    - **循环 N 次** (例如 `num_inference_steps=10`)：
        
        1. **Action Expert 前向**：输入当前的动作猜测 $x_t$ 和时间 $t$。
            
        2. **查阅缓存**：Action Expert 通过注意力机制，去“看”Step 1 中缓存的 VLM 特征。
            
        3. **预测速度**：模型输出速度 $v$。
            
        4. **欧拉积分 (Euler Step)**：更新动作 $x_{t+1} = x_t + v \times dt$。
            
    - _结果_：循环结束后，噪声变为了清晰的 7D 物理指令。
        
- **Step 3: 适配器 (Adapter)**
    
    - `src/agent/env_adapter/simpler.py` 负责脏活累活：
        
        - **归一化还原**：模型输出通常在 [-1, 1]，需要反归一化到机器人的实际物理单位。
            
        - **Gripper处理**：虽然 Flow Matching 输出是连续的，但在仿真环境中可能需要将夹爪动作二值化（例如 >0.5 则闭合）。
            

## 深入解析：Pi0 的 "降本增效" 魔法

### 1. 为什么叫 "Joint Model" (混合专家 MoT)？

OpenVLA 强行把视觉特征塞进 LLM 的 Embedding 空间，让 LLM "独自承担所有"。

Pi0 的 JointModel () 则是分工合作：

- **VLM Expert**：你是大脑，你负责懂图、懂话。你只看你自己。
    
- **Action Expert**：你是小脑/手，你负责动。你看着大脑(VLM)的信号来决定怎么动。
    
- **连接方式**：**Attention**。Action Expert 的 Query 去查询 VLM Expert 的 Key/Value。
    
- **优势**：Action Expert 可以很轻量 (0.3B)，训练极快；VLM 可以很重 (3B+)，但推理时只需要算一次 (Prefill)。
    

### 2. KV Cache 的妙用 (速度提升的核心)

在 OpenVLA 中，每生成一个动作 token，整个大模型都要跑一遍。

在 Pi0 中，虽然生成动作需要迭代 10 步 (Flow Matching)，但 3B 参数的 VLM 只需要跑第 1 步。

剩下的 9 步，只有 0.3B 参数的 Action Expert 在跑，并且它只是去查表 (Cross Attention) 之前算好的 VLM 特征。

这使得 Pi0 即使需要多步迭代，推理速度依然能达到 50-75ms (配合 torch.compile)，完全满足实时控制需求。

### 3. 连续 vs 离散 (Tokenization vs Flow Matching)

- **OpenVLA (离散)**：动作 0.123 必须变成 "Token_50"。这丢失了精度，且 token 之间没有数学上的“远近”关系（Token_50 和 Token_51 在 ID 上只差1，但在语义空间可能差很远）。
    
- **Pi0 (连续)**：直接在连续空间操作。$0.123$ 就是 $0.123$。模型预测的是**梯度/速度**。这对于这就好比：
    
    - OpenVLA 是在**做选择题**（从256个选项里选一个）。
        
    - Pi0 是在**画画**（从草稿逐步描绘出精确线条）。对于机械臂这种需要精细控制的任务，"画画" (回归/生成) 往往比 "做题" (分类) 效果上限更高。