# 环境配置

```
pip install -e .

# [Optinal] 如果愿意用SWANLAB替代WANDB进行可视化
pip install swanlab
```

# Entropy 计算
```
def entropy_from_logits(logits: torch.Tensor):
    """
    Calculate entropy from logits.
    Entropy: H(X) = -∑p(x)log(p(x))
    For logits, we have：
        p(x) = softmax(logits) = exp(logits) / sum(exp(logits))
        log(p(x)) = logits - logsumexp(logits)
    Involve them into Entropy Compute:
        H(X) = -∑p(x)(logits - logsumexp(logits))
        = -∑p(x)logits + logsumexp(logits)∑p(x)
        = -∑p(x)logits + logsumexp(logits)
        = logsumexp(logits) - ∑p(x)logits
    This is the formula implemented in the code.
    This implementation is more numerically stable than directly calculating -∑p(x)log(p(x)), 
    because it avoids the problem of numerical overflow or underflow during the calculation process.
    """
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy
```

# 训练

## Prompt 修改
请修改 [examples/format_prompt/math_format.jinja](examples/format_prompt/math_format.jinja) 文件

### 解释
解释下该文件内容：

```
{{ content | trim }} A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>
```

python代码是这样处理该jinja的：

```python
from jinja2 import Template

with open(format_prompt, encoding="utf-8") as f:
    self.format_prompt = f.read()

prompt_str: str = example[self.prompt_key]
format_prompt = Template(self.format_prompt.strip())
prompt_str = format_prompt.render(content=prompt_str)
```

其中`prompt_str`是用户输入的prompt，`format_prompt`是`math_format.jinja`文件的内容。
prompt_str = format_prompt.render(content=prompt_str) 这行代码将`jinja`中的`{{ content | trim }}`替换为`prompt_str`。所以`format_prompt.render`函数中会有`content=prompt_str`这个变量赋值，`trim`函数是去除`prompt_str`两端的空白字符(类似于strip())。

**Note:** 这里是默认system prompt（You are a helpful assistant!），jinja里的内容在render之后 = user query + jinja format-prompt，都是user message！

## score function修改
主要是改config中的以下内容：
```yaml
worker:
  reward:
    reward_type: function
    score_function: ./examples/score_function/agiqa3k.py:compute_score
    score_function_kwargs:
      format_weight: 0.5
      threshold: 0.35
```
- `score_function`写py名 + 冒号 + 里面的函数名
    - 在代码里会自动用冒号进行split，之后`config.score_function`是py文件名（会被import为module），`config.score_function_name`是冒号后的函数名
- `score_function_kwargs`来输入函数的一些参数

# vllm部署 + 测试

## 部署
命令:

```
vllm serve $CKPT_PATH \
  --served-model-name $MODEL_NAME \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192 \
  --tensor-parallel-size 4 \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  --limit-mm-per-prompt image=5,video=5 \
  --mm-processor-kwargs '{"max_pixels": 1048576, "min_pixels": 262144}'
```

> --tensor-parallel-size 2
> --`tensor-parallel-size 2`表示使用Tensor Parallelism技术来分配模型跨两个GPU
> Tensor Parallelism是一种分布式深度学习技术，用于处理大型模型。
> 当--tensor-parallel-size 设置为 2 时，模型的参数和计算会被分割成两部分，分别在两个GPU上进行处理。
> 这种方法可以有效地减少每个GPU上的内存使用，使得能够加载和运行更大的模型。
> 同时，它还可以在一定程度上提高计算速度，因为多个GPU可以并行处理模型的不同部分。
> Tensor Parallelism对于大型语言模型（如 Qwen2.5-14B-Instruct）特别有用，因为这些模型通常太大，无法完全加载到单个GPU的内存中。


运行：

```
bash scripts/vllm_serve.sh /code/All-In-One/qbw/EasyR1-20250410/cache/output/agiqa3k_qual_n16_continuous-thres0p75_format0p1_on-policy_newcode_20250616/global_step_144/actor/huggingface agiqa3k_qual_n16_continuous-thres0p75_format0p1_on-policy_newcode_20250616_step144
```
# 踩坑记录

## Ray CPU OOM
详细报错：
> ray.exceptions.outofmemoryerror: task was killed due to the node running low on memory.

原因：
- CPU内存不足

解决方式：
1. 关掉actor的一些offload，减少CPU Memory占用
2. 减小val_batch_size（最开始是-1，感觉是一些全load去做eval），改成了512

## (NCCL问题) RuntimeError: NCCL error: invalid usage
查看log发现有一条`NCCL version 2.21.5+cuda11.0`，而我的cuda版本是12.4，所以需要提升NCCL版本。

**完整操作可参考：** [https://blog.csdn.net/kabuto_hui/article/details/145949489](https://blog.csdn.net/kabuto_hui/article/details/145949489)

解决方案：重新编译cuda12.4下的NCCL
```bash
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j12 src.build BUILDDIR=path-to-nccl CUDA_HOME=/usr/local/cuda NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86"
```

参数解释：

- j12: 表示使用12个核心，使用nproc查看总核心数，根据具体情况进行调整；
- BUILDDIR: 表示编译后，一些文件的存储路径；默认是nccl/build；当然如果是root用户可以指定到/usr/local/ncc/；
- CUDA_HOME: 表示CUDA的目录，默认就是/usr/local/cuda；
- NVCC_GENCODE：如果不添加该字段，默认会编译支持所有架构；为了加速编译以及降低二进制文件大小，添加该字段，具体comute_35,sm_35是应该是和显卡算力相匹配，如A100是compute_80,sm_80。RTX3090是compute_86,sm_86。


完成编译后，编译产物出现在nccl/path-to-nccl/lib中：
![](./readme_qbw_assets/pic1.png)

将编译产物拷贝到当前的虚拟环境中：虚拟环境地址`/lib/python3.11/site-packages/nvidia/nccl`替换原来的头文件（上图绿色框内的头文件）和动态库（上图红色框里的.so和.a文件）：
![](./readme_qbw_assets/pic2.png)

**成功后的运行信息**：`NCCL version 2.26.2+cuda12.4`