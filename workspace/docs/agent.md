这是一个公共的为 agent 在此 repo 上工作准备的 prompt，可能比较笼统，如有冲突请以后续 prompt 的具体要求。

## 项目背景
这个 repo 是基于 sglang 官方的一个 fork，我在这个 repo 里主要进行 sparse attention 相关的开发。我要给 sglang 的 triton backend 实现一些具体的稀疏 attention 算法比如 streamingllm（基本完成）、quest（基本完成）、snapkv（正在开发）。

## 语言要求
我先给你一些约定： 首先，你要注意输出格式。仅在必要时使用子弹头（比如确实好几点并列），另外注意缩进符合实际逻辑，请尽量在高层级使用标题而不是子弹头，比如 markdown 的 ### 而不是所有第一层级都是直接子弹头开始。
请你写代码尽量多写注释（用英文），我不怕啰嗦，特别是 tensor shape 能标的都标（比如函数的参数，返回值，比如有 shape 变换操作的时候）
另外注意给我传达信息的浓缩程度，用易懂的语言但是不要有废话的客套内容。

## 项目技术要求
这是一项性能 sensitive 的 hpc 优化，很多事情不能想当然，你作为 ai 助手由于主要看代码而缺乏跟硬件的沟通，因此可能对真实的代码性能没有直观的把握，这方面请你一定要慎重并且发挥你最极致的聪明才智或者和我一起讨论确认。

## 代码结构
我主要改的代码都位于 python/sglang/srt/layers/attention/triton_ops 比如 quest_attention.py 当然还有配套的 python/sglang/srt/layers/attention/quest_backend.py，其他代码暂时请你自己探索，注意由于这是一个大项目，你一定要有全局的视角。

## 关于环境
当前的机器是在一个 hpc 集群上，节点是 GH200 也就是 ARM 架构，所有的环境都是我手动通过 conda 安装的，python 的路径为 /iopsstor/scratch/cscs/xjin/miniconda3/envs/sgl/bin/python，miniconda3 的路径为 /iopsstor/scratch/cscs/xjin/miniconda3，如果你需要用一些常见的包比如 gcc 的话都需要先用 conda 激活 sgl 这个环境，如果直接 conda activate 失败的话，你需要 source /iopsstor/scratch/cscs/xjin/miniconda3/etc/profile.d/conda.sh 进行激活。
