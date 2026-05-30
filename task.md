# NVIDIA-Nemotron-3 系列混合模型适配

适配NVIDIA-Nemotron-3 系列nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/NVIDIA-Nemotron-3-Nano-4B-BF16模型，采用异构 Mamba-Transformer 混合专家 (mixture-of-experts, MoE) 架构、交互式环境强化学习 (reinforcement learning, RL)，以及原生 100 万 token 上下文窗口，可为多智能体应用提供高吞吐量、长时推理能力

1. 在 test/registered/ascend 添加完整测试用例
2. 在 docs/platforms/ascend/ascend_npu_support_models.md 更新模型支持文档，在docs/platforms/ascend/下创建模型适配教程文档
3. 在对应issue下提交SKILL.md（记录AI辅助开发经验、有效提示词及最佳实践）

# Jet-Nemotron-2B 新型混合架构语言模型模型适配

适配Jet-Nemotron-2B模型，采用Post Neural Architecture Search（后神经架构搜索），是一种高效的后训练架构探索和适应流程，适用于任意预训练的Transformer模型；
其线性模块JetBlock 是一种新型的线性注意力模块，其性能显著优于之前的 Mamba2 等设计。

1. 在 test/registered/ascend 添加完整测试用例
2. 在 docs/platforms/ascend/ascend_npu_support_models.md 更新模型支持文档，在docs/platforms/ascend/下创建模型适配教程文档
3. 在对应issue下提交SKILL.md（记录AI辅助开发经验、有效提示词及最佳实践）"

# Tencent-Hunyuan-Large综合能力模型适配

适配Tencent-Hunyuan-Large 389B Moe模型，混元大模型在常识理解、问答、数学推理、编程及综合任务上均表现出色，总参数量为 389B，激活参数量为 52B，实现了超大规模模型性能与高推理效率的平衡，整体适配难度适中。

1. 在 test/registered/ascend 添加完整测试用例
2. 在 docs/platforms/ascend/ascend_npu_support_models.md 更新模型支持文档，在docs/platforms/ascend/下创建模型适配教程文档
3. 在对应issue下提交SKILL.md（记录AI辅助开发经验、有效提示词及最佳实践）

# Qwen3-ASR 多语言自动语音识别模型适配

适配 Qwen/Qwen3-ASR-1.7B（2026年1月），1.7B参数高效ASR模型，多语言转录精度高，架构轻量，适合实时语音处理部署。需适配语音输入预处理和音频特征提取算子。

1. 在 test/registered/ascend 添加完整测试用例
2. 在 docs/platforms/ascend/ascend_npu_support_models.md 更新模型支持文档，在docs/platforms/ascend/下创建模型适配教程文档
3. 在对应issue下提交SKILL.md（记录AI辅助开发经验、有效提示词及最佳实践）"

# NVILA-8B 原生视觉语言大模型适配

适配 NVILA-8B模型，NVILA是同时优化效率与准确性的开放式视觉语言模型家族。基于 VILA 模型，通过“扩展-压缩”（scale-then-compress）方法高效处理高分辨率图像和长时间视频。 需要同时验证精度和推理性能。

1. 在 test/registered/ascend 添加完整测试用例
2. 在 docs/platforms/ascend/ascend_npu_support_models.md 更新模型支持文档，在docs/platforms/ascend/下创建模型适配教程文档
3. 在对应issue下提交SKILL.md（记录AI辅助开发经验、有效提示词及最佳实践）

# step-3.5-flash 大语言模型适配优化

适配step-3.5-flash轻量化高速大语言模型，该模型具备低延迟、高吞吐、通用对话能力强的特性，适配Ascend NPU硬件架构；完成模型权重迁移、推理逻辑适配、算子兼容改造，优化模型推理调度逻辑，保障模型在SGLang框架下稳定、高效运行，适配通用对话、文案生成、智能问答等业务场景。

1. 在 test/registered/ascend 添加完整功能性、性能测试用例；
2. 在 docs/platforms/ascend/ascend_npu_support_models.md 补充该模型适配信息，新增模型部署适配教程文档；
3. 在对应issue下提交SKILL.md，记录模型适配踩坑点、AI辅助开发经验及最优部署方案

# DeepSeek-OCR2 多模态OCR模型适配优化

适配DeepSeek-OCR2高精度图文识别模型，该模型支持多语言文字识别、手写体识别、复杂版式文档解析；基于SGLang框架完成图像预处理、视觉编码、文字解码链路适配，兼容Ascend NPU算力特性，优化图像推理算力占用，保障OCR识别精度与推理速度达标。

1. 在 test/registered/ascend 添加图片识别、多场景版式识别测试用例；
2. 更新模型支持清单文档，编写OCR模型专属适配部署教程；
3. 提交SKILL.md，记录多模态模型适配技巧、算子优化方案及开发最佳实践

# gemma4 通用开源大语言模型适配优化

适配Google gemma4新一代开源通用大语言模型，模型包括Gemma 4 31B/E4B/E2B, 该模型兼容模型原生Transformer架构，适配Ascend NPU算子生态；完成模型权重转换、推理链路改造，优化上下文窗口、文本生成速度，适配开源社区通用推理、微调使用场景，对齐原版模型推理效果。

1. 在 test/registered/ascend 添加文本生成、上下文对话、精度校验测试用例；
2. 完善模型支持文档，制作开源简易适配部署教程；
3. 提交SKILL.md，记录开源大模型适配通用流程、兼容问题解决方案


------------

**上边是任务，下边是模板**

------------

## 任务名称

NVIDIA-Nemotron-3 系列混合模型适配

### 任务描述

适配 NVIDIA-Nemotron-3 系列中的 **NVIDIA-Nemotron-3-Super-120B-A12B-BF16** 与 **NVIDIA-Nemotron-3-Nano-4B-BF16** 模型。
该系列采用异构 Mamba-Transformer 混合专家（mixture-of-experts, MoE）架构，并结合交互式环境强化学习（reinforcement learning, RL）
训练，支持原生 100 万 token 上下文窗口，专为多智能体应用场景提供高吞吐量、长时推理能力。

本次任务目标是在 Ascend NPU 平台上完整实现模型的训练/推理适配，充分发挥硬件算力，并保证精度与性能指标。

本期任务的具体信息如下：
1. 模型名称：[NVIDIA-Nemotron-3-Super-120B-A12B-BF16](https://www.modelscope.cn/models/nv-community/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) / [NVIDIA-Nemotron-3-Nano-4B-BF16](https://www.modelscope.cn/models/nv-community/NVIDIA-Nemotron-3-Nano-4B-BF16)
2. 开发平台：Atlas 800T A2 或 A3

### 验收标准

**一、任务交付件**

本期任务基于现有框架/仓库进行功能扩展，需提交开发代码与文档，主要开发点如下：

1. 提交PR
  - 完整的推理适配代码
  - test/registered/ascend添加测试用例，看护精度
  - 更新模型支持文档docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_models.mdx
  - 填写完整PR描述，包含精度，性能结果，以社区规范为准
  - （可选）适配中遇到并解决的问题欢迎补充docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx
2. 补充Issue
  - Issue中可以补充开发经验，以及遇到的问题和解决方式

**二、验收标准**

1）精度/性能要求
- 精度：适配后模型在代表性评测任务上的精度与官方基准误差不超过 1%，官方基准可以从下载页的官方卡片获取。
- 性能：N/A

2）PR中包含完整的推理代码，测试代码和PR描述

### PR 合入

本地完成测试验证后，向 https://github.com/sgl-project/sglang 的 main 分支发起 PR。

### 对接人

@amote-i

欢迎加入社区，感谢您对社区的贡献 🎉!
