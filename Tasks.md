## Task Name
NVIDIA-Nemotron-3 Series Hybrid Model Adaptation

### Task Description
Adapt the **NVIDIA-Nemotron-3-Super-120B-A12B-BF16** and **NVIDIA-Nemotron-3-Nano-4B-BF16** models from the NVIDIA-Nemotron-3 family.
This series employs a heterogeneous Mamba-Transformer Mixture-of-Experts (MoE) architecture, trained with interactive environment reinforcement learning (RL), and supports a native 1-million-token context window to deliver high-throughput, long-context reasoning for multi-agent applications.

The goal is to achieve full inference/training adaptation on the Ascend NPU platform, fully exploit hardware capabilities, and ensure that accuracy and performance meet the required benchmarks.

Detailed information for this phase:
1. Model name: [NVIDIA-Nemotron-3-Super-120B-A12B-BF16](https://www.modelscope.cn/models/nv-community/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) / [NVIDIA-Nemotron-3-Nano-4B-BF16](https://www.modelscope.cn/models/nv-community/NVIDIA-Nemotron-3-Nano-4B-BF16)
2. Development platform: Atlas 800T A2 or A3

### Acceptance Criteria

**I. Deliverables**

This phase extends the existing framework/repository and requires development code and documentation. The main development points are:

1. Submit a PR
  - Complete inference adaptation code.
  - Add test cases under `test/registered/ascend` to guard accuracy.
  - Update the model support document `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_models.mdx`.
  - Provide a full PR description that includes accuracy and performance results, following community conventions.
  - (Optional) Supplement `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` with problems encountered and solved during adaptation.
2. Update the Issue
  - The issue may include development experience, problems encountered, and their solutions.

**II. Acceptance Criteria**

1) Accuracy / Performance requirements
- Accuracy: The adapted model's accuracy on representative evaluation tasks must not deviate more than 1% from the official baseline. The official baseline can be obtained from the model card on the download page.
- Performance: N/A

2) The PR must contain complete inference code, test code, and a PR description.

### PR Merge
After local testing and verification, create a PR against the `main` branch of https://github.com/sgl-project/sglang.

### Point of Contact
@amote-i

Welcome to the community and thank you for your contribution 🎉!

---

## Task Name
Jet-Nemotron-2B Novel Hybrid Architecture Language Model Adaptation

### Task Description
Adapt the **Jet-Nemotron-2B** model. This model leverages Post Neural Architecture Search, an efficient post-training architecture exploration and adaptation workflow applicable to any pre-trained Transformer model. Its core linear module, JetBlock, is a novel linear attention block whose performance significantly surpasses previous designs such as Mamba2, balancing efficiency and model quality.

The goal is to achieve inference adaptation on the Ascend NPU platform and to verify architectural compatibility and inference accuracy.

Detailed information for this phase:
1. Model name: [Jet-Nemotron-2B](https://www.modelscope.cn/models/nv-community/Jet-Nemotron-2B)
2. Development platform: Atlas 800T A2 or A3

### Acceptance Criteria

**I. Deliverables**

This phase extends the existing framework/repository and requires development code and documentation. The main development points are:

1. Submit a PR
  - Complete inference adaptation code.
  - Add test cases under `test/registered/ascend` to guard accuracy.
  - Update the model support document `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_models.mdx`.
  - Provide a full PR description that includes accuracy and performance results, following community conventions.
  - (Optional) Supplement `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` with problems encountered and solved during adaptation.
2. Update the Issue
  - The issue may include development experience, problems encountered, and their solutions.

**II. Acceptance Criteria**

1) Accuracy / Performance requirements
- Accuracy: The adapted model's accuracy on representative evaluation tasks must not deviate more than 1% from the official baseline.
- Performance: N/A

2) The PR must contain complete inference code, test code, and a PR description.

### PR Merge
After local testing and verification, create a PR against the `main` branch of https://github.com/sgl-project/sglang.

### Point of Contact
@amote-i

Welcome to the community and thank you for your contribution 🎉!

---

## Task Name
Tencent-Hunyuan-Large General Capability Model Adaptation

### Task Description
Adapt the **Tencent-Hunyuan-Large** 389B MoE model. Hunyuan-Large demonstrates outstanding performance across commonsense understanding, question answering, mathematical reasoning, programming, and comprehensive tasks. With 389B total parameters and 52B activated parameters, it achieves a balance between super-large-scale model performance and high inference efficiency. The overall adaptation difficulty is moderate.

The goal is to fully implement inference adaptation on the Ascend NPU platform, ensuring efficient execution of the large-parameter model on NPUs with aligned accuracy.

Detailed information for this phase:
1. Model name: [Tencent-Hunyuan-Large](https://www.modelscope.cn/models/tencent/Hunyuan-Large)
2. Development platform: Atlas 800T A2 or A3

### Acceptance Criteria

**I. Deliverables**

This phase extends the existing framework/repository and requires development code and documentation. The main development points are:

1. Submit a PR
  - Complete inference adaptation code.
  - Add test cases under `test/registered/ascend` to guard accuracy.
  - Update the model support document `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_models.mdx`.
  - Provide a full PR description that includes accuracy and performance results, following community conventions.
  - (Optional) Supplement `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` with problems encountered and solved during adaptation.
2. Update the Issue
  - The issue may include development experience, problems encountered, and their solutions.

**II. Acceptance Criteria**

1) Accuracy / Performance requirements
- Accuracy: The adapted model's accuracy on representative evaluation tasks must not deviate more than 1% from the official baseline.
- Performance: N/A

2) The PR must contain complete inference code, test code, and a PR description.

### PR Merge
After local testing and verification, create a PR against the `main` branch of https://github.com/sgl-project/sglang.

### Point of Contact
@amote-i

Welcome to the community and thank you for your contribution 🎉!

---

## Task Name
Qwen3-ASR Multilingual Automatic Speech Recognition Model Adaptation

### Task Description
Adapt the **Qwen/Qwen3-ASR-1.7B** multilingual automatic speech recognition model. Released in January 2026, this 1.7B-parameter model offers high-accuracy multilingual transcription with a lightweight architecture suitable for real-time speech processing deployment. The adaptation must address NPU compatibility for speech input preprocessing and audio feature extraction operators.

The goal is to implement a complete speech recognition inference pipeline on the Ascend NPU and ensure transcription accuracy meets the target.

Detailed information for this phase:
1. Model name: [Qwen3-ASR-1.7B](https://www.modelscope.cn/models/qwen/Qwen3-ASR-1.7B)
2. Development platform: Atlas 800T A2 or A3

### Acceptance Criteria

**I. Deliverables**

This phase extends the existing framework/repository and requires development code and documentation. The main development points are:

1. Submit a PR
  - Complete inference adaptation code (including speech preprocessing and feature extraction adaptation).
  - Add test cases under `test/registered/ascend` to guard accuracy.
  - Update the model support document `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_models.mdx`.
  - Provide a full PR description that includes accuracy and performance results, following community conventions.
  - (Optional) Supplement `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` with problems encountered and solved during adaptation.
2. Update the Issue
  - The issue may include development experience, problems encountered, and their solutions.

**II. Acceptance Criteria**

1) Accuracy / Performance requirements
- Accuracy: The adapted model's character/word error rate on representative multilingual test sets must not differ by more than 1% (absolute) from the official baseline, which can be obtained from the model card on the download page.
- Performance: N/A

2) The PR must contain complete inference code, test code, and a PR description.

### PR Merge
After local testing and verification, create a PR against the `main` branch of https://github.com/sgl-project/sglang.

### Point of Contact
@amote-i

Welcome to the community and thank you for your contribution 🎉!

---

## Task Name
NVILA-8B Native Vision-Language Model Adaptation

### Task Description
Adapt the **NVILA-8B** vision-language model. NVILA is a family of open vision-language models that simultaneously optimize efficiency and accuracy. Built upon the VILA model, it uses a "scale-then-compress" approach to efficiently process high-resolution images and long videos. This adaptation must validate both inference accuracy and inference performance, ensuring efficient execution of multimodal tasks on NPUs.

Detailed information for this phase:
1. Model name: [NVILA-8B](https://www.modelscope.cn/models/NVILA/NVILA-8B)
2. Development platform: Atlas 800T A2 or A3

### Acceptance Criteria

**I. Deliverables**

This phase extends the existing framework/repository and requires development code and documentation. The main development points are:

1. Submit a PR
  - Complete inference adaptation code (covering image and video inputs).
  - Add test cases under `test/registered/ascend` to guard accuracy and performance.
  - Update the model support document `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_models.mdx`.
  - Provide a full PR description that includes accuracy and performance results, following community conventions.
  - (Optional) Supplement `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` with problems encountered and solved during adaptation.
2. Update the Issue
  - The issue may include development experience, problems encountered, and their solutions.

**II. Acceptance Criteria**

1) Accuracy / Performance requirements
- Accuracy: The adapted model's accuracy on representative multimodal evaluation tasks must not deviate more than 1% from the official baseline.
- Performance: The PR must include inference throughput/latency test results, with no significant performance regression (using equivalent-scale model baselines within SGLang as a reference).

2) The PR must contain complete inference code, test code, and a PR description.

### PR Merge
After local testing and verification, create a PR against the `main` branch of https://github.com/sgl-project/sglang.

### Point of Contact
@amote-i

Welcome to the community and thank you for your contribution 🎉!

---

## Task Name
step-3.5-flash Large Language Model Adaptation and Optimization

### Task Description
Adapt the **step-3.5-flash** lightweight high-speed large language model. This model offers low latency, high throughput, and strong general dialogue capabilities, and must be adapted to the Ascend NPU hardware architecture. The work includes model weight migration, inference logic adaptation, operator compatibility modifications, and optimization of inference scheduling logic to ensure stable, efficient execution under the SGLang framework, meeting the requirements of general dialogue, copy generation, and intelligent Q&A scenarios.

Detailed information for this phase:
1. Model name: [step-3.5-flash](https://www.modelscope.cn/models/stepfun-ai/step-3.5-flash)
2. Development platform: Atlas 800T A2 or A3

### Acceptance Criteria

**I. Deliverables**

This phase extends the existing framework/repository and requires development code and documentation. The main development points are:

1. Submit a PR
  - Complete inference adaptation and scheduling optimization code.
  - Add functional and performance test cases under `test/registered/ascend`.
  - Update the model support document `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_models.mdx`.
  - Provide a full PR description that includes accuracy and performance results, following community conventions.
  - (Optional) Supplement `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` with problems encountered and solved during adaptation.
2. Update the Issue
  - The issue may include development experience, problems encountered, and their solutions.

**II. Acceptance Criteria**

1) Accuracy / Performance requirements
- Accuracy: The output quality of the adapted model on representative general dialogue and copy generation tests must not deviate more than 1% from the official baseline.
- Performance: The PR must showcase low-latency, high-throughput performance test results, with no significant performance regression.

2) The PR must contain complete inference code, test code, and a PR description.

### PR Merge
After local testing and verification, create a PR against the `main` branch of https://github.com/sgl-project/sglang.

### Point of Contact
@amote-i

Welcome to the community and thank you for your contribution 🎉!

---

## Task Name
DeepSeek-OCR2 Multimodal OCR Model Adaptation and Optimization

### Task Description
Adapt the **DeepSeek-OCR2** high-precision image-text recognition model. This model supports multilingual text recognition, handwriting recognition, and complex layout document parsing. Based on the SGLang framework, the adaptation covers the image preprocessing, visual encoding, and text decoding pipeline for NPUs, optimizes image inference compute footprint, and ensures that OCR recognition accuracy and inference speed meet practical requirements.

Detailed information for this phase:
1. Model name: [DeepSeek-OCR2](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-OCR2)
2. Development platform: Atlas 800T A2 or A3

### Acceptance Criteria

**I. Deliverables**

This phase extends the existing framework/repository and requires development code and documentation. The main development points are:

1. Submit a PR
  - Complete inference adaptation code for image recognition and multi-scenario layout recognition.
  - Add test cases for image recognition and multi-scenario layout recognition under `test/registered/ascend`.
  - Update the model support document `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_models.mdx`.
  - Provide a full PR description that includes accuracy and performance results, following community conventions.
  - (Optional) Supplement `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` with problems encountered and solved during adaptation.
2. Update the Issue
  - The issue may include development experience, problems encountered, and their solutions.

**II. Acceptance Criteria**

1) Accuracy / Performance requirements
- Accuracy: The adapted model's text recognition accuracy and multi-scenario layout parsing metrics must not differ by more than 1% (absolute) from the official baseline.
- Performance: The PR must include inference throughput/latency test results, with no significant performance regression.

2) The PR must contain complete inference code, test code, and a PR description.

### PR Merge
After local testing and verification, create a PR against the `main` branch of https://github.com/sgl-project/sglang.

### Point of Contact
@amote-i

Welcome to the community and thank you for your contribution 🎉!

---

## Task Name
gemma4 General Open-Source Large Language Model Adaptation and Optimization

### Task Description
Adapt Google's **gemma4** next-generation open-source general-purpose large language models, including Gemma 4 31B, E4B, and E2B variants. The models are compatible with the native Transformer architecture and must be adapted to the Ascend NPU operator ecosystem. The work includes weight conversion and inference pipeline modification, optimization of the context window and text generation speed, alignment with the original inference quality, and support for general inference and fine-tuning scenarios within the open-source community.

Detailed information for this phase:
1. Model name: [gemma-4-31b-it](https://www.modelscope.cn/models/google/gemma-4-31b-it) (31B example; E4B/E2B follow the same pattern)
2. Development platform: Atlas 800T A2 or A3

### Acceptance Criteria

**I. Deliverables**

This phase extends the existing framework/repository and requires development code and documentation. The main development points are:

1. Submit a PR
  - Complete inference adaptation code (covering three model sizes).
  - Add test cases for text generation, contextual dialogue, and accuracy validation under `test/registered/ascend`.
  - Update the model support document `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_support_models.mdx`.
  - Provide a full PR description that includes accuracy and performance results, following community conventions.
  - (Optional) Supplement `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_faq.mdx` with problems encountered and solved during adaptation.
2. Update the Issue
  - The issue may include development experience, problems encountered, and their solutions.

**II. Acceptance Criteria**

1) Accuracy / Performance requirements
- Accuracy: The adapted model's accuracy on representative evaluation tasks must not deviate more than 1% from the official baseline.
- Performance: N/A (If performance optimization is included, attach test results in the PR.)

2) The PR must contain complete inference code, test code, and a PR description.

### PR Merge
After local testing and verification, create a PR against the `main` branch of https://github.com/sgl-project/sglang.

### Point of Contact
@amote-i

Welcome to the community and thank you for your contribution 🎉!
