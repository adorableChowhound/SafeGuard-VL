# SafeGuard-VL: 面向策略自适应的图像安全护栏

<p align="center">
  <b>CVPR 2026</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.01228">论文</a> |
  <a href="https://huggingface.co/tyodd/SafeGuard-VL-RL">模型</a> |
  <a href="https://huggingface.co/datasets/tyodd/SafeEditBench">基准测试</a> |
  <a href="./README.md">English</a>
</p>

## 概述

SafeGuard-VL 是一个两阶段训练框架，用于构建**策略自适应的视觉安全护栏**。与以往过拟合于单一固定安全策略的护栏不同，SafeGuard-VL 能够在多样化且不断演变的安全策略间泛化。

- **阶段一 (SFT)：** 通过新颖的自重描述 (self-recaptioning) 机制，教会模型理解不安全的视觉语义。
- **阶段二 (RL)：** 采用 GRPO 和可验证奖励，实现策略感知的安全/不安全判别。

我们还提出了 **SafeEditBench**，一个跨策略评估基准，包含语义对齐的安全-不安全图像对，覆盖五种不同的安全策略 (L1-L5)。

<p align="center">
  <img src="assets/pipeline.png" width="90%">
</p>

## 资源

| 资源 | 链接 | 说明 |
|------|------|------|
| SafeGuard-VL-RL | [HuggingFace](https://huggingface.co/tyodd/SafeGuard-VL-RL) | 模型权重 (Qwen2.5-VL-7B-Instruct + 阶段二 RL) |
| Qwen2.5-VL-7B-Instruct | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 基础模型 |
| SafeEditBench | [HuggingFace](https://huggingface.co/datasets/tyodd/SafeEditBench) | 跨策略评估基准，包含 5 个策略级别 |
| UnsafeBench | [HuggingFace](https://huggingface.co/datasets/yiting/UnsafeBench) | 覆盖 9 类有害内容的安全基准 |
| LlavaGuard | [HuggingFace](https://huggingface.co/datasets/AIML-TUDA/LlavaGuard) | 训练数据与基准；测试集中的不安全图像用于构建 SafeEditBench |

## 安装

```bash
conda create -n safeguard python=3.10 -y
conda activate safeguard
pip install -r requirements.txt
```

## 快速开始

### 推理

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "tyodd/SafeGuard-VL-RL",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("tyodd/SafeGuard-VL-RL")

policy = "在此填写安全策略文本..."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.jpg"},
            {"type": "text", "text": f"Based on the following policy, determine whether this image is safe or unsafe.\n\nPolicy: {policy}"},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=512)
output = processor.batch_decode(
    output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
)[0]
print(output)
```

### 在 SafeEditBench 上评测

```bash
python eval/safeeditbench.py --model_path tyodd/SafeGuard-VL-RL --test_jsonl /path/to/test.jsonl --gpus 0,1,2,3
```

### 在 UnsafeBench 上评测

```bash
# 推理 + 评分
python eval/unsafebench.py --model_path tyodd/SafeGuard-VL-RL --test_jsonl /path/to/test.jsonl --gpus 0,1,2,3

# 仅评分（基于已有预测结果）
python eval/unsafebench.py --score_only --pred_file output/predictions.jsonl
```

### 生成安全对应图像（图像编辑）

通过两步流程生成语义对齐的安全版本图像：
(1) 安全评估模型生成编辑指令，(2) 图像编辑模型执行最小化编辑。

```bash
python data/safe_edit/safe_edit.py \
    --image_dir /path/to/unsafe_images \
    --output_dir /path/to/safe_images \
    --api_key YOUR_API_KEY \
    --base_url YOUR_BASE_URL
```

### 自重描述（阶段一数据构建）

生成包含丰富不安全语义的重描述训练数据。

```bash
python data/recaption/recaption.py \
    --input_jsonl /path/to/captions.jsonl \
    --output_jsonl /path/to/recaptioned.jsonl \
    --api_key YOUR_API_KEY \
    --base_url YOUR_BASE_URL \
    --model gemma-27b
```

### 训练

训练基于 [ms-swift](https://github.com/modelscope/ms-swift)（测试版本为 2025.9.16）。

```bash
# 阶段一：SFT
bash train/sft.sh

# 阶段二：GRPO（奖励函数在 train/plugin.py 中）
bash train/grpo.sh
```

## 项目结构

```
SafeGuard-VL/
├── README.md               # 英文说明
├── README_zh.md            # 中文说明
├── LICENSE
├── requirements.txt
├── .gitignore
├── assets/                  # README 展示用图
│   └── pipeline.png
├── data/
│   ├── recaption/           # 自重描述流程
│   │   └── recaption.py
│   └── safe_edit/           # 不安全→安全图像编辑
│       └── safe_edit.py
├── train/
│   ├── sft.sh               # 阶段一：SFT 训练脚本
│   ├── grpo.sh              # 阶段二：GRPO 训练脚本
│   └── plugin.py            # GRPO 奖励函数
└── eval/
    ├── safeeditbench.py     # SafeEditBench 评测
    └── unsafebench.py       # UnsafeBench 评测
```

## 引用

如果本工作对您有帮助，请引用：

```bibtex
@inproceedings{piao2026towards,
  title={Towards policy-adaptive image guardrail: Benchmark and method},
  author={Piao, Caiyong and Yan, Zhiyuan and Xu, Haoming and Zhao, Yunzhen and Lin, Kaiqing and Xu, Feiyang and Zhou, Shuigeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16614--16623},
  year={2026}
}
```

## 许可证

本项目基于 [Apache 2.0 许可证](LICENSE) 发布。

## 免责声明

本仓库包含视觉语言模型安全护栏的研究内容。数据集和模型仅供研究用途。部分数据可能包含用于评测目的的敏感内容。
