# nanoGPT

一个简洁、高效的 GPT 模型实现，用于训练和微调语言模型。

## 功能特性

- 🚀 **简洁实现**：核心代码清晰易懂，适合学习和研究
- 📚 **字符级训练**：支持字符级别的文本生成
- ⚡ **高效训练**：支持 PyTorch 2.0 编译加速（需要 CUDA）
- 🔧 **灵活配置**：支持配置文件或命令行参数覆盖
- 💾 **检查点保存**：自动保存训练检查点，支持断点续训
- 📊 **可选日志**：支持 wandb 训练日志（可选）

## 安装

### 环境要求

- Python 3.8+
- PyTorch 2.0.0+
- CUDA（可选，用于 GPU 训练）

### 安装步骤

1. 克隆或下载项目

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 准备数据集：

**Shakespeare 数据集：**
```bash
cd data/shakespeare_char
python prepare.py
cd ../..
```

**中文四大名著数据集：**
```bash
cd data/chinese
python prepare.py
cd ../..
```

注意：中文数据集需要确保 `data/chinese/` 目录下包含四大名著的文本文件。

## 使用方法

### 基本训练

使用默认配置训练：
```bash
python train.py
```

使用配置文件训练：
```bash
python train.py config/train_shakespeare_char.py
```

### 命令行参数覆盖

可以通过命令行参数覆盖配置：
```bash
python train.py config/train_shakespeare_char.py --batch_size=32 --learning_rate=1e-4
```

### 配置文件格式

配置文件是 Python 文件，包含训练参数：

```python
# 输出目录
out_dir = 'out-shakespeare-char'

# 数据集
dataset = 'shakespeare_char'
batch_size = 64
block_size = 256

# 模型配置
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# 训练配置
learning_rate = 1e-3
max_iters = 5000
```

## 主要配置参数

### 数据集相关
- `dataset`: 数据集名称（对应 `data/` 目录下的文件夹）
- `batch_size`: 批次大小
- `block_size`: 上下文长度（序列长度）

### 模型相关
- `n_layer`: Transformer 层数
- `n_head`: 注意力头数
- `n_embd`: 嵌入维度
- `dropout`: Dropout 比率
- `bias`: 是否使用偏置（默认 False）

### 训练相关
- `learning_rate`: 学习率
- `max_iters`: 最大训练迭代次数
- `warmup_iters`: 学习率预热步数
- `lr_decay_iters`: 学习率衰减步数
- `min_lr`: 最小学习率
- `weight_decay`: 权重衰减
- `beta1`, `beta2`: AdamW 优化器的动量参数
- `grad_clip`: 梯度裁剪阈值

### 系统相关
- `device`: 设备类型（'cuda' 或 'cpu'）
- `dtype`: 数据类型（'float32', 'bfloat16', 'float16'）
- `compile`: 是否使用 PyTorch 2.0 编译（CPU 模式下需要 C++ 编译器）

## 注意事项

### CPU 训练
- 在 CPU 模式下，`torch.compile` 会自动禁用（需要 C++ 编译器）
- 如需启用编译，请安装 Visual Studio Build Tools（Windows）或 GCC/Clang（Linux/Mac）

### 检查点
- 检查点保存在 `out_dir` 目录下
- 使用 `init_from = 'resume'` 可以从检查点恢复训练

### 分布式训练
- 支持多 GPU 分布式训练（DDP）
- 使用环境变量 `RANK`, `LOCAL_RANK`, `WORLD_SIZE` 配置

## 项目结构

```
nanoGPT/
├── train.py              # 训练脚本
├── model.py              # GPT 模型定义
├── configurator.py       # 配置解析器
├── sample.py             # Shakespeare 文本生成脚本
├── sample_chinese.py     # 中文文本生成脚本
├── requirements.txt      # 依赖列表
├── config/               # 配置文件目录
│   ├── train_shakespeare_char.py  # Shakespeare 训练配置
│   └── train_chinese.py           # 中文四大名著训练配置
├── data/                 # 数据集目录
│   ├── shakespeare_char/
│   │   ├── prepare.py    # 数据预处理脚本
│   │   ├── train.bin    # 训练数据
│   │   └── val.bin      # 验证数据
│   └── chinese/
│       ├── prepare.py    # 中文数据预处理脚本
│       ├── hongloumeng.txt    # 红楼梦
│       ├── sanguoyanyi.txt    # 三国演义
│       ├── shuihuzhuan.txt    # 水浒传
│       ├── xiyouji.txt        # 西游记
│       ├── train.bin    # 训练数据
│       └── val.bin      # 验证数据
└── out/                  # 输出目录（训练检查点）
    ├── out-shakespeare-char/  # Shakespeare 模型检查点
    └── out-chinese/           # 中文模型检查点
```

## 快速开始

### 1. Shakespeare 字符级模型

#### 数据预处理
```bash
cd data/shakespeare_char
python prepare.py
cd ../..
```

#### 训练模型
```bash
python train.py config/train_shakespeare_char.py
```

训练完成后，检查点会保存在 `out-shakespeare-char/ckpt.pt`。

#### 生成文本
```bash
# 方式1：通过命令行参数指定 out_dir
python sample.py --out_dir=out-shakespeare-char

# 方式2：修改 sample.py 中的参数
# 编辑 sample.py，设置：
# out_dir = 'out-shakespeare-char'
# 然后运行：
python sample.py
```

可以同时使用命令行参数覆盖其他参数：
```bash
python sample.py --out_dir=out-shakespeare-char --start="ROMEO:" --num_samples=5 --temperature=0.9
```

或者修改 `sample.py` 中的参数：
```python
init_from = 'resume'
out_dir = 'out-shakespeare-char'  # 指定检查点目录
start = '\n'  # 起始提示词
num_samples = 10  # 生成样本数量
max_new_tokens = 500  # 最大生成长度
temperature = 0.8  # 采样温度
top_k = 200  # Top-k 采样
```

### 2. 四大名著中文模型

#### 数据预处理
首先确保 `data/chinese/` 目录下有以下文件：
- `hongloumeng.txt` (红楼梦)
- `sanguoyanyi.txt` (三国演义)
- `shuihuzhuan.txt` (水浒传)
- `xiyouji.txt` (西游记)

然后运行预处理脚本：
```bash
cd data/chinese
python prepare.py
cd ../..
```

这会生成：
- `train.bin` - 训练数据
- `val.bin` - 验证数据
- `meta.pkl` - 字符编码映射

#### 训练模型
```bash
python train.py config/train_chinese.py
```

训练配置（针对 RTX 3060 优化）：
- `batch_size = 32` - 充分利用 12GB 显存
- `block_size = 512` - 适合中文文本的上下文长度
- `n_layer = 12, n_head = 12, n_embd = 768` - 中等规模模型
- `max_iters = 5000` - 训练步数（可根据需要调整）

训练完成后，检查点会保存在 `out-chinese/ckpt.pt`。

#### 生成文本
```bash
# 方式1：通过命令行参数指定 out_dir
python sample_chinese.py --out_dir=out-chinese

# 方式2：修改 sample_chinese.py 中的参数
# 编辑 sample_chinese.py，设置：
# out_dir = 'out-chinese'
# 然后运行：
python sample_chinese.py
```

可以同时使用命令行参数覆盖其他参数：
```bash
python sample_chinese.py --out_dir=out-chinese --start="话说" --num_samples=3 --temperature=0.7
```

或者修改 `sample_chinese.py` 中的参数：
```python
init_from = 'resume'
out_dir = 'out-chinese'  # 指定检查点目录
start = '\n'  # 起始提示词，例如：'话说'、'却说'、'且说' 等
num_samples = 5  # 生成样本数量
max_new_tokens = 500  # 最大生成长度
temperature = 0.8  # 采样温度（越低越确定）
top_k = 200  # Top-k 采样
```

### 3. 从检查点恢复训练

如果训练中断，可以从检查点恢复：
```bash
# 修改配置文件中的 init_from
# init_from = 'resume'  # 改为 'resume'

# 然后重新运行训练
python train.py config/train_chinese.py
```

## 示例

训练 Shakespeare 字符级模型：
```bash
python train.py config/train_shakespeare_char.py
```

训练完成后，检查点会保存在 `out-shakespeare-char/ckpt.pt`。

## 许可证

本项目基于 MIT 许可证开源。
