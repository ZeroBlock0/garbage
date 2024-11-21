# 猫狗图像分类项目

## 环境配置

### 安装步骤

1. **安装 Conda**
   - 下载 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/download)
   - 按照安装向导完成安装

2. **创建环境**
   ```bash
   # 使用 environment.yml
   conda env create -f environment.yml

   # 或使用 requirements.txt
   conda create -n tf-gpu python=3.10
   conda activate tf-gpu
   pip install -r requirements.txt
   ```

3. **激活环境**
   ```bash
   conda activate tf-gpu
   ```

### 快速开始

## 项目结构

### 数据集目录 (dataset/)

dataset/
├── train/
│   ├── cats/
│   │   └── [放入训练集猫咪图片]
│   └── dogs/
│       └── [放入训练集狗狗图片]
└── validation/
    ├── cats/
    │   └── [放入验证集猫咪图片]
    └── dogs/
        └── [放入验证集狗狗图片]

### 模型目录 (models/)

models/
├── best_model.keras # ✨ 训练过程中的最佳模型
└── interrupted_model.keras # 💾 训练中断时的备份模型

### 推理目录 (inference/)

inference/
└── 📸 待预测图片
├── image1.jpg
├── image2.png
├── image3.jpeg
└── ...

### 推理输出目录 (inference_output/)

inference_output/
└── 📅 [时间戳目录]/
├── cats/ # 🐱 分类为猫的图片
│ └── [预测结果]
├── dogs/ # 🐕 分类为狗的图片
│ └── [预测结果]
└── inference_results.txt # 📊 详细预测结果

## 使用说明

1. **准备数据集**
   - 将训练图片放入 `dataset/train/` 对应目录
   - 将验证图片放入 `dataset/validation/` 对应目录

2. **训练模型**
   - 运行 `train.py` 开始训练
   - 最佳模型将保存在 `models/` 目录

3. **评估模型**
   - 运行 `evaluate.py` 评估模型性能
   - 查看详细评估指标：
     - 准确率 (Accuracy)
     - 精确率 (Precision)
     - 召回率 (Recall)
     - F1分数 (F1-Score)
     - 混淆矩阵
     - 分类报告

4. **图片预测**
   - 将待预测图片放入 `inference/` 目录
   - 运行 `inference.py` 进行预测
   - 查看 `inference_output/` 获取结果

5. **视频处理**
   - 运行 `video_inference.py` 处理视频
   - 输出文件将保存为 `out.mp4`

## 项目文件说明

### 核心脚本
- `train.py`: 模型训练脚本
- `evaluate.py`: 模型评估脚本
- `inference.py`: 图片预测脚本
- `video_inference.py`: 视频处理脚本

### 环境检查脚本
- `check_gpu.py`: GPU 可用性检查
- `check_env.py`: 环境变量检查
- `fast_check_gpu.py`: 快速 GPU 检测

### 环境配置文件
- `environment.yml`: Conda 环境完整配置
- `requirements.txt`: 依赖包列表

## 注意事项

- 推荐使用 GPU 进行训练和推理
- 定期备份重要模型文件
- 确保已正确配置 CUDA 和 cuDNN
- 确保图片格式为jpg、jpeg或png
- 建议训练集和验证集比例为8:2
- 每个类别的图片数量要相对平衡
- 图片文件名不影响训练，但建议有规律的命名
- 评估结果可能随数据集变化而变化