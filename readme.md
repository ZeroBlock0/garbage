# 猫狗图像分类项目

## 环境配置

bash
激活 conda 环境
conda activate tf-gpu

## 项目结构

### 数据集目录 (dataset/)
dataset/
├── train/ # 训练集
│ ├── cats/ # 猫咪图片
│ │ ├── cat.1.jpg
│ │ ├── cat.2.jpg
│ │ └── ...
│ └── dogs/ # 狗狗图片
│ ├── dog.1.jpg
│ ├── dog.2.jpg
│ └── ...
└── validation/ # 验证集
├── cats/
│ ├── cat.1.jpg
│ ├── cat.2.jpg
│ └── ...
└── dogs/
├── dog.1.jpg
├── dog.2.jpg
└── ...

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

3. **图片预测**
   - 将待预测图片放入 `inference/` 目录
   - 运行 `inference.py` 进行预测
   - 查看 `inference_output/` 获取结果

4. **视频处理**
   - 运行 `video_inference.py` 处理视频
   - 输出文件将保存为 `out.mp4`

## 环境检查

可使用以下脚本检查环境配置：
- `check_gpu.py`: GPU 可用性检查
- `check_env.py`: 环境变量检查
- `fast_check_gpu.py`: 快速 GPU 检测

## 注意事项

- 确保已正确配置 CUDA 和 cuDNN
- 推荐使用 GPU 进行训练和推理
- 定期备份重要模型文件