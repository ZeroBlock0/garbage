import tensorflow as tf
import numpy as np
import os
from PIL import Image
import gc
import sys
import shutil
from datetime import datetime
import glob


def load_and_preprocess_image(image_path):
    """加载并预处理图片"""
    # 读取图片
    img = Image.open(image_path)
    # 调整大小到模型需要的尺寸
    img = img.resize((150, 150))
    # 转换为numpy数组
    img_array = np.array(img)
    # 确保图片是RGB格式
    if img_array.shape[-1] == 4:  # 如果是RGBA格式
        img_array = img_array[..., :3]
    # 归一化
    img_array = img_array.astype("float32") / 255.0
    # 扩展维度以匹配模型输入
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def create_output_structure():
    """创建输出目录结构"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("inference_output", timestamp)

    # 创建分类目录
    cat_dir = os.path.join(base_output_dir, "cats")
    dog_dir = os.path.join(base_output_dir, "dogs")

    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(dog_dir, exist_ok=True)

    return base_output_dir, cat_dir, dog_dir


def find_latest_model():
    """查找最新的模型文件夹中的最佳模型"""
    model_dirs = glob.glob("model_checkpoints/*/")
    if not model_dirs:
        raise FileNotFoundError("未找到模型文件夹！请先训练模型。")

    # 获取最新的模型文件夹
    latest_dir = max(model_dirs, key=os.path.getctime)
    best_model_path = os.path.join(latest_dir, "best_model.keras")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"在 {latest_dir} 中未找到 best_model.keras！")

    return best_model_path


def main():
    try:
        # 创建输出目录
        base_output_dir, cat_dir, dog_dir = create_output_structure()
        
        # 初始化结果列表
        results = []
        
        # 加载模型
        best_model_path = os.path.join('models', 'best_model.keras')
        interrupted_model_path = os.path.join('models', 'interrupted_model.keras')
        
        print("正在加载模型...")
        if os.path.exists(best_model_path):
            try:
                model = tf.keras.models.load_model(best_model_path)
                print("已成功加载best_model.keras")
            except Exception as e:
                print(f"加载best_model.keras失败: {str(e)}")
                if os.path.exists(interrupted_model_path):
                    model = tf.keras.models.load_model(interrupted_model_path)
                    print("已加载interrupted_model.keras作为备选")
                else:
                    raise FileNotFoundError("未找到可用的模型文件")
        elif os.path.exists(interrupted_model_path):
            model = tf.keras.models.load_model(interrupted_model_path)
            print("未找到best_model.keras，已加载interrupted_model.keras")
        else:
            raise FileNotFoundError("未找到任何可用的模型文件")
        
        # 获取需要预测的图片
        inference_dir = 'inference'
        if not os.path.exists(inference_dir):
            raise FileNotFoundError("未找到inference目录，请创建该目录并放入需要预测的图片")

        image_files = [f for f in os.listdir(inference_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not image_files:
            print("inference目录中没有找到图片文件")
            return

        print(f"\n找到 {len(image_files)} 张图片待预测")
        
        # 处理每张图片
        for img_name in image_files:
            try:
                img_path = os.path.join(inference_dir, img_name)
                
                # 加载和预处理图片
                img_array = load_and_preprocess_image(img_path)
                if img_array is None:
                    continue

                # 进行预测
                prediction = model.predict(img_array, verbose=0)[0][0]

                # 确定类别（0.5为阈值）
                is_dog = prediction > 0.5
                confidence = prediction if is_dog else 1 - prediction
                class_name = "dog" if is_dog else "cat"

                # 确定目标目录
                target_dir = dog_dir if is_dog else cat_dir

                # 复制文件到对应目录
                filename = os.path.basename(img_path)
                base_name, ext = os.path.splitext(filename)
                new_filename = f"{base_name}_{class_name}_{confidence:.2f}{ext}"
                target_path = os.path.join(target_dir, new_filename)
                shutil.copy2(img_path, target_path)

                # 收集结果
                results.append({
                    "filename": filename,
                    "prediction": class_name,
                    "confidence": confidence,
                })

                print(f"处理: {filename} -> {class_name} (置信度: {confidence:.2f})")

            except Exception as e:
                print(f"处理 {img_name} 时出错: {str(e)}")
                continue

        # 保存推理结果到文本文件
        results_file = os.path.join(base_output_dir, "inference_results.txt")
        with open(results_file, "w", encoding="utf-8") as f:
            f.write("推理结果汇总:\n")
            f.write("=" * 50 + "\n")
            for result in results:
                f.write(f"文件名: {result['filename']}\n")
                f.write(f"预测类别: {result['prediction']}\n")
                f.write(f"置信度: {result['confidence']:.2f}\n")
                f.write("-" * 50 + "\n")

        print(f"\n推理完成! 共处理 {len(results)} 张图片")
        print(f"结果保存在: {base_output_dir}")
        print(f"详细结果已保存到: {results_file}")

    except Exception as e:
        print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
