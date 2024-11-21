import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import os

def evaluate_model():
    """评估模型性能"""
    
    # 配置参数
    IMAGE_SIZE = (150, 150)
    BATCH_SIZE = 32
    MODEL_PATH = os.path.join('models', 'best_model.keras')
    TEST_DIR = os.path.join('dataset', 'validation')
    
    try:
        # 加载模型
        print("正在加载模型...")
        model = load_model(MODEL_PATH)
        
        # 创建数据生成器
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # 加载测试数据
        print("正在加载测试数据...")
        test_generator = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
        
        # 评估模型
        print("\n开始评估模型...")
        scores = model.evaluate(test_generator)
        print(f"\n测试集损失值: {scores[0]:.4f}")
        print(f"测试集准确率: {scores[1]:.4f}")
        
        # 获取预测结果
        print("\n正在生成详细预测报告...")
        predictions = model.predict(test_generator)
        predicted_labels = (predictions > 0.5).astype(int)
        true_labels = test_generator.classes
        
        # 使用sklearn计算混淆矩阵
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # 计算详细指标
        tn, fp, fn, tp = cm.ravel()
        total = len(true_labels)
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 打印详细报告
        print("\n详细评估报告:")
        print("-" * 50)
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数 (F1-Score): {f1_score:.4f}")
        print("-" * 50)
        print("\n混淆矩阵:")
        print("预测值 →")
        print("实际值 ↓")
        print(f"          负类    正类")
        print(f"负类      {tn}     {fp}")
        print(f"正类      {fn}     {tp}")
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(true_labels, predicted_labels, 
                                 target_names=['猫', '狗']))
        
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")

if __name__ == "__main__":
    evaluate_model()
