import tensorflow as tf
from tensorflow.keras import layers, models
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc
import sys

# 基础配置
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32

def create_model():
    model = models.Sequential([
        # 第一个卷积块
        layers.Conv2D(64, (3, 3), padding='same', input_shape=(*IMAGE_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 第二个卷积块
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 第三个卷积块
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 第四个卷积块
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 全连接层
        layers.Flatten(),
        layers.Dense(1024),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def cleanup():
    """清理资源和内存"""
    try:
        # 清理 TensorFlow 会话
        tf.keras.backend.clear_session()
        
        # 释放 GPU 内存
        if tf.config.list_physical_devices('GPU'):
            for device in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.reset_memory_stats(device)
        
        # 强制进行垃圾回收
        gc.collect()
        
        print("\n已清理内存和相关资源")
    except Exception as e:
        print(f"清理资源时出错: {str(e)}")

def main():
    try:
        # 数据生成器
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # 加载数据
        train_generator = train_datagen.flow_from_directory(
            'dataset/train',
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            'dataset/validation',
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )
        
        # 创建保存目录
        os.makedirs('models', exist_ok=True)
        best_model_path = os.path.join('models', 'best_model.keras')
        
        # 创建检查点回调
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor='val_accuracy',
            save_best_only=True,  # 只保存最好的模型
            mode='max',
            verbose=1
        )
        
        # 创建早停回调
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,  # 10轮没有改善就停止
            restore_best_weights=True  # 恢复最佳权重
        )
        
        # 创建和训练模型
        model = create_model()
        
        # 训练模型
        history = model.fit(
            train_generator,
            epochs=70,
            validation_data=validation_generator,
            callbacks=[checkpoint_callback, early_stopping]
        )
        
        print(f"\n最佳模型已保存到: {os.path.abspath(best_model_path)}")
        
    except KeyboardInterrupt:
        print("\n检测到用户中断，正在保存当前模型...")
        # 保存中断时的模型
        interrupted_model_path = os.path.join('models', 'interrupted_model.keras')
        model.save(interrupted_model_path)
        print(f"中断时的模型已保存到: {os.path.abspath(interrupted_model_path)}")
        
    except Exception as e:
        print(f"训练过程发生错误: {str(e)}")
        raise
    finally:
        cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n检测到用户中断，正在清理资源...")
        cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"\n程序异常终止: {str(e)}")
        cleanup()
        sys.exit(1)
