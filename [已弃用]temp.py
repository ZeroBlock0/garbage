import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import os
import zipfile
import urllib.request
from datetime import datetime
import matplotlib.pyplot as plt
import logging
import time


def download_and_extract_data():
    base_dir = "dataset"
    
    if os.path.exists(base_dir):
        print(f"数据集目录 '{base_dir}' 已存在，跳过下载步骤。")
        return True

    url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    file_name = "cats_and_dogs_filtered.zip"

    print("开始下载数据集...")
    try:
        urllib.request.urlretrieve(url, file_name)
    except urllib.error.URLError as e:
        print(f"下载失败: {e}")
        raise
    except Exception as e:
        print(f"发生未知错误: {e}")
        raise

    print("解压数据集...")
    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall()

    # 可以添加短暂延迟确保文件系统操作完成
    time.sleep(1)  # 等待1秒

    # 指定旧文件夹名称和新文件夹名称
    old_folder_name = file_name.replace('.zip', '')
    new_folder_name = "dataset"

    # 检查文件夹是否存在并重命名
    if os.path.exists(old_folder_name):
        os.rename(old_folder_name, new_folder_name)
        print(f"'{old_folder_name}' 已成功重命名为 '{new_folder_name}'")
    else:
        print(f"文件夹 '{old_folder_name}' 不存在")

    os.remove(file_name)
    print("数据集准备完成")

    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "validation")

    if not all(os.path.exists(d) for d in [train_dir, validation_dir]):
        raise RuntimeError("数据集结构不完整，请删除数据集目录后重新运行程序。")

    # Verify the directory structure
    required_subdirs = ['cats', 'dogs']
    
    for dir_path in [train_dir, validation_dir]:
        if not os.path.exists(dir_path):
            return False
        for subdir in required_subdirs:
            if not os.path.exists(os.path.join(dir_path, subdir)):
                return False
    
    return True


def prepare_data():
    try:
        base_dir = "dataset"
        train_dir = os.path.join(base_dir, "train")
        validation_dir = os.path.join(base_dir, "validation")

        # 训练集数据增强
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=30,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest",
            brightness_range=[0.8, 1.2],
            channel_shift_range=20.0
        )

        # 验证集只做归一化
        validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255
        )

        # 修改数据生成器配置
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=CONFIG['IMAGE_SIZE'],
            batch_size=CONFIG['BATCH_SIZE'],
            class_mode="binary",
            shuffle=True,
            seed=42
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=CONFIG['IMAGE_SIZE'],
            batch_size=CONFIG['BATCH_SIZE'],
            class_mode="binary",
            shuffle=False
        )

        # 将生成器包装为tf.data.Dataset
        train_ds = tf.data.Dataset.from_generator(
            lambda: train_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, *CONFIG['IMAGE_SIZE'], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        )
        
        val_ds = tf.data.Dataset.from_generator(
            lambda: validation_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, *CONFIG['IMAGE_SIZE'], 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        )

        logging.info(f"训练集样本数: {train_generator.samples}")
        logging.info(f"验证集样本数: {validation_generator.samples}")
        logging.info(f"类别分布: {train_generator.class_indices}")

        return train_ds, val_ds
    except Exception as e:
        logging.error(f"数据准备过程出错: {str(e)}")
        raise


def create_model():
    try:
        regularizer = tf.keras.regularizers.l2(CONFIG['L2_REGULARIZATION'])
        
        model = models.Sequential([
            # 输入层和第一个卷积块
            layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer, 
                         input_shape=(*CONFIG['IMAGE_SIZE'], 3)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # 第二个卷积块
            layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # 第三个卷积块
            layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # 全连接层
            layers.Flatten(),
            layers.Dropout(CONFIG['DROPOUT_RATE']),
            layers.Dense(512, kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(CONFIG['DROPOUT_RATE']),
            layers.Dense(1, activation='sigmoid')
        ])

        # 使用余弦退火学���率调度
        initial_learning_rate = CONFIG['LEARNING_RATE']
        decay_steps = 1000
        
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            decay_steps,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

        return model
    except Exception as e:
        logging.error(f"模型创建过程出错: {str(e)}")
        raise


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.best_accuracy = 0
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get("val_accuracy", 0)
        current_loss = logs.get("val_loss", float('inf'))
        
        # 记录训练进度
        logging.info(f"Epoch {epoch + 1}")
        logging.info(f"训练损失: {logs.get('loss'):.4f}")
        logging.info(f"训练准确率: {logs.get('accuracy'):.4f}")
        logging.info(f"验证损失: {current_loss:.4f}")
        logging.info(f"验证准确率: {current_accuracy:.4f}")
        
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.no_improvement_count = 0
            logging.info(f"新的最佳验证准确率: {self.best_accuracy:.4f}")
        else:
            self.no_improvement_count += 1
            
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            logging.info(f"新的最佳验证损失: {self.best_loss:.4f}")


def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # 绘制准确率
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    
    # 绘制损失
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def setup_logger(checkpoint_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(checkpoint_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# 添加配置常量
CONFIG = {
    'IMAGE_SIZE': (224, 224),     # 使用标准的图像尺寸
    'BATCH_SIZE': 32,
    'EPOCHS': 150,
    'LEARNING_RATE': 0.001,
    'MIN_LEARNING_RATE': 1e-6,    # 添加最小学习率限制
    'PATIENCE': 15,
    'VALIDATION_SPLIT': 0.2,
    'L2_REGULARIZATION': 0.01,    # L2正则化系数
    'DROPOUT_RATE': 0.5,          # Dropout比率
    'EARLY_STOPPING_DELTA': 0.001 # 早停阈值
}


def main():
    try:
        # Move CONFIG to top of file, before any function definitions
        
        # Set up logging first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join("model_checkpoints", timestamp)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = setup_logger(checkpoint_dir)
        
        # Download and verify data first
        if not download_and_extract_data():
            raise RuntimeError("数据集下载或验证失败，请检查数据集结构")
            
        # Set GPU memory growth
        set_memory_growth()
        
        # Prepare data
        train_ds, validation_ds = prepare_data()
        
        # Create model
        model = create_model()
        model.summary()
        
        # Define callback functions
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, "model_step_{epoch:02d}.keras"),
                save_freq='epoch',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=CONFIG['PATIENCE'],
                restore_best_weights=True,
                verbose=1,
                min_delta=CONFIG['EARLY_STOPPING_DELTA']
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=5,
                min_lr=CONFIG['MIN_LEARNING_RATE'],
                verbose=1
            ),
            CustomCallback(checkpoint_dir)
        ]
        
        # Calculate class weights
        class_weights = None
        if hasattr(train_ds, 'classes'):
            total_samples = len(train_ds.classes)
            n_classes = len(np.unique(train_ds.classes))
            class_counts = np.bincount(train_ds.classes)
            class_weights = {i: total_samples / (n_classes * count) 
                           for i, count in enumerate(class_counts)}
            logging.info(f"使用类别权重: {class_weights}")
        
        # Train model
        logging.info("开始训练模型...")
        history = model.fit(
            train_ds,
            epochs=CONFIG['EPOCHS'],
            validation_data=validation_ds,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate model
        logging.info("评估最终模型...")
        final_results = model.evaluate(validation_ds, verbose=1)
        metrics_names = model.metrics_names
        for name, value in zip(metrics_names, final_results):
            logging.info(f"最终 {name}: {value:.4f}")
        
        # Save model
        final_model_path = os.path.join(checkpoint_dir, "final_model.keras")
        model.save(final_model_path)
        logging.info(f"最终模型已保存至: {final_model_path}")
        
        # Plot training history
        plot_training_history(history)
        
    except Exception as e:
        logging.error(f"训练过程发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
