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
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import json
import gc
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob


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
    base_dir = "dataset"
    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "validation")

    # Add validation of image files
    def validate_images(directory):
        invalid_images = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, filename)
                    try:
                        with Image.open(file_path) as img:
                            img.verify()  # Verify the image
                    except Exception as e:
                        invalid_images.append((file_path, str(e)))
        return invalid_images

    # Check for corrupted images before training
    logging.info("验证训练和验证集中的图像...")
    invalid_train = validate_images(train_dir)
    invalid_val = validate_images(validation_dir)

    if invalid_train or invalid_val:
        logging.error("发现损坏的图像文件:")
        for path, error in invalid_train + invalid_val:
            logging.error(f"文件: {path}, 错误: {error}")
            try:
                os.remove(path)  # Remove corrupted files
                logging.info(f"已删除损坏的文件: {path}")
            except OSError as e:
                logging.error(f"无法删除文件 {path}: {e}")
        logging.info("已清理损坏的图像文件")

    # Rest of your existing prepare_data code
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Add error handling for flow_from_directory
    try:
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=CONFIG['IMAGE_SIZE'],
            batch_size=CONFIG['BATCH_SIZE'],
            class_mode='binary'
        )

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=CONFIG['IMAGE_SIZE'],
            batch_size=CONFIG['BATCH_SIZE'],
            class_mode='binary'
        )
    except Exception as e:
        logging.error(f"创建数据生成器时出错: {str(e)}")
        raise

    return train_generator, validation_generator


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
            
            # 新增的第四个卷积块
            layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # 修改后的全连接层
            layers.Flatten(),
            layers.Dropout(CONFIG['DROPOUT_RATE']),
            # 第一个全连接层，增加神经元数量
            layers.Dense(1024, kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(CONFIG['DROPOUT_RATE']),
            # 添加一个中间层
            layers.Dense(512, kernel_regularizer=regularizer),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(CONFIG['DROPOUT_RATE']),
            # 输出层保持不变
            layers.Dense(1, activation='sigmoid')
        ])

        # Replace the CosineDecayRestarts with a simpler learning rate schedule
        initial_learning_rate = CONFIG['LEARNING_RATE']
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate,  # Use fixed learning rate instead
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


def get_gpu_memory_usage():
    try:
        import nvidia_smi
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2  # Convert to MB
    except:
        return 0

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.start_time = time.time()
        self.best_accuracy = 0
        self.best_loss = float('inf')
        # 添加计数器来控制日志频率
        self.batch_log_frequency = 50  # 每50个批次记录一次
        self.batch_counter = 0
    
    def on_epoch_end(self, epoch, logs=None):
        """仅在每个epoch结束时执行，完全避免batch级别操作"""
        logs = logs or {}
        
        # 计算基本指标
        current_accuracy = logs.get("val_accuracy", 0)
        current_loss = logs.get("val_loss", float('inf'))
        
        # 更新最佳指标并记录
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            logging.info(f"新最佳准确率: {self.best_accuracy:.4f}")
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            logging.info(f"新的最佳损失: {self.best_loss:.4f}")
        
        # 简单的进度打印
        logging.info(
            f"\n轮次 {epoch + 1} - "
            f"损失: {logs.get('loss', 0):.4f} - "
            f"准确率: {logs.get('accuracy', 0):.4f} - "
            f"验证损失: {current_loss:.4f} - "
            f"验证准确率: {current_accuracy:.4f}"
        )


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


# 添加置常量
CONFIG = {
    'IMAGE_SIZE': (150, 150),
    'BATCH_SIZE': 32,
    'EPOCHS': 30,
    'LEARNING_RATE': 0.001,
    'MIN_LEARNING_RATE': 1e-6,
    'PATIENCE': 10,
    'VALIDATION_SPLIT': 0.2,
    'L2_REGULARIZATION': 0.005,
    'DROPOUT_RATE': 0.4,
    'EARLY_STOPPING_DELTA': 0.0005
}

# 在main()函数开始前添加GPU配置检查
def check_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        logging.info(f"找到 {len(physical_devices)} 个 GPU:")
        for gpu in physical_devices:
            logging.info(f"  {gpu}")
        # 设置内存增长
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logging.info("GPU 内存增长已启用")
    else:
        logging.warning("未找到 GPU 设备，将使用 CPU 运行")

# Add memory management function
def limit_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Limit GPU memory to 80% of available memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 4)]  # Limit to 4GB
                )
        except RuntimeError as e:
            print(e)

def validate_config():
    required_keys = [
        'IMAGE_SIZE', 'BATCH_SIZE', 'EPOCHS', 'LEARNING_RATE',
        'MIN_LEARNING_RATE', 'PATIENCE', 'VALIDATION_SPLIT',
        'L2_REGULARIZATION', 'DROPOUT_RATE', 'EARLY_STOPPING_DELTA'
    ]
    
    for key in required_keys:
        if key not in CONFIG:
            raise ValueError(f"配置少必要参数: {key}")
            
    if CONFIG['BATCH_SIZE'] <= 0:
        raise ValueError("BATCH_SIZE 必须大于0")
    if CONFIG['EPOCHS'] <= 0:
        raise ValueError("EPOCHS 必须大于0")
    if not (0 < CONFIG['LEARNING_RATE'] <= 1):
        raise ValueError("LEARNING_RATE 必须在0到1之间")

def cleanup_memory():
    """清GPU和CPU内存"""
    import gc
    gc.collect()
    tf.keras.backend.clear_session()
    if tf.config.list_physical_devices('GPU'):
        for gpu in tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.reset_memory_stats(gpu)
            except:
                pass

def evaluate_model_performance(model, validation_ds):
    """评估模型性能并生成详细报告"""
    try:
        # 计算混淆矩阵
        predictions = model.predict(validation_ds)
        y_pred = (predictions > 0.5).astype(int)
        y_true = np.concatenate([y for x, y in validation_ds], axis=0)
        
        # 计算各种指标
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        
        logging.info("\n模型性能评估报:")
        logging.info(f"混淆矩阵:\n{cm}")
        logging.info(f"详细分类报告:\n{report}")
        
        return cm, report
    except Exception as e:
        logging.error(f"性能评估过程出错: {str(e)}")
        raise

def plot_training_metrics(history, checkpoint_dir):
    """绘制详细的训练指标图表"""
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        plt.plot(history.history[metric], label=f'训练{metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'验证{metric}')
        plt.title(f'模型{metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'training_metrics.png'))
    plt.close()

def get_latest_checkpoint_dir():
    """获取最新的检查点目录"""
    base_dir = "model_checkpoints"
    if not os.path.exists(base_dir):
        return None
    
    # 获取有检查点目录
    checkpoint_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d))]
    
    if not checkpoint_dirs:
        return None
    
    # 返回最新的目录
    return max(checkpoint_dirs, key=os.path.getmtime)

def find_best_model():
    checkpoint_dir = get_latest_checkpoint_dir()
    if not checkpoint_dir:
        return None
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "best_model-*"))
    if not checkpoint_files:
        return None
        
    # 从文件名中提取准确率并找到最高的
    best_file = max(checkpoint_files, 
                   key=lambda x: float(x.split('-')[-1]))
    
    print(f"最佳模型文件: {os.path.basename(best_file)}")
    print(f"验证准确率: {float(best_file.split('-')[-1])}")
    
    return best_file

def main():
    try:
        # 首先确保数据已下载
        if not download_and_extract_data():
            raise RuntimeError("��据集准备失败")
            
        # 创建数据生成器
        train_generator, validation_generator = prepare_data()
        
        # 创建模型
        model = create_model()
        
        # Initialize initial_epoch
        initial_epoch = 0
        
        # Check for existing checkpoint
        checkpoint_dir = get_latest_checkpoint_dir()
        
        if checkpoint_dir:
            logging.info(f"发现已有检查点目录: {checkpoint_dir}")
            
            # 询问用户是否要查看最佳模型
            view_best = input("是否查看最佳模型信息？(y/n): ").lower().strip() == 'y'
            if view_best:
                best_model = find_best_model()
                if best_model:
                    print("\n=== 最佳模型信息 ===")
                    print(f"模型路径: {best_model}")
                    print(f"验证准确率: {float(best_model.split('-')[-1]):.4f}")
                    print("=" * 20 + "\n")
            
            # 询问是否继续训练
            use_existing = input("是否使用已有检查点继续训练？(y/n): ").lower().strip() == 'y'
            
            if use_existing:
                # 查找最新的检查点文件
                checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "best_model-*"))
                if checkpoint_files:
                    try:
                        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
                        # 验证文件是否存在可访问
                        if os.path.isfile(latest_checkpoint) and os.access(latest_checkpoint, os.R_OK):
                            # 从文件名中提取轮次数
                            epoch_str = os.path.basename(latest_checkpoint).split('-')[1]
                            initial_epoch = int(epoch_str)
                            # 加载模型权重
                            model.load_weights(latest_checkpoint)
                            logging.info(f"从轮次 {initial_epoch} 继续训练")
                        else:
                            logging.warning(f"检查点文件无法访问: {latest_checkpoint}")
                            checkpoint_dir = None
                            initial_epoch = 0
                    except Exception as e:
                        logging.error(f"加载检查点时出错: {str(e)}")
                        checkpoint_dir = None
                        initial_epoch = 0
                else:
                    logging.warning("未找到有效的检查点文件")
                    checkpoint_dir = None
                    initial_epoch = 0
            else:
                checkpoint_dir = None
                initial_epoch = 0

        # 如果没有检查点或选择不使用已有检查点，创建新目录
        if not checkpoint_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join("model_checkpoints", timestamp)
            os.makedirs(checkpoint_dir, exist_ok=True)
            logging.info(f"创建新的检查点目录: {checkpoint_dir}")
        
        # 在训练开始前生成 TensorBoard 启动文件
        tensorboard_cmd = f"tensorboard --logdir={os.path.join(checkpoint_dir, 'logs')}"
        bat_file_path = os.path.join(checkpoint_dir, "start_tensorboard.bat")
        with open(bat_file_path, "w") as f:
            f.write(tensorboard_cmd)
        
        print(f"\n=== TensorBoard 启动文件置 ===")
        print(f"文件路径: {os.path.abspath(bat_file_path)}")
        print(f"双击该文件即可启动 TensorBoard\n")
        
        # 修改 TensorBoard 回调的配置
        tensorboard_callback = TensorBoard(
            log_dir=os.path.join(checkpoint_dir, 'logs'),
            histogram_freq=1,  # 每个epoch记录直方图
            write_graph=True,  # 写入计算图
            write_images=True,  # 写入图像
            update_freq='epoch',  # 每个epoch更新
            profile_batch='500,520'  # 性能分析的batch范围
        )
        
        # 扩展回调列表
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, "best_model-{epoch:02d}-{val_accuracy:.4f}"),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            tensorboard_callback,
            # 添加学习率记录器
            tf.keras.callbacks.CSVLogger(
                os.path.join(checkpoint_dir, 'training_log.csv'),
                separator=',',
                append=True
            )
        ]

        # 在训练前打印提示信息
        print("\n=== TensorBoard 使用说明 ===")
        print(f"1. 双击此文件启动 TensorBoard: {os.path.abspath(bat_file_path)}")
        print("2. 在浏览器中打开: http://localhost:6006")
        print("3. 等待几秒钟让数据加载完成")
        print("注意：训练开始后才会显示数据\n")

        # Calculate class weights
        class_weights = None
        if hasattr(train_generator, 'classes'):
            total_samples = len(train_generator.classes)
            n_classes = len(np.unique(train_generator.classes))
            class_counts = np.bincount(train_generator.classes)
            class_weights = {i: total_samples / (n_classes * count) 
                           for i, count in enumerate(class_counts)}
            logging.info(f"用类别权重: {class_weights}")
        
        # Calculate steps_per_epoch and validation_steps
        steps_per_epoch = len(train_generator)
        validation_steps = len(validation_generator)
        
        # Train model with initial_epoch parameter
        logging.info("开始训练模型...")
        history = model.fit(
            train_generator,
            epochs=CONFIG['EPOCHS'],
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
        )
        
        # Evaluate model
        logging.info("评估最终模型...")
        final_results = model.evaluate(validation_generator, verbose=1)
        metrics_names = model.metrics_names
        for name, value in zip(metrics_names, final_results):
            logging.info(f"最终 {name}: {value:.4f}")
        
        # Save model
        final_model_path = os.path.join(checkpoint_dir, "final_model.keras")
        model.save(final_model_path)
        logging.info(f"最终模型已保存至: {final_model_path}")
        
        # Plot training history
        plot_training_history(history)
        
        # 保存训练历史和性能指标
        history_dict = history.history
        metrics_file = os.path.join(checkpoint_dir, 'training_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(history_dict, f)
        logging.info(f"训练指标已保存至: {metrics_file}")
        
        # 保存模型架构图
        tf.keras.utils.plot_model(
            model,
            to_file=os.path.join(checkpoint_dir, 'model_architecture.png'),
            show_shapes=True,
            show_layer_names=True
        )
        
        # 在训练束后评估模型
        cm, report = evaluate_model_performance(model, validation_generator)
        
        # 保存评估结果
        with open(os.path.join(checkpoint_dir, 'evaluation_report.txt'), 'w') as f:
            f.write(f"混淆矩阵:\n{cm}\n\n")
            f.write(f"详细分类报告:\n{report}")
        
        # 绘制训练指标
        plot_training_metrics(history, checkpoint_dir)
        
        # 保存 TensorBoard 启动命令到文件
        tensorboard_cmd = f"tensorboard --logdir={os.path.join(checkpoint_dir, 'logs')}"
        bat_file_path = os.path.join(checkpoint_dir, "start_tensorboard.bat")
        with open(bat_file_path, "w") as f:
            f.write(tensorboard_cmd)
        
        print(f"\n=== TensorBoard 启动文件位置 ===")
        print(f"文件路径: {os.path.abspath(bat_file_path)}")
        print(f"双击该文件即可启动 TensorBoard\n")
        
        # 在训练后添加此代码来验证日志文件
        log_dir = os.path.join(checkpoint_dir, 'logs')
        print(f"\n检查日志文件夹内容：")
        if os.path.exists(log_dir):
            print(f"日志目录: {log_dir}")
            print("包含的文件:")
            for file in os.listdir(log_dir):
                print(f"- {file}")
        else:
            print("警告：日志目录不存在！")
        
    except Exception as e:
        logging.error(f"训练过程发生错误: {str(e)}")
        cleanup_memory()  # 发生错误时清理内存
        raise
    finally:
        cleanup_memory()  # 训练结后清理内存


if __name__ == "__main__":
    main()
