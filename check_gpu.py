import tensorflow as tf
import os
import sys

# 设置内存增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 系统信息
print("系统信息:")
print("-" * 50)
print(f"Python 版本: {sys.version}")
print(f"TensorFlow 版本: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"CUDA 是否可用: {tf.test.is_built_with_cuda()}")
print(f"GPU 是否可用: {tf.test.is_built_with_gpu_support()}")

# 环境变量
print("\n环境变量:")
print("-" * 50)
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', '未设置')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '未设置')}")

# 设备信息
print("\n设备信息:")
print("-" * 50)
devices = tf.config.list_physical_devices()
for device in devices:
    print(f"{device.device_type}: {device.name}")

# GPU 测试
if gpus:
    print("\nGPU 性能测试:")
    print("-" * 50)
    try:
        with tf.device('/GPU:0'):
            # 创建大矩阵
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            
            # 预热
            _ = tf.matmul(a, b)
            
            # 计时测试
            start = tf.timestamp()
            c = tf.matmul(a, b)
            end = tf.timestamp()
            
            print(f"矩阵乘法 (5000x5000) 耗时: {end - start:.2f} 秒")
    except Exception as e:
        print(f"GPU 测试失败: {e}")
else:
    print("\n故障排除:")
    print("-" * 50)
    print("1. 确认 NVIDIA 驱动是否最新")
    print("2. 运行 nvidia-smi 检查 CUDA 版本")
    print("3. 确认 CUDA 11.2 和 cuDNN 8.1.0 是否正确安装")
    print("4. 检查环境变量设置")