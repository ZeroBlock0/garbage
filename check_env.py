import os
import sys
import tensorflow as tf

def check_environment():
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    
    # 检查CUDA环境变量
    cuda_path = os.environ.get('CUDA_PATH')
    print(f"CUDA_PATH: {cuda_path}")
    
    # 检查PATH中的CUDA相关目录
    path = os.environ.get('PATH')
    cuda_in_path = [p for p in path.split(';') if 'cuda' in p.lower()]
    print("\nCUDA directories in PATH:")
    for p in cuda_in_path:
        print(f"  {p}")
    
    # 检查GPU可用性
    print("\nGPU Information:")
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    print(f"GPU support: {tf.test.is_built_with_gpu_support()}")
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs: {len(physical_devices)}")
    print(f"GPU devices: {physical_devices}")

if __name__ == "__main__":
    check_environment()
