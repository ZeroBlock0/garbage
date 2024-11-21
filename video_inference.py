import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import time
import signal
import sys
import gc
from threading import Thread
from queue import Queue
import tkinter as tk
from tkinter import filedialog

# 保持与训练时相同的配置
IMAGE_SIZE = (150, 150)

# 在代码开始处添加 GPU 配置
def configure_gpu():
    """配置GPU设置"""
    try:
        # 获取可用的 GPU 设备
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                # 允许 GPU 内存动态增长
                tf.config.experimental.set_memory_growth(gpu, True)
                
            # 设置只使用第一个 GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"使用 GPU: {gpus[0].name}")
        else:
            print("未找到可用的 GPU 设备，将使用 CPU")
    except Exception as e:
        print(f"GPU 配置出错: {str(e)}")

class VideoProcessor:
    def __init__(self, video_path=None):
        if video_path and not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
        self.cap = cv2.VideoCapture(video_path if video_path else 0)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取输入视频的属性
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建输出视频写入器
        output_path = 'out.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        self.model = None
        self.is_running = True
        self.tracker = None
        self.tracking_box = None
        self.tracking_success = False
        # 添加级联分类器
        try:
            # 加载预训练的级联分类器（这里用haarcascade_frontalcatface.xml作为示例）
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
            self.cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"加载级联分类器失败: {str(e)}")
            self.cascade = None
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        self.process_thread = None
        self.frame_buffer = []
        self.prediction_buffer = []
        self.cleaned_up = False

    def init_tracker(self):
        """初始化跟踪器"""
        # OpenCV 3 和 OpenCV 4 的跟踪器 API 不同
        # 对于 OpenCV 4.5.1+
        try:
            return cv2.TrackerCSRT.create()
        except AttributeError:
            # 对于旧版本的 OpenCV
            return cv2.legacy.TrackerCSRT_create()

    def cleanup(self):
        """清理资源"""
        if self.cleaned_up:
            return
            
        print("\n正在清理资源...")
        
        # 释放视频捕获和写入器
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()
            
        # 清理OpenCV窗口
        cv2.destroyAllWindows()
        
        # 清理GPU内存
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        tf.keras.backend.clear_session()
        
        # 触发垃圾回收
        gc.collect()
        
        self.cleaned_up = True
        print("资源清理完成")

    def signal_handler(self, signum, frame):
        """处理中断信号"""
        print("\n接收到中断信号，正在安全退出...")
        self.is_running = False

    def load_model(self):
        """加载模型"""
        try:
            # 配置 GPU
            configure_gpu()
            
            best_model_path = os.path.join('models', 'best_model.keras')
            interrupted_model_path = os.path.join('models', 'interrupted_model.keras')
            
            with tf.device('/GPU:0'):  # 指定使用 GPU
                if os.path.exists(best_model_path):
                    self.model = tf.keras.models.load_model(best_model_path)
                    print("已成功加载 best_model.keras (GPU)")
                elif os.path.exists(interrupted_model_path):
                    self.model = tf.keras.models.load_model(interrupted_model_path)
                    print("已加载 interrupted_model.keras (GPU)")
                else:
                    raise FileNotFoundError("未找到可用的模型文件")
            return True
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return False

    def detect_object(self, frame):
        """检测图像中的目标物体"""
        if self.cascade is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # 如果检测到物体，返回最大的边界框
        if len(objects) > 0:
            # 选择面积最大的边界框
            areas = [w * h for (x, y, w, h) in objects]
            max_index = areas.index(max(areas))
            return tuple(objects[max_index])
        return None

    def process_frame(self, frame):
        """处理单帧图像并返回预测结果和边界框"""
        try:
            # 修改缩放因子，保持更多细节
            scale_factor = 0.75  # 改为0.75以保留更多细节
            frame_resized = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
            
            original_height, original_width = frame.shape[:2]  # 使用原始frame的尺寸
            resized_height, resized_width = frame_resized.shape[:2]
            
            # 处理图像用于分类
            img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            img = img.resize(IMAGE_SIZE)
            img_array = np.array(img)
            img_array = img_array.astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # 使用 GPU 进行预测
            with tf.device('/GPU:0'):
                prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            # 如果没有在跟踪，尝试检测物体
            if not self.tracking_success:
                detected_box = self.detect_object(frame_resized)
                if detected_box is not None:
                    x, y, w, h = detected_box
                    # 增加边界框的padding
                    padding_x = int(w * 0.2)  # 增加到20%
                    padding_y = int(h * 0.2)
                    x = max(0, x - padding_x)
                    y = max(0, y - padding_y)
                    w = min(resized_width - x, w + 2 * padding_x)
                    h = min(resized_height - y, h + 2 * padding_y)
                    
                    # 将坐标转换回原始尺寸
                    x = int(x / scale_factor)
                    y = int(y / scale_factor)
                    w = int(w / scale_factor)
                    h = int(h / scale_factor)
                    
                    self.tracking_box = (x, y, w, h)
                    
                    # 重新初始化跟踪器，使用原始尺寸的frame
                    self.tracker = self.init_tracker()
                    self.tracker.init(frame, self.tracking_box)
                    self.tracking_success = True
                else:
                    # 如果没有检测到物体，使用更大的默认中心框
                    box_width = int(original_width * 0.4)  # 增加默认框大小
                    box_height = int(original_height * 0.4)
                    x = (original_width - box_width) // 2
                    y = (original_height - box_height) // 2
                    self.tracking_box = (x, y, box_width, box_height)
                    
                    self.tracker = self.init_tracker()
                    self.tracker.init(frame, self.tracking_box)
                    self.tracking_success = True
            else:
                # 使用原始尺寸的frame进行跟踪
                self.tracking_success, self.tracking_box = self.tracker.update(frame)
                
                if not self.tracking_success:
                    # 重新检测
                    detected_box = self.detect_object(frame_resized)
                    if detected_box is not None:
                        x, y, w, h = detected_box
                        # 转换回原始尺寸
                        x = int(x / scale_factor)
                        y = int(y / scale_factor)
                        w = int(w / scale_factor)
                        h = int(h / scale_factor)
                        
                        self.tracking_box = (x, y, w, h)
                        self.tracker = self.init_tracker()
                        self.tracker.init(frame, self.tracking_box)
                        self.tracking_success = True
            
            return img_array, self.tracking_box if self.tracking_success else None
            
        except Exception as e:
            print(f"处理帧时出错: {str(e)}")
            return None, None

    def draw_box(self, frame, box, class_name, confidence):
        """在帧上绘制边界框和标签"""
        if not box:
            return
            
        x, y, w, h = [int(v) for v in box]
        
        # 绘制矩形框
        color = (0, 255, 0) if class_name == "Dog" else (0, 0, 255)  # 狗是绿色，猫是红色
        thickness = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # 准备标签文本
        label = f"{class_name}: {confidence:.2f}"
        
        # 获取文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, 1)
        
        # 绘制标签背景
        cv2.rectangle(frame, (x, y - label_height - 10), (x + label_width, y), color, -1)
        
        # 绘制标签文本
        cv2.putText(frame, label, (x, y - 5), font, font_scale, (255, 255, 255), 1)

    def process_video(self):
        if not self.cap.isOpened():
            print("错误：无法打开视频捕获设备或视频文件")
            return

        print("开始处理视频...")
        processed_frames = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                processed_frames += 1
                
                # 处理帧
                img_array, box = self.process_frame(frame)
                
                if img_array is not None:
                    # 获取预测结果
                    with tf.device('/GPU:0'):
                        prediction = self.model.predict(img_array, verbose=0)[0][0]
                    
                    # 确定类别
                    is_dog = prediction > 0.5
                    confidence = prediction if is_dog else 1 - prediction
                    class_name = "Dog" if is_dog else "Cat"
                    
                    # 在帧上绘制
                    self.draw_box(frame, box, class_name, confidence)
                
                # 写入处理后的帧
                self.out.write(frame)
                
                # 显示进度
                progress = (processed_frames / self.total_frames) * 100
                print(f"\r处理进度: {progress:.1f}%", end="")
                
        except Exception as e:
            print(f"\n视频处理出错: {str(e)}")
        finally:
            print("\n处理完成")
            self.cleanup()

def main():
    # 创建一个隐藏的 tkinter 根窗口
    root = tk.Tk()
    root.withdraw()

    # 打开文件选择对话框
    video_path = filedialog.askopenfilename(
        title='选择视频文件',
        filetypes=[
            ('视频文件', '*.mp4 *.avi *.mov *.mkv'),
            ('所有文件', '*.*')
        ]
    )

    if not video_path:
        print("未选择文件，程序退出")
        return

    print(f"已选择文件: {video_path}")
    print("开始处理，输出文件将保存为 'out.mp4'")
    
    # 使用选择的视频文件创建处理器
    processor = VideoProcessor(video_path)
    
    try:
        if not processor.load_model():  # 确保模型加载成功
            return
        processor.process_video()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n发生未预期的错误: {str(e)}")
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()
