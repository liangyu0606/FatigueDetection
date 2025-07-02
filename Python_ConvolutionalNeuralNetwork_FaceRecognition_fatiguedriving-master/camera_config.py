#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摄像头配置文件
用于调整摄像头参数和图像处理设置
"""

class CameraConfig:
    """摄像头配置类"""
    
    def __init__(self):
        # 摄像头基本设置
        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 480
        self.CAMERA_FPS = 15  # 降低FPS以减少处理负担
        self.CAMERA_BUFFER_SIZE = 1
        
        # 亮度和对比度设置 - 针对暗环境优化
        self.BRIGHTNESS = 150      # 提高亮度 (0-255)
        self.CONTRAST = 140        # 提高对比度 (0-255)
        self.SATURATION = 110      # 饱和度 (0-255)
        self.GAIN = 50             # 增益 (0-100)
        self.AUTO_EXPOSURE = 0.75  # 自动曝光 (0-1)
        self.EXPOSURE = -4         # 曝光值 (负值表示更长曝光)
        self.WB_TEMPERATURE = 4000 # 白平衡温度
        
        # 图像增强设置
        self.DARK_THRESHOLD = 120           # 暗图像阈值
        self.BRIGHTNESS_ALPHA = 1.6         # 亮度调整系数
        self.BRIGHTNESS_BETA = 40           # 亮度调整偏移
        self.CLAHE_CLIP_LIMIT = 4.0         # CLAHE剪切限制
        self.CLAHE_TILE_SIZE = (8, 8)       # CLAHE瓦片大小
        self.GAMMA_CORRECTION = 0.7         # Gamma校正值
        self.ENHANCEMENT_WEIGHT = 0.8       # 增强图像权重
        
        # 帧率控制设置
        self.FRAME_SKIP_DETECTION = 3       # 每N帧进行一次人脸检测（减少以提高检测频率）
        self.FRAME_SKIP_DISPLAY = 1         # 每帧都显示（确保视频流畅）
        self.MAIN_LOOP_DELAY = 0.033        # 主循环延迟 (秒)
        
        # 性能优化设置
        self.ERROR_REPORT_INTERVAL = 30     # 错误报告间隔
        self.STATUS_REPORT_INTERVAL = 10.0  # 状态报告间隔 (秒)
        self.MAX_ERROR_COUNT = 100          # 最大错误计数
        
    def get_camera_properties(self):
        """获取摄像头属性字典"""
        import cv2
        return {
            cv2.CAP_PROP_FRAME_WIDTH: self.CAMERA_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT: self.CAMERA_HEIGHT,
            cv2.CAP_PROP_FPS: self.CAMERA_FPS,
            cv2.CAP_PROP_BUFFERSIZE: self.CAMERA_BUFFER_SIZE,
            cv2.CAP_PROP_AUTO_EXPOSURE: self.AUTO_EXPOSURE,
            cv2.CAP_PROP_BRIGHTNESS: self.BRIGHTNESS,
            cv2.CAP_PROP_CONTRAST: self.CONTRAST,
            cv2.CAP_PROP_SATURATION: self.SATURATION,
            cv2.CAP_PROP_GAIN: self.GAIN,
            cv2.CAP_PROP_EXPOSURE: self.EXPOSURE,
            cv2.CAP_PROP_AUTOFOCUS: 0,  # 关闭自动对焦
            cv2.CAP_PROP_AUTO_WB: 1,    # 启用自动白平衡
            cv2.CAP_PROP_WB_TEMPERATURE: self.WB_TEMPERATURE
        }
    
    def update_brightness_settings(self, brightness=None, contrast=None, gain=None):
        """更新亮度相关设置"""
        if brightness is not None:
            self.BRIGHTNESS = max(0, min(255, brightness))
        if contrast is not None:
            self.CONTRAST = max(0, min(255, contrast))
        if gain is not None:
            self.GAIN = max(0, min(100, gain))
    
    def update_enhancement_settings(self, dark_threshold=None, alpha=None, beta=None, gamma=None):
        """更新图像增强设置"""
        if dark_threshold is not None:
            self.DARK_THRESHOLD = max(50, min(200, dark_threshold))
        if alpha is not None:
            self.BRIGHTNESS_ALPHA = max(1.0, min(3.0, alpha))
        if beta is not None:
            self.BRIGHTNESS_BETA = max(0, min(100, beta))
        if gamma is not None:
            self.GAMMA_CORRECTION = max(0.3, min(2.0, gamma))
    
    def update_performance_settings(self, frame_skip_detection=None, frame_skip_display=None, loop_delay=None):
        """更新性能设置"""
        if frame_skip_detection is not None:
            self.FRAME_SKIP_DETECTION = max(1, min(10, frame_skip_detection))
        if frame_skip_display is not None:
            self.FRAME_SKIP_DISPLAY = max(1, min(10, frame_skip_display))
        if loop_delay is not None:
            self.MAIN_LOOP_DELAY = max(0.01, min(0.1, loop_delay))
    
    def save_config(self, filename="camera_config.json"):
        """保存配置到文件"""
        import json
        config_dict = {
            'camera': {
                'width': self.CAMERA_WIDTH,
                'height': self.CAMERA_HEIGHT,
                'fps': self.CAMERA_FPS,
                'buffer_size': self.CAMERA_BUFFER_SIZE,
                'brightness': self.BRIGHTNESS,
                'contrast': self.CONTRAST,
                'saturation': self.SATURATION,
                'gain': self.GAIN,
                'auto_exposure': self.AUTO_EXPOSURE,
                'exposure': self.EXPOSURE,
                'wb_temperature': self.WB_TEMPERATURE
            },
            'enhancement': {
                'dark_threshold': self.DARK_THRESHOLD,
                'brightness_alpha': self.BRIGHTNESS_ALPHA,
                'brightness_beta': self.BRIGHTNESS_BETA,
                'clahe_clip_limit': self.CLAHE_CLIP_LIMIT,
                'clahe_tile_size': self.CLAHE_TILE_SIZE,
                'gamma_correction': self.GAMMA_CORRECTION,
                'enhancement_weight': self.ENHANCEMENT_WEIGHT
            },
            'performance': {
                'frame_skip_detection': self.FRAME_SKIP_DETECTION,
                'frame_skip_display': self.FRAME_SKIP_DISPLAY,
                'main_loop_delay': self.MAIN_LOOP_DELAY,
                'error_report_interval': self.ERROR_REPORT_INTERVAL,
                'status_report_interval': self.STATUS_REPORT_INTERVAL,
                'max_error_count': self.MAX_ERROR_COUNT
            }
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
            print(f"配置已保存到 {filename}")
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def load_config(self, filename="camera_config.json"):
        """从文件加载配置"""
        import json
        import os
        
        if not os.path.exists(filename):
            print(f"配置文件 {filename} 不存在，使用默认配置")
            return
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 加载摄像头设置
            if 'camera' in config_dict:
                camera_config = config_dict['camera']
                self.CAMERA_WIDTH = camera_config.get('width', self.CAMERA_WIDTH)
                self.CAMERA_HEIGHT = camera_config.get('height', self.CAMERA_HEIGHT)
                self.CAMERA_FPS = camera_config.get('fps', self.CAMERA_FPS)
                self.CAMERA_BUFFER_SIZE = camera_config.get('buffer_size', self.CAMERA_BUFFER_SIZE)
                self.BRIGHTNESS = camera_config.get('brightness', self.BRIGHTNESS)
                self.CONTRAST = camera_config.get('contrast', self.CONTRAST)
                self.SATURATION = camera_config.get('saturation', self.SATURATION)
                self.GAIN = camera_config.get('gain', self.GAIN)
                self.AUTO_EXPOSURE = camera_config.get('auto_exposure', self.AUTO_EXPOSURE)
                self.EXPOSURE = camera_config.get('exposure', self.EXPOSURE)
                self.WB_TEMPERATURE = camera_config.get('wb_temperature', self.WB_TEMPERATURE)
            
            # 加载增强设置
            if 'enhancement' in config_dict:
                enhancement_config = config_dict['enhancement']
                self.DARK_THRESHOLD = enhancement_config.get('dark_threshold', self.DARK_THRESHOLD)
                self.BRIGHTNESS_ALPHA = enhancement_config.get('brightness_alpha', self.BRIGHTNESS_ALPHA)
                self.BRIGHTNESS_BETA = enhancement_config.get('brightness_beta', self.BRIGHTNESS_BETA)
                self.CLAHE_CLIP_LIMIT = enhancement_config.get('clahe_clip_limit', self.CLAHE_CLIP_LIMIT)
                self.CLAHE_TILE_SIZE = tuple(enhancement_config.get('clahe_tile_size', self.CLAHE_TILE_SIZE))
                self.GAMMA_CORRECTION = enhancement_config.get('gamma_correction', self.GAMMA_CORRECTION)
                self.ENHANCEMENT_WEIGHT = enhancement_config.get('enhancement_weight', self.ENHANCEMENT_WEIGHT)
            
            # 加载性能设置
            if 'performance' in config_dict:
                performance_config = config_dict['performance']
                self.FRAME_SKIP_DETECTION = performance_config.get('frame_skip_detection', self.FRAME_SKIP_DETECTION)
                self.FRAME_SKIP_DISPLAY = performance_config.get('frame_skip_display', self.FRAME_SKIP_DISPLAY)
                self.MAIN_LOOP_DELAY = performance_config.get('main_loop_delay', self.MAIN_LOOP_DELAY)
                self.ERROR_REPORT_INTERVAL = performance_config.get('error_report_interval', self.ERROR_REPORT_INTERVAL)
                self.STATUS_REPORT_INTERVAL = performance_config.get('status_report_interval', self.STATUS_REPORT_INTERVAL)
                self.MAX_ERROR_COUNT = performance_config.get('max_error_count', self.MAX_ERROR_COUNT)
            
            print(f"配置已从 {filename} 加载")
            
        except Exception as e:
            print(f"加载配置失败: {e}，使用默认配置")

# 创建全局配置实例
camera_config = CameraConfig()

# 尝试加载配置文件
camera_config.load_config()

if __name__ == "__main__":
    # 测试配置
    print("当前摄像头配置:")
    print(f"分辨率: {camera_config.CAMERA_WIDTH}x{camera_config.CAMERA_HEIGHT}")
    print(f"FPS: {camera_config.CAMERA_FPS}")
    print(f"亮度: {camera_config.BRIGHTNESS}")
    print(f"对比度: {camera_config.CONTRAST}")
    print(f"暗图像阈值: {camera_config.DARK_THRESHOLD}")
    print(f"跳帧检测: {camera_config.FRAME_SKIP_DETECTION}")
    
    # 保存默认配置
    camera_config.save_config()
