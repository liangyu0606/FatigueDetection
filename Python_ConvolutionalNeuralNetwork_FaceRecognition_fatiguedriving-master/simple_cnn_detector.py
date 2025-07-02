#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN疲劳检测模块 - 简化版
专注于使用训练好的模型进行疲劳检测
"""

import numpy as np
import cv2
import json
import os
from typing import Optional, Dict

# 尝试导入深度学习库
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow未安装，CNN功能将不可用")


class CNNFatigueDetector:
    """基于CNN的疲劳检测器 - 简化版"""
    
    def __init__(self, model_path: str = None):
        """
        初始化CNN疲劳检测器
        
        Args:
            model_path: 训练好的模型路径
        """
        self.model = None
        self.class_indices = None
        self.input_size = (224, 224)
        
        # 预测历史（用于平滑结果）
        self.prediction_history = []
        self.history_length = 5
        
        if TENSORFLOW_AVAILABLE and model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
        """
        try:
            # 加载模型
            self.model = load_model(model_path)
            print(f"✅ 成功加载模型: {model_path}")
            
            # 加载类别映射
            class_indices_path = os.path.join(os.path.dirname(model_path), 'class_indices.json')
            if os.path.exists(class_indices_path):
                with open(class_indices_path, 'r') as f:
                    self.class_indices = json.load(f)
                print(f"✅ 成功加载类别映射: {class_indices_path}")
            else:
                # 默认类别映射
                self.class_indices = {
                    'alert': 0,
                    'drowsy': 1, 
                    'eyes_closed': 2,
                    'yawning': 3
                }
                print("⚠️  使用默认类别映射")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        if image is None or image.size == 0:
            return None
        
        try:
            # 调整大小
            resized = cv2.resize(image, self.input_size)
            
            # 归一化
            normalized = resized.astype(np.float32) / 255.0
            
            # 添加批次维度
            batch_image = np.expand_dims(normalized, axis=0)
            
            return batch_image
            
        except Exception as e:
            print(f"图像预处理失败: {e}")
            return None
    
    def predict_fatigue(self, face_image: np.ndarray) -> Optional[Dict]:
        """
        预测疲劳状态
        
        Args:
            face_image: 人脸图像
            
        Returns:
            预测结果字典
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return None
        
        # 预处理图像
        processed_image = self.preprocess_image(face_image)
        if processed_image is None:
            return None
        
        try:
            # 预测
            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            
            # 获取类别名称
            class_name = None
            for name, idx in self.class_indices.items():
                if idx == predicted_class_idx:
                    class_name = name
                    break
            
            if class_name is None:
                class_name = f"class_{predicted_class_idx}"
            
            # 添加到历史记录
            self.prediction_history.append({
                'class': class_name,
                'confidence': confidence
            })
            
            # 保持历史记录长度
            if len(self.prediction_history) > self.history_length:
                self.prediction_history.pop(0)
            
            # 计算平滑后的结果
            smoothed_result = self._smooth_predictions()
            
            result = {
                'predicted_class': class_name,
                'confidence': confidence,
                'smoothed_class': smoothed_result['class'],
                'smoothed_confidence': smoothed_result['confidence'],
                'fatigue_level': self._map_to_fatigue_level(smoothed_result['class']),
                'all_predictions': predictions[0].tolist()
            }
            
            return result
            
        except Exception as e:
            print(f"疲劳预测失败: {e}")
            return None
    
    def _smooth_predictions(self) -> Dict:
        """
        平滑预测结果
        
        Returns:
            平滑后的预测结果
        """
        if not self.prediction_history:
            return {'class': 'unknown', 'confidence': 0.0}
        
        # 统计各类别的出现次数和平均置信度
        class_stats = {}
        for pred in self.prediction_history:
            class_name = pred['class']
            if class_name not in class_stats:
                class_stats[class_name] = {'count': 0, 'total_confidence': 0.0}
            
            class_stats[class_name]['count'] += 1
            class_stats[class_name]['total_confidence'] += pred['confidence']
        
        # 找到出现次数最多的类别
        best_class = max(class_stats.keys(), key=lambda x: class_stats[x]['count'])
        avg_confidence = class_stats[best_class]['total_confidence'] / class_stats[best_class]['count']
        
        return {
            'class': best_class,
            'confidence': avg_confidence
        }
    
    def _map_to_fatigue_level(self, predicted_class: str) -> str:
        """
        将预测类别映射到疲劳等级
        
        Args:
            predicted_class: 预测的类别
            
        Returns:
            疲劳等级
        """
        fatigue_mapping = {
            'alert': '正常',
            'drowsy': '疲劳',
            'eyes_closed': '重度疲劳',
            'yawning': '轻度疲劳'
        }
        
        return fatigue_mapping.get(predicted_class, '未知')
    
    def get_fatigue_analysis(self, face_image: np.ndarray) -> Optional[Dict]:
        """
        获取疲劳分析结果
        
        Args:
            face_image: 人脸图像
            
        Returns:
            疲劳分析结果
        """
        prediction_result = self.predict_fatigue(face_image)
        
        if prediction_result is None:
            return None
        
        # 构建分析结果
        analysis = {
            'fatigue_detected': prediction_result['smoothed_class'] != 'alert',
            'fatigue_level': prediction_result['fatigue_level'],
            'confidence': prediction_result['smoothed_confidence'],
            'raw_prediction': prediction_result['predicted_class'],
            'recommendation': self._generate_recommendation(prediction_result['fatigue_level'])
        }
        
        return analysis
    
    def _generate_recommendation(self, fatigue_level: str) -> str:
        """
        根据疲劳等级生成建议
        
        Args:
            fatigue_level: 疲劳等级
            
        Returns:
            建议文本
        """
        recommendations = {
            '正常': '状态良好，继续保持',
            '轻度疲劳': '建议适当休息，保持警觉',
            '疲劳': '建议立即休息，避免继续驾驶',
            '重度疲劳': '严重疲劳，必须立即停车休息'
        }
        
        return recommendations.get(fatigue_level, '请注意安全')
    
    def is_available(self) -> bool:
        """
        检查检测器是否可用
        
        Returns:
            是否可用
        """
        return TENSORFLOW_AVAILABLE and self.model is not None
