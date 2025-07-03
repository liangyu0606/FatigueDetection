#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摄像头曝光调整工具
解决摄像头过曝问题
"""

import cv2
import sys
import time

class CameraExposureTool:
    """摄像头曝光调整工具"""
    
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.current_exposure = -6
        self.current_brightness = 0.4
        self.current_contrast = 0.6
        self.current_gain = 0.3
        
    def initialize_camera(self):
        """初始化摄像头"""
        print(f"🎥 正在初始化摄像头 {self.camera_index}...")
        
        try:
            # 尝试使用DirectShow后端（Windows）
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                # 如果DirectShow失败，尝试默认后端
                self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print("❌ 无法打开摄像头")
                return False
            
            # 设置基本参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print("✅ 摄像头初始化成功")
            return True
            
        except Exception as e:
            print(f"❌ 摄像头初始化失败: {e}")
            return False
    
    def apply_exposure_settings(self):
        """应用曝光设置"""
        if self.cap is None or not self.cap.isOpened():
            return False
        
        try:
            print("🔧 正在应用曝光设置...")
            
            # 禁用自动曝光
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            
            # 设置手动参数
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.current_exposure)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.current_brightness)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.current_contrast)
            self.cap.set(cv2.CAP_PROP_GAIN, self.current_gain)
            
            # 验证设置
            actual_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            actual_brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            actual_contrast = self.cap.get(cv2.CAP_PROP_CONTRAST)
            actual_gain = self.cap.get(cv2.CAP_PROP_GAIN)
            
            print(f"📊 当前设置:")
            print(f"  曝光: {actual_exposure:.2f}")
            print(f"  亮度: {actual_brightness:.2f}")
            print(f"  对比度: {actual_contrast:.2f}")
            print(f"  增益: {actual_gain:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 设置应用失败: {e}")
            return False
    
    def adjust_exposure(self, delta):
        """调整曝光值"""
        self.current_exposure = max(-13, min(-1, self.current_exposure + delta))
        self.apply_exposure_settings()
        print(f"🎯 曝光调整为: {self.current_exposure}")
    
    def adjust_brightness(self, delta):
        """调整亮度"""
        self.current_brightness = max(0, min(1, self.current_brightness + delta))
        self.apply_exposure_settings()
        print(f"💡 亮度调整为: {self.current_brightness:.2f}")
    
    def adjust_contrast(self, delta):
        """调整对比度"""
        self.current_contrast = max(0, min(1, self.current_contrast + delta))
        self.apply_exposure_settings()
        print(f"🎨 对比度调整为: {self.current_contrast:.2f}")
    
    def reset_settings(self):
        """重置到推荐设置"""
        self.current_exposure = -6
        self.current_brightness = 0.4
        self.current_contrast = 0.6
        self.current_gain = 0.3
        self.apply_exposure_settings()
        print("🔄 已重置到推荐设置")
    
    def run_interactive_tool(self):
        """运行交互式调整工具"""
        if not self.initialize_camera():
            return
        
        self.apply_exposure_settings()
        
        print("\n" + "=" * 60)
        print("🎥 摄像头曝光调整工具")
        print("=" * 60)
        print("键盘控制:")
        print("  Q/A - 增加/减少曝光")
        print("  W/S - 增加/减少亮度")
        print("  E/D - 增加/减少对比度")
        print("  R   - 重置到推荐设置")
        print("  ESC - 退出")
        print("=" * 60)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            # 显示当前参数
            cv2.putText(frame, f"Exposure: {self.current_exposure:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Brightness: {self.current_brightness:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Contrast: {self.current_contrast:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示控制提示
            cv2.putText(frame, "Q/A:Exposure W/S:Brightness E/D:Contrast R:Reset ESC:Exit", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Camera Exposure Tool", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('q') or key == ord('Q'):
                self.adjust_exposure(0.5)
            elif key == ord('a') or key == ord('A'):
                self.adjust_exposure(-0.5)
            elif key == ord('w') or key == ord('W'):
                self.adjust_brightness(0.05)
            elif key == ord('s') or key == ord('S'):
                self.adjust_brightness(-0.05)
            elif key == ord('e') or key == ord('E'):
                self.adjust_contrast(0.05)
            elif key == ord('d') or key == ord('D'):
                self.adjust_contrast(-0.05)
            elif key == ord('r') or key == ord('R'):
                self.reset_settings()
        
        self.cleanup()
    
    def get_optimal_settings(self):
        """获取当前优化的设置"""
        return {
            'exposure': self.current_exposure,
            'brightness': self.current_brightness,
            'contrast': self.current_contrast,
            'gain': self.current_gain
        }
    
    def cleanup(self):
        """清理资源"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 资源已清理")

def main():
    """主函数"""
    print("🚀 启动摄像头曝光调整工具")
    
    # 检查命令行参数
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print("❌ 无效的摄像头索引，使用默认值 0")
    
    print(f"📹 使用摄像头索引: {camera_index}")
    
    # 创建工具实例
    tool = CameraExposureTool(camera_index)
    
    try:
        # 运行交互式工具
        tool.run_interactive_tool()
        
        # 显示最终设置
        settings = tool.get_optimal_settings()
        print("\n" + "=" * 60)
        print("📋 最终优化设置:")
        print("=" * 60)
        print(f"曝光: {settings['exposure']}")
        print(f"亮度: {settings['brightness']:.2f}")
        print(f"对比度: {settings['contrast']:.2f}")
        print(f"增益: {settings['gain']:.2f}")
        print("\n💡 将这些设置应用到main.py中的setup_camera_exposure_control方法")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"❌ 工具运行失败: {e}")
    finally:
        tool.cleanup()

if __name__ == "__main__":
    main()
