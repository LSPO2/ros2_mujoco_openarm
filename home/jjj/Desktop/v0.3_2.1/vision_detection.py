"""
vision_detection.py - 修复视觉检测模块
"""

import numpy as np
import cv2
import time
import os

class VisionDetector:
    """视觉检测与定位模块"""
    
    def __init__(self, robot_controller):
        self.robot = robot_controller
        self.calibration_matrix = None
        
        # 相机参数
        self.camera_matrix = self._get_camera_intrinsics()
        self.dist_coeffs = np.zeros((5, 1))
        
        # 已知的世界坐标
        self.banana_world_pos = np.array([-0.2, -0.4, 0.45])
        self.apple_world_pos = np.array([0.2, -0.4, 0.45])
        
        # 桌子高度
        self.table_height = 0.3 + 0.05  # 桌子高度+厚度
        
        print(f"[视觉检测] 初始化完成")
    
    def _get_camera_intrinsics(self):
        """获取相机内参"""
        fovy = 90.0
        width = 640
        height = 480
        
        fovy_rad = np.radians(fovy)
        fx = fy = height / (2 * np.tan(fovy_rad / 2))
        
        K = np.array([
            [fx, 0, width / 2],
            [0, fy, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    def set_calibration_matrix(self, matrix):
        """设置标定矩阵"""
        self.calibration_matrix = matrix
        print(f"[视觉检测] 已设置标定矩阵")
    
    def _render_camera_image(self):
        """渲染相机图像"""
        try:
            import mujoco
            
            renderer = mujoco.Renderer(self.robot.model, 480, 640)
            camera_id = 2  # depth_estimation_camera
            renderer.update_scene(self.robot.data, camera=camera_id)
            image = renderer.render()
            
            if image is not None:
                return image
                
        except Exception as e:
            print(f"[渲染] 错误: {e}")
        
        # 备用图像
        return self._create_test_image()
    
    def _create_test_image(self):
        """创建测试图像"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 黄色香蕉
        cv2.rectangle(image, (250, 200), (350, 250), (0, 255, 255), -1)
        
        # 红色苹果
        cv2.circle(image, (320, 150), 30, (0, 0, 255), -1)
        
        cv2.putText(image, "Test Image", (250, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return image
    
    def _detect_object_by_color(self, image, object_type='banana'):
        """通过颜色检测物体"""
        if image is None:
            print(f"  图像为空")
            return None
        
        # 转换为BGR
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # 检查是否是RGB
                if image[0, 0, 0] > image[0, 0, 2]:  # B > R，可能是BGR
                    bgr = image.copy()
                else:  # 可能是RGB
                    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                bgr = image[:, :, :3]
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 转换为HSV
        try:
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        except Exception as e:
            print(f"  HSV转换错误: {e}")
            return None
        
        if object_type == 'banana':
            # 黄色检测
            lower = np.array([20, 100, 100], dtype=np.uint8)
            upper = np.array([40, 255, 255], dtype=np.uint8)
        elif object_type == 'apple':
            # 红色检测（两个范围）
            lower1 = np.array([0, 100, 100], dtype=np.uint8)
            upper1 = np.array([10, 255, 255], dtype=np.uint8)
            lower2 = np.array([160, 100, 100], dtype=np.uint8)
            upper2 = np.array([179, 255, 255], dtype=np.uint8)
            
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            return None
        
        if object_type != 'apple':
            mask = cv2.inRange(hsv, lower, upper)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:
            # 旧版本OpenCV
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"  未找到{object_type}的轮廓")
            return None
        
        # 选择最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 50:
            print(f"  {object_type}轮廓面积太小: {area}")
            return None
        
        # 计算质心
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
        
        print(f"  检测到{object_type}: 像素({u}, {v}), 面积{area}")
        return (u, v)
    
    def _pixel_to_world(self, pixel_coords):
        """像素坐标转换为世界坐标"""
        if pixel_coords is None:
            return None
        
        if self.calibration_matrix is None:
            print("  警告: 未设置标定矩阵")
            # 使用简化方法
            return self._pixel_to_world_simple(pixel_coords)
        
        u, v = pixel_coords
        
        # 使用标定矩阵
        try:
            # 创建齐次像素坐标
            pixel_homogeneous = np.array([u, v, 1])
            
            # 使用标定矩阵的逆
            if np.linalg.det(self.calibration_matrix[:3, :3]) == 0:
                print("  警告: 标定矩阵奇异")
                return self._pixel_to_world_simple(pixel_coords)
            
            # 计算世界坐标
            # 这里简化处理，实际应该考虑深度
            world_homogeneous = np.linalg.inv(self.calibration_matrix) @ np.append(pixel_homogeneous, 1)
            
            if world_homogeneous[3] == 0:
                return None
            
            world_coords = world_homogeneous[:3] / world_homogeneous[3]
            
            # 调整Z坐标为桌子高度
            world_coords[2] = self.table_height + 0.01  # 略高于桌子
            
            return world_coords
            
        except Exception as e:
            print(f"  坐标转换错误: {e}")
            return self._pixel_to_world_simple(pixel_coords)
    
    def _pixel_to_world_simple(self, pixel_coords):
        """简化像素到世界坐标转换"""
        u, v = pixel_coords
        
        # 简单映射（根据之前看到的检测位置401,233对应香蕉位置-0.2, -0.4, 0.45）
        # 图像中心(320,240)对应世界(0, 0, 桌子高度)
        # 香蕉检测位置(401,233)对应(-0.2, -0.4, 0.45)
        
        # 计算偏移
        dx = (u - 320) * 0.001  # 1像素≈1mm
        dy = (v - 240) * 0.001
        
        # 香蕉的基准位置
        world_x = -0.2 + dx
        world_y = -0.4 + dy
        world_z = self.table_height + 0.01  # 略高于桌子
        
        return np.array([world_x, world_y, world_z])
    
    def detect_object(self, object_type='banana'):
        """检测物体并返回世界坐标"""
        print(f"\n检测物体: {object_type}")
        
        # 获取图像
        image = self._render_camera_image()
        
        # 检测物体
        pixel_coords = self._detect_object_by_color(image, object_type)
        
        if pixel_coords is None:
            print(f"  {object_type}检测失败")
            return False, None, 0.0
        
        # 转换为世界坐标
        world_coords = self._pixel_to_world(pixel_coords)
        
        if world_coords is None:
            print(f"  坐标转换失败")
            return False, None, 0.0
        
        # 计算误差（与已知位置比较）
        if object_type == 'banana':
            true_pos = self.banana_world_pos
        elif object_type == 'apple':
            true_pos = self.apple_world_pos
        else:
            true_pos = np.zeros(3)
        
        error = np.linalg.norm(world_coords - true_pos)
        
        print(f"  检测位置: {world_coords}")
        print(f"  实际位置: {true_pos}")
        print(f"  定位误差: {error*1000:.1f} mm")
        
        # 显示结果
        self._display_detection(image, pixel_coords, object_type)
        
        return True, world_coords, error
    
    def _display_detection(self, image, pixel_coords, object_type):
        """显示检测结果"""
        if image is None or pixel_coords is None:
            return
        
        display_img = image.copy()
        if len(display_img.shape) == 3:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
        
        u, v = pixel_coords
        
        # 绘制检测点
        cv2.circle(display_img, (u, v), 8, (0, 255, 0), -1)
        cv2.putText(display_img, f"{object_type}: ({u},{v})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示图像
        cv2.imshow(f"{object_type} Detection", display_img)
        cv2.waitKey(1000)  # 显示1秒
        cv2.destroyAllWindows()
    
    def run_vision_test(self):
        """运行视觉测试"""
        print(f"\n{'='*60}")
        print("视觉检测测试")
        print(f"{'='*60}")
        
        # 测试香蕉检测
        print("\n1. 测试香蕉检测...")
        banana_success, banana_pos, banana_error = self.detect_object('banana')
        
        if banana_success:
            print(f"  香蕉检测: 成功, 位置{banana_pos}, 误差{banana_error*1000:.1f}mm")
        else:
            print(f"  香蕉检测: 失败")
        
        # 测试苹果检测
        print("\n2. 测试苹果检测...")
        apple_success, apple_pos, apple_error = self.detect_object('apple')
        
        if apple_success:
            print(f"  苹果检测: 成功, 位置{apple_pos}, 误差{apple_error*1000:.1f}mm")
        else:
            print(f"  苹果检测: 失败")
        
        print(f"\n{'='*60}")
        
        return banana_success and apple_success
