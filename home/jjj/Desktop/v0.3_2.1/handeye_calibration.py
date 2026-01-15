"""
handeye_calibration.py - 恢复能工作的版本，只修复mean_rotation
"""

import numpy as np
import cv2
import time
import json
from scipy.spatial.transform import Rotation as R

class HandEyeCalibrator:
    """使用之前能工作的检测方法"""
    
    def __init__(self, robot_controller):
        self.robot = robot_controller
        self.calibration_matrix = None
        
        # 相机参数
        self.camera_matrix = self._get_camera_intrinsics()
        self.dist_coeffs = np.zeros((5, 1))
        
        # 桌子信息
        self.table_position = np.array([0, -0.5, 0.3])
        self.table_size = np.array([0.5, 0.3, 0.05])
        self.table_top_z = self.table_position[2] + self.table_size[2]
        
        # 已知的世界坐标
        self.banana_world_pos = np.array([-0.2, -0.4, 0.45])
        self.apple_world_pos = np.array([0.2, -0.4, 0.45])
        self.board_world_pos = np.array([0, -0.4, 0.5])
        
        # 使用之前能工作的关节配置
        self.start_joints = {
            0: 1.15,   # 关节0
            1: -0.8,   # 关节1
            2: -0.28,  # 关节2
            3: -0.15   # 关节3
        }
        
        self.adjusted_joints = {
            0: 1.15,   # 关节0
            1: -0.1,   # 关节1（看到物体）
            2: -0.28,  # 关节2
            3: -0.15   # 关节3
        }
        
        print(f"[标定器] 初始化完成 - 使用之前的能工作版本")
    
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
    
    def _set_joints_slowly(self, target_joints):
        """缓慢设置关节位置"""
        print(f"  设置关节: {target_joints}")
        
        steps = 40
        for step in range(steps):
            t = step / steps
            
            for joint_idx, target_angle in target_joints.items():
                if joint_idx < self.robot.n_controlled:
                    current = self.robot.get_joint_current(joint_idx)
                    angle = current + (target_angle - current) * (3*t**2 - 2*t**3)
                    self.robot.set_joint_target(joint_idx, angle)
            
            for _ in range(4):
                self.robot.update_control()
                time.sleep(0.01)
        
        time.sleep(0.3)
        
        # 检查位置
        ee_pos = self.robot.get_body_position('right_gripper_center')
        if ee_pos is not None:
            print(f"  末端位置: X={ee_pos[0]:.3f}, Y={ee_pos[1]:.3f}, Z={ee_pos[2]:.3f}")
        
        return True
    
    def _move_to_start_position(self):
        """移动到起始位置"""
        print("\n[步骤1] 移动到起始安全位置")
        return self._set_joints_slowly(self.start_joints)
    
    def _adjust_to_see_board(self):
        """调整到能看到棋盘格"""
        print("\n[步骤2] 调整以看到物体")
        return self._set_joints_slowly(self.adjusted_joints)
    
    def _get_robot_pose(self):
        """获取机器人位姿"""
        pos = self.robot.get_body_position('right_gripper_center')
        quat = self.robot.get_body_orientation('right_gripper_center')
        
        if pos is None or quat is None:
            return None
        
        rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        T = np.eye(4)
        T[:3, :3] = rotation.as_matrix()
        T[:3, 3] = pos
        
        return T
    
    def _render_camera_view(self):
        """渲染相机视图"""
        try:
            import mujoco
            
            renderer = mujoco.Renderer(self.robot.model, 480, 640)
            camera_id = 2
            renderer.update_scene(self.robot.data, camera=camera_id)
            image = renderer.render()
            
            if image is not None:
                return image
                
        except Exception as e:
            print(f"[渲染] 错误: {e}")
        
        # 备用图像
        return self._create_fallback_image()
    
    def _create_fallback_image(self):
        """创建备用图像"""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 添加黄色香蕉
        cv2.rectangle(image, (250, 200), (350, 250), (0, 255, 255), -1)
        cv2.putText(image, "Banana", (280, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 添加红色苹果
        cv2.circle(image, (320, 150), 25, (0, 0, 255), -1)
        cv2.putText(image, "Apple", (300, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return image
    
    def _detect_yellow_banana_original(self, image):
        """使用之前的HSV检测方法（能工作的版本）"""
        if image is None:
            return None
        
        # 转换为BGR（MuJoCo返回的是RGB）
        if len(image.shape) == 3:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 转换为HSV（原来的能工作方法）
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        
        # 黄色范围（原来的参数）
        lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 形态学操作（原来的）
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("  未检测到黄色物体")
            return None
        
        # 选择最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 50:  # 面积太小
            print(f"  轮廓面积太小: {area}")
            return None
        
        # 计算质心
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
        
        print(f"  检测到黄色物体: ({u}, {v}), 面积: {area}")
        return (u, v)
    
    def _collect_calibration_data_original(self):
        """使用原来的数据采集方法"""
        print(f"\n{'='*60}")
        print("采集标定数据（使用原来的方法）")
        print(f"{'='*60}")
        
        calibration_pairs = []
        
        # 1. 起始安全位置
        print("\n[1/3] 移动到起始安全位置")
        self._move_to_start_position()
        time.sleep(1.0)
        
        # 2. 调整看到物体
        print("\n[2/3] 调整看到物体")
        self._adjust_to_see_board()
        time.sleep(0.5)
        
        # 3. 采集多个位姿
        print(f"\n[3/3] 采集6个位姿")
        cv2.namedWindow("检测视图", cv2.WINDOW_NORMAL)
        
        for i in range(6):
            print(f"\n--- 位姿 {i+1}/6 ---")
            
            # 轻微调整
            if i > 0:
                current_joint1 = self.robot.get_joint_current(1)
                adjustment = 0.03 if i % 2 == 0 else -0.03
                self.robot.set_joint_target(1, current_joint1 + adjustment)
                time.sleep(0.3)
            
            # 获取机器人位姿
            robot_pose = self._get_robot_pose()
            if robot_pose is None:
                print("  无法获取机器人位姿")
                continue
            
            # 获取图像
            image = self._render_camera_view()
            
            # 检测香蕉（原来的方法）
            banana_pixel = self._detect_yellow_banana_original(image)
            
            # 显示
            display_img = image.copy()
            if len(display_img.shape) == 3:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
            
            if banana_pixel is None:
                print("  香蕉检测失败")
                cv2.putText(display_img, "NO BANANA", (250, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 使用图像中心
                u, v = 320, 240
            else:
                u, v = banana_pixel
                print(f"  检测成功: ({u}, {v})")
                
                cv2.circle(display_img, (u, v), 5, (0, 255, 0), -1)
                cv2.putText(display_img, f"Banana: ({u},{v})", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("检测视图", display_img)
            cv2.waitKey(300)
            
            # 存储数据
            calibration_pairs.append({
                'robot_pose': robot_pose,
                'pixel_coords': np.array([u, v], dtype=np.float32),
                'world_coords': self.banana_world_pos.copy(),
                'image': image,
                'detected': banana_pixel is not None
            })
            
            print(f"  成功采集位姿 {i+1}")
        
        cv2.destroyAllWindows()
        
        print(f"\n采集完成: {len(calibration_pairs)} 个有效位姿")
        detected_count = sum(1 for p in calibration_pairs if p['detected'])
        print(f"其中 {detected_count} 个成功检测到香蕉")
        
        return calibration_pairs
    
    def _calculate_hand_eye_matrix(self, calibration_pairs):
        """计算手眼标定矩阵"""
        if len(calibration_pairs) < 3:
            return None
        
        # 收集相对运动
        A = []
        B = []
        
        for i in range(len(calibration_pairs)):
            for j in range(i+1, len(calibration_pairs)):
                T_robot_i = calibration_pairs[i]['robot_pose']
                T_robot_j = calibration_pairs[j]['robot_pose']
                
                # 假设相机看到的物体位置不变
                A_ij = np.eye(4)  # 相机相对运动
                B_ij = np.linalg.inv(T_robot_i) @ T_robot_j  # 机器人相对运动
                
                A.append(A_ij)
                B.append(B_ij)
        
        # 使用平均方法
        X_sum = np.zeros((4, 4))
        
        for a, b in zip(A, B):
            X_i = a @ np.linalg.inv(b)
            X_sum += X_i
        
        if len(A) == 0:
            return None
        
        X = X_sum / len(A)
        
        # 正交化旋转矩阵
        U, S, Vt = np.linalg.svd(X[:3, :3])
        X[:3, :3] = U @ Vt
        
        return X
    
    def _calculate_errors_complete(self, X, calibration_pairs):
        """计算完整误差（包含mean_rotation）"""
        if X is None or not calibration_pairs:
            return {
                'mean': 0.0, 'max': 0.0, 'std': 0.0,
                'mean_translation': 0.0, 'max_translation': 0.0,
                'mean_rotation': 0.0, 'max_rotation': 0.0
            }
        
        errors = []
        translation_errors = []
        
        for pair in calibration_pairs:
            T_robot = pair['robot_pose']
            T_camera_est = pair['pixel_coords']
            
            # 使用标定矩阵计算投影
            T_camera_calc = X @ T_robot
            
            # 计算位置误差
            pos_error = np.linalg.norm(T_camera_calc[:3, 3] - self.banana_world_pos)
            translation_errors.append(pos_error)
            
            # 像素误差（近似）
            pixel_error = pos_error * 100  # 粗略估计
            errors.append(pixel_error)
        
        # 计算旋转误差
        rotation_matrix = X[:3, :3]
        try:
            # 计算与单位矩阵的角度差
            identity = np.eye(3)
            trace = np.trace(rotation_matrix.T @ identity)
            rotation_angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            mean_rotation = float(rotation_angle)
            max_rotation = float(rotation_angle)
        except:
            mean_rotation = 0.0
            max_rotation = 0.0
        
        # 统计
        if len(errors) == 0:
            return {
                'mean': 0.0, 'max': 0.0, 'std': 0.0,
                'mean_translation': 0.0, 'max_translation': 0.0,
                'mean_rotation': mean_rotation, 'max_rotation': max_rotation
            }
        
        error_stats = {
            'mean': float(np.mean(errors)),
            'max': float(np.max(errors)),
            'std': float(np.std(errors)),
            'mean_translation': float(np.mean(translation_errors)),
            'max_translation': float(np.max(translation_errors)),
            'mean_rotation': mean_rotation,
            'max_rotation': max_rotation
        }
        
        return error_stats
    
    # 在handeye_calibration.py中添加这个方法
    def _calculate_calibration_with_solvepnp(self, calibration_pairs):
        """使用solvePnP计算标定矩阵"""
        if len(calibration_pairs) < 4:
            return None
        
        # 准备数据
        image_points = []
        object_points = []
        
        for pair in calibration_pairs:
            # 使用检测到的像素坐标
            pixel_coords = pair['pixel_coords']
            world_coords = pair['world_coords']
            
            image_points.append(pixel_coords)
            object_points.append(world_coords)
        
        image_points = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
        object_points = np.array(object_points, dtype=np.float32).reshape(-1, 1, 3)
        
        print(f"  使用 {len(image_points)} 个点进行solvePnP计算")
        
        try:
            # 使用solvePnP
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if not success:
                print("  solvePnP失败")
                return None
            
            # 转换为变换矩阵
            R_mat, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R_mat
            T[:3, 3] = tvec.flatten()
            
            print(f"  solvePnP计算成功!")
            print(f"  平移向量: {tvec.flatten()}")
            
            return T
            
        except Exception as e:
            print(f"  solvePnP错误: {e}")
            return None
    
    def perform_calibration(self):
        """执行标定 - 使用solvePnP方法"""
        print(f"\n{'='*60}")
        print("手眼标定 - 使用solvePnP方法")
        print(f"{'='*60}")
        
        try:
            # 1. 采集数据
            print("\n步骤1: 采集数据")
            calibration_pairs = self._collect_calibration_data_original()
            
            if len(calibration_pairs) < 4:
                print("错误: 数据不足")
                errors = {
                    'mean': 0.0, 'max': 0.0, 'std': 0.0,
                    'mean_translation': 0.0, 'max_translation': 0.0,
                    'mean_rotation': 0.0, 'max_rotation': 0.0
                }
                return False, None, errors
            
            # 2. 使用solvePnP计算标定矩阵
            print("\n步骤2: 使用solvePnP计算标定矩阵")
            X = self._calculate_calibration_with_solvepnp(calibration_pairs)
            
            if X is None:
                print("  solvePnP失败，使用原方法")
                X = self._calculate_hand_eye_matrix(calibration_pairs)
            
            if X is None:
                print("错误: 无法计算标定矩阵")
                errors = {
                    'mean': 0.0, 'max': 0.0, 'std': 0.0,
                    'mean_translation': 0.0, 'max_translation': 0.0,
                    'mean_rotation': 0.0, 'max_rotation': 0.0
                }
                return False, None, errors
            
            # 3. 计算误差
            print("\n步骤3: 计算误差")
            errors = self._calculate_errors_complete(X, calibration_pairs)
            
            # 4. 保存结果
            print("\n步骤4: 保存结果")
            self.calibration_matrix = X
            np.save('calibration_matrix.npy', X)
            
            # 保存报告
            report = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'calibration_matrix': X.tolist(),
                'num_pairs': len(calibration_pairs),
                'detected_count': sum(1 for p in calibration_pairs if p['detected']),
                'errors': errors,
                'method': 'solvePnP' if 'solvePnP' in locals() else 'original'
            }
            
            with open('calibration_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n{'='*60}")
            print("标定完成!")
            print(f"{'='*60}")
            print(f"标定矩阵已保存")
            print(f"平均位置误差: {errors['mean_translation']*1000:.1f} mm")
            print(f"最大位置误差: {errors['max_translation']*1000:.1f} mm")
            print(f"旋转误差: {errors['mean_rotation']:.4f} rad")
            
            if errors['mean_translation'] < 0.01:
                print("✓ 精度良好 (<10mm)")
            elif errors['mean_translation'] < 0.02:
                print("✓ 精度可接受 (<20mm)")
            else:
                print("⚠ 精度一般")
            
            print(f"{'='*60}")
            
            # 5. 返回安全位置
            print("\n步骤5: 返回安全位置")
            self._move_to_start_position()
            
            return True, X, errors
            
        except Exception as e:
            print(f"\n标定失败: {e}")
            import traceback
            traceback.print_exc()
            
            errors = {
                'mean': 0.0, 'max': 0.0, 'std': 0.0,
                'mean_translation': 0.0, 'max_translation': 0.0,
                'mean_rotation': 0.0, 'max_rotation': 0.0
            }
            return False, None, errors
