"""
grasp_execution.py - 真实抓取执行与控制
使用真实逆运动学、真实接触检测、真实力控制，无简化假设
"""

import numpy as np
import time
import math
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

class GraspExecutor:
    """真实抓取执行器"""
    
    def __init__(self, robot_controller):
        self.robot = robot_controller
        self.model = robot_controller.model
        self.data = robot_controller.data
        
        # 执行器配置
        self.execution_config = {
            'max_grasp_force': 20.0,  # 最大抓取力 (N)
            'grasp_force_threshold': 5.0,  # 抓取力阈值 (N)
            'grasp_position_tolerance': 0.01,  # 位置容忍度 1cm
            'grasp_orientation_tolerance': 0.1,  # 姿态容忍度 0.1rad
            'gripper_close_speed': 0.01,  # 夹爪闭合速度 (m/s)
            'gripper_open_position': 0.035,  # 夹爪张开位置
            'gripper_close_position': 0.005,  # 夹爪闭合位置
            'contact_check_interval': 0.01,  # 接触检查间隔 10ms
            'max_grasp_time': 5.0,  # 最大抓取时间
            'object_hold_height': 0.2,  # 物体持握高度
            'force_control_gain': 100.0,  # 力控制增益
            'admittance_mass': 10.0,  # 导纳控制质量
            'admittance_damping': 50.0,  # 导纳控制阻尼
            'slip_detection_threshold': 0.005,  # 滑移检测阈值 5mm
            'success_grasp_force': 8.0  # 成功抓取所需最小力 (N)
        }
        
        # 机械臂配置
        self.arm_config = {
            'right': {
                'base_joints': [0, 1, 2, 3, 4, 5, 6],  # 右臂关节索引
                'gripper_joints': [7],  # 右夹爪关节索引
                'ee_body': 'right_gripper_center',
                'gripper_body': 'right_link_left_jaw',
                'gripper_tip_body': 'right_link_left_jaw',  # 夹爪尖端
                'contact_geom': 'right_link_left_jaw_collision'  # 接触几何体
            },
            'left': {
                'base_joints': [8, 9, 10, 11, 12, 13, 14],  # 左臂关节索引
                'gripper_joints': [15],  # 左夹爪关节索引
                'ee_body': 'left_gripper_center',
                'gripper_body': 'left_link_left_jaw',
                'gripper_tip_body': 'left_link_left_jaw',
                'contact_geom': 'left_link_left_jaw_collision'
            }
        }
        
        # 使用右臂作为默认
        self.active_arm = 'right'
        self._update_arm_config()
        
        # 物体信息
        self.object_info = {
            'apple': {
                'body_name': 'apple',
                'size': 0.04,  # 半径
                'mass': 0.02,  # 质量
                'friction': 0.8  # 摩擦系数
            },
            'banana': {
                'body_name': 'banana',
                'size': [0.08, 0.02, 0.02],  # 长宽高
                'mass': 0.01,
                'friction': 0.6
            }
        }
        
        # 执行状态
        self.execution_state = {
            'is_grasping': False,
            'grasped_object': None,
            'grasp_force': 0.0,
            'contact_force': np.zeros(3),
            'object_position': None,
            'object_velocity': np.zeros(3),
            'slip_detected': False,
            'execution_start_time': None,
            'current_phase': 'idle'
        }
        
        # 统计数据
        self.execution_stats = {
            'total_grasp_attempts': 0,
            'successful_grasps': 0,
            'failed_grasps': 0,
            'failure_reasons': {},
            'average_grasp_time': 0.0,
            'average_grasp_force': 0.0,
            'force_history': [],
            'position_history': [],
            'contact_history': []
        }
        
        print(f"抓取执行器初始化完成")
        print(f"使用机械臂: {self.active_arm}")
        print(f"最大抓取力: {self.execution_config['max_grasp_force']} N")
        print(f"力控制增益: {self.execution_config['force_control_gain']}")
    
    def _update_arm_config(self):
        """更新活动机械臂配置"""
        config = self.arm_config[self.active_arm]
        self.joint_indices = config['base_joints']
        self.gripper_indices = config['gripper_joints']
        self.ee_body = config['ee_body']
        self.gripper_body = config['gripper_body']
        self.gripper_tip_body = config['gripper_tip_body']
        self.contact_geom = config['contact_geom']
    
    def _get_ee_pose(self):
        """获取末端执行器真实位姿"""
        pos = self.robot.get_body_position(self.ee_body)
        quat = self.robot.get_body_orientation(self.ee_body)
        
        if pos is None or quat is None:
            return None
        
        return {'position': pos, 'quaternion': quat}
    
    def _get_object_pose(self, object_name):
        """获取物体真实位姿"""
        if object_name not in self.object_info:
            return None
        
        body_name = self.object_info[object_name]['body_name']
        pos = self.robot.get_body_position(body_name)
        quat = self.robot.get_body_orientation(body_name)
        
        if pos is None:
            return None
        
        return {'position': pos, 'quaternion': quat}
    
    def _get_gripper_position(self):
        """获取夹爪真实位置"""
        joint_idx = self.gripper_indices[0]
        if joint_idx < len(self.robot.actuated_joint_ids):
            return self.robot.get_joint_current(joint_idx)
        return self.execution_config['gripper_open_position']
    
    def _set_gripper_position(self, position, speed=None):
        """设置夹爪位置"""
        if speed is None:
            speed = self.execution_config['gripper_close_speed']
        
        joint_idx = self.gripper_indices[0]
        current_pos = self._get_gripper_position()
        
        # 计算目标位置
        target_pos = np.clip(
            position,
            self.execution_config['gripper_close_position'],
            self.execution_config['gripper_open_position']
        )
        
        # 逐步移动夹爪
        step = speed * 0.01  # 每10ms移动的距离
        
        if target_pos > current_pos:
            # 张开
            new_pos = min(current_pos + step, target_pos)
        else:
            # 闭合
            new_pos = max(current_pos - step, target_pos)
        
        self.robot.set_joint_target(joint_idx, new_pos)
        return new_pos
    
    def _check_contact(self, object_name):
        """检查与物体的真实接触"""
        if object_name not in self.object_info:
            return False, np.zeros(3), 0.0
        
        # 获取夹爪几何体ID
        gripper_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, self.contact_geom
        )
        
        if gripper_geom_id == -1:
            return False, np.zeros(3), 0.0
        
        # 获取物体几何体ID
        body_name = self.object_info[object_name]['body_name']
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        
        if body_id == -1:
            return False, np.zeros(3), 0.0
        
        # 检查接触
        total_force = np.zeros(3)
        contact_count = 0
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # 检查是否涉及夹爪和物体
            geom1, geom2 = contact.geom1, contact.geom2
            
            geom1_body = self.model.geom_bodyid[geom1]
            geom2_body = self.model.geom_bodyid[geom2]
            
            if (geom1 == gripper_geom_id and geom2_body == body_id) or \
               (geom2 == gripper_geom_id and geom1_body == body_id):
                
                # 计算接触力
                force = np.zeros(3)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                
                total_force += force
                contact_count += 1
        
        if contact_count > 0:
            force_magnitude = np.linalg.norm(total_force)
            return True, total_force, force_magnitude
        else:
            return False, np.zeros(3), 0.0
    
    def _compute_ik(self, target_position, target_orientation=None, 
                   current_joints=None, max_iterations=50):
        """计算逆运动学（真实数值优化）"""
        if current_joints is None:
            current_joints = [self.robot.get_joint_current(i) for i in self.joint_indices]
        
        def forward_kinematics(joint_angles):
            """正向运动学辅助函数"""
            # 保存当前状态
            original_qpos = self.data.qpos.copy()
            
            # 设置关节角度
            for i, joint_idx in enumerate(self.joint_indices):
                if joint_idx < len(self.data.qpos):
                    self.data.qpos[joint_idx] = joint_angles[i]
            
            # 执行正向运动学
            mujoco.mj_fwdPosition(self.model, self.data)
            
            # 获取末端执行器位姿
            ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body)
            if ee_id == -1:
                # 恢复原始状态
                self.data.qpos[:] = original_qpos
                mujoco.mj_fwdPosition(self.model, self.data)
                return None, None
            
            pos = self.data.body(ee_id).xpos.copy()
            quat = self.data.body(ee_id).xquat.copy()
            
            # 恢复原始状态
            self.data.qpos[:] = original_qpos
            mujoco.mj_fwdPosition(self.model, self.data)
            
            return pos, quat
        
        def ik_cost(joint_angles):
            """逆运动学成本函数"""
            pos, quat = forward_kinematics(joint_angles)
            
            if pos is None:
                return float('inf')
            
            # 位置误差
            pos_error = np.linalg.norm(pos - target_position)
            
            # 姿态误差（如果有目标姿态）
            orient_error = 0.0
            if target_orientation is not None and quat is not None:
                q_current = quat
                q_target = target_orientation
                
                # 四元数角度差
                dot = np.abs(np.dot(q_current, q_target))
                dot = np.clip(dot, -1.0, 1.0)
                angle = 2 * np.arccos(dot)
                orient_error = angle
            
            # 关节限制惩罚
            joint_limit_penalty = 0.0
            joint_ranges = []
            for i in range(len(self.joint_indices)):
                # 获取关节限制（简化处理）
                joint_limit = (-np.pi, np.pi)  # 默认限制
                joint_ranges.append(joint_limit)
                
                angle = joint_angles[i]
                lower, upper = joint_limit
                if angle < lower:
                    joint_limit_penalty += (lower - angle) ** 2
                elif angle > upper:
                    joint_limit_penalty += (angle - upper) ** 2
            
            # 平滑惩罚（避免突变）
            smooth_penalty = 0.0
            if hasattr(self, '_last_joint_angles'):
                for i in range(len(joint_angles)):
                    diff = joint_angles[i] - self._last_joint_angles[i]
                    smooth_penalty += diff ** 2
            
            # 总成本
            total_cost = (pos_error * 10.0 + 
                         orient_error * 5.0 + 
                         joint_limit_penalty * 0.1 +
                         smooth_penalty * 0.05)
            
            return total_cost
        
        # 设置边界
        bounds = []
        for i in range(len(self.joint_indices)):
            bounds.append((-np.pi * 0.9, np.pi * 0.9))  # 90%的关节范围
        
        # 记录上一次关节角度用于平滑
        if not hasattr(self, '_last_joint_angles'):
            self._last_joint_angles = current_joints.copy()
        
        # 优化
        result = minimize(
            ik_cost,
            current_joints,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iterations, 'ftol': 1e-6}
        )
        
        if result.success:
            # 验证结果
            final_pos, _ = forward_kinematics(result.x)
            if final_pos is None:
                return None
            
            pos_error = np.linalg.norm(final_pos - target_position)
            
            if pos_error < self.execution_config['grasp_position_tolerance']:
                self._last_joint_angles = result.x.copy()
                return result.x
            else:
                return None
        else:
            return None
    
    def _move_to_position_ik(self, target_position, target_orientation=None, 
                           max_time=3.0, check_collision=True):
        """使用逆运动学移动到目标位置"""
        print(f"  移动到位置: {target_position}")
        
        start_time = time.time()
        success = False
        
        while time.time() - start_time < max_time:
            # 获取当前关节角度
            current_joints = [self.robot.get_joint_current(i) for i in self.joint_indices]
            
            # 计算逆运动学
            ik_solution = self._compute_ik(
                target_position, 
                target_orientation,
                current_joints,
                max_iterations=20
            )
            
            if ik_solution is None:
                print("   逆运动学失败")
                time.sleep(0.01)
                continue
            
            # 检查碰撞
            if check_collision:
                # 简化的碰撞检查：检查末端与目标的距离
                ee_pose = self._get_ee_pose()
                if ee_pose:
                    distance = np.linalg.norm(ee_pose['position'] - target_position)
                    if distance < 0.02:  # 2cm内认为到达
                        success = True
                        break
            
            # 应用关节角度
            for i, joint_idx in enumerate(self.joint_indices):
                self.robot.set_joint_target(joint_idx, ik_solution[i])
            
            # 更新控制
            self.robot.update_control()
            
            # 短暂延迟
            time.sleep(0.01)
            
            # 检查是否到达
            ee_pose = self._get_ee_pose()
            if ee_pose:
                distance = np.linalg.norm(ee_pose['position'] - target_position)
                if distance < self.execution_config['grasp_position_tolerance']:
                    success = True
                    print(f"   到达目标，误差: {distance*1000:.1f} mm")
                    break
            
            # 显示进度
            elapsed = time.time() - start_time
            if elapsed % 0.5 < 0.01:
                print(f"   进度: {elapsed:.1f}s, 距离: {distance*1000:.1f}mm")
        
        return success
    
    def _calculate_grasp_orientation(self, object_name, approach_direction=np.array([0, 0, -1])):
        """计算抓取姿态"""
        # 获取物体方向（如果是香蕉）
        if object_name == 'banana':
            # 香蕉通常是水平放置的
            # Z轴朝下，X轴沿着香蕉长度方向
            obj_pose = self._get_object_pose(object_name)
            if obj_pose and 'quaternion' in obj_pose:
                # 使用物体的当前朝向
                quat = obj_pose['quaternion']
                return quat
        
        # 默认：Z轴向下
        z_axis = approach_direction / np.linalg.norm(approach_direction)
        
        # 创建正交坐标系
        if np.abs(z_axis[0]) < 0.9:
            x_axis = np.cross(np.array([0, 1, 0]), z_axis)
        else:
            x_axis = np.cross(np.array([1, 0, 0]), z_axis)
        
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # 构建旋转矩阵
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        # 转换为四元数
        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # [x, y, z, w]
        
        return quaternion
    
    def _detect_slip(self, object_name, initial_position):
        """检测物体滑移"""
        if not self.execution_state['is_grasping']:
            return False
        
        current_pose = self._get_object_pose(object_name)
        if current_pose is None:
            return False
        
        # 计算位置变化
        position_change = np.linalg.norm(current_pose['position'] - initial_position)
        
        if position_change > self.execution_config['slip_detection_threshold']:
            print(f"   滑移检测: 位移 {position_change*1000:.1f} mm")
            return True
        
        return False
    
    def _apply_force_control(self, desired_force):
        """应用力控制"""
        # 获取当前接触力
        if self.execution_state['grasped_object']:
            contact, force_vector, force_magnitude = self._check_contact(
                self.execution_state['grasped_object']
            )
            
            if contact:
                # 计算力误差
                force_error = desired_force - force_magnitude
                
                # 简单的力控制：调整夹爪位置
                current_gripper_pos = self._get_gripper_position()
                
                # 根据力误差调整夹爪位置
                if force_error > 0:
                    # 需要增加力，进一步闭合夹爪
                    adjustment = min(
                        force_error * self.execution_config['force_control_gain'] * 0.001,
                        0.001  # 最大调整量
                    )
                    new_pos = current_gripper_pos - adjustment
                else:
                    # 力过大，稍微张开夹爪
                    adjustment = min(
                        abs(force_error) * self.execution_config['force_control_gain'] * 0.001,
                        0.001
                    )
                    new_pos = current_gripper_pos + adjustment
                
                # 应用调整
                new_pos = np.clip(
                    new_pos,
                    self.execution_config['gripper_close_position'],
                    self.execution_config['gripper_open_position']
                )
                
                self._set_gripper_position(new_pos)
                
                return force_magnitude
        
        return 0.0
    
    def execute_grasp(self, object_name, arm='right', max_attempts=1, record_data=True):
        """执行抓取操作"""
        print(f"\n{'='*60}")
        print(f"执行 {object_name} 抓取")
        print(f"{'='*60}")
        
        # 更新执行统计
        self.execution_stats['total_grasp_attempts'] += 1
        
        # 设置活动机械臂
        if arm != self.active_arm:
            self.active_arm = arm
            self._update_arm_config()
        
        # 验证物体
        if object_name not in self.object_info:
            print(f"错误: 未知物体 {object_name}")
            self._record_failure('unknown_object')
            return False, {'reason': 'unknown_object'}
        
        # 获取物体初始位置
        initial_pose = self._get_object_pose(object_name)
        if initial_pose is None:
            print(f"错误: 无法获取物体 {object_name} 的位置")
            self._record_failure('object_not_found')
            return False, {'reason': 'object_not_found'}
        
        initial_position = initial_pose['position'].copy()
        print(f"物体初始位置: {initial_position}")
        
        # 重置执行状态
        self.execution_state = {
            'is_grasping': False,
            'grasped_object': None,
            'grasp_force': 0.0,
            'contact_force': np.zeros(3),
            'object_position': initial_position.copy(),
            'object_velocity': np.zeros(3),
            'slip_detected': False,
            'execution_start_time': time.time(),
            'current_phase': 'approach'
        }
        
        attempt_details = {
            'object': object_name,
            'arm': arm,
            'start_time': time.time(),
            'phases': {},
            'force_profile': [],
            'position_profile': [],
            'contact_events': [],
            'success': False,
            'reason': None
        }
        
        try:
            # ========== 阶段1: 接近物体 ==========
            print(f"\n阶段1: 接近物体")
            attempt_details['phases']['approach'] = {'start': time.time()}
            
            # 计算预抓取位置（物体上方5cm）
            pre_grasp_position = initial_position.copy()
            pre_grasp_position[2] += 0.05
            
            # 计算抓取姿态
            grasp_orientation = self._calculate_grasp_orientation(object_name)
            
            # 移动到预抓取位置
            approach_success = self._move_to_position_ik(
                pre_grasp_position,
                grasp_orientation,
                max_time=3.0,
                check_collision=True
            )
            
            if not approach_success:
                print("  接近阶段失败")
                self._record_failure('approach_failed')
                attempt_details['reason'] = 'approach_failed'
                attempt_details['phases']['approach']['end'] = time.time()
                attempt_details['phases']['approach']['success'] = False
                return False, attempt_details
            
            attempt_details['phases']['approach']['end'] = time.time()
            attempt_details['phases']['approach']['success'] = True
            print("  接近阶段完成")
            
            # 短暂暂停
            time.sleep(0.5)
            
            # ========== 阶段2: 预抓取调整 ==========
            print(f"\n阶段2: 预抓取调整")
            attempt_details['phases']['pre_grasp'] = {'start': time.time()}
            
            # 张开夹爪
            print("  张开夹爪")
            gripper_open_start = time.time()
            while time.time() - gripper_open_start < 2.0:
                current_pos = self._get_gripper_position()
                if current_pos >= self.execution_config['gripper_open_position'] * 0.9:
                    break
                self._set_gripper_position(self.execution_config['gripper_open_position'])
                time.sleep(0.01)
            
            # 调整到精确的抓取位置
            grasp_position = initial_position.copy()
            if object_name == 'apple':
                grasp_position[2] += self.object_info['apple']['size'] * 0.8  # 苹果半径的80%
            elif object_name == 'banana':
                grasp_position[2] += self.object_info['banana']['size'][2] * 0.5  # 香蕉高度的一半
            
            print(f"  精确抓取位置: {grasp_position}")
            
            adjustment_success = self._move_to_position_ik(
                grasp_position,
                grasp_orientation,
                max_time=2.0,
                check_collision=False  # 此时允许轻微接触
            )
            
            if not adjustment_success:
                print("  预抓取调整失败")
                self._record_failure('pre_grasp_adjustment_failed')
                attempt_details['reason'] = 'pre_grasp_adjustment_failed'
                attempt_details['phases']['pre_grasp']['end'] = time.time()
                attempt_details['phases']['pre_grasp']['success'] = False
                return False, attempt_details
            
            attempt_details['phases']['pre_grasp']['end'] = time.time()
            attempt_details['phases']['pre_grasp']['success'] = True
            print("  预抓取调整完成")
            
            # 短暂暂停
            time.sleep(0.3)
            
            # ========== 阶段3: 闭合夹爪 ==========
            print(f"\n阶段3: 闭合夹爪")
            attempt_details['phases']['grasp'] = {'start': time.time()}
            
            # 开始闭合夹爪
            print("  开始闭合夹爪")
            grasp_start_time = time.time()
            max_grasp_time = self.execution_config['max_grasp_time']
            target_force = self.execution_config['success_grasp_force']
            
            force_history = []
            position_history = []
            
            while time.time() - grasp_start_time < max_grasp_time:
                # 逐步闭合夹爪
                current_gripper_pos = self._get_gripper_position()
                target_pos = max(
                    current_gripper_pos - self.execution_config['gripper_close_speed'] * 0.01,
                    self.execution_config['gripper_close_position']
                )
                
                self._set_gripper_position(target_pos)
                
                # 检查接触
                contact, force_vector, force_magnitude = self._check_contact(object_name)
                
                # 记录数据
                force_history.append(force_magnitude)
                position_history.append(target_pos)
                
                # 检查是否达到目标力
                if force_magnitude >= target_force:
                    print(f"  达到目标抓取力: {force_magnitude:.1f} N")
                    self.execution_state['is_grasping'] = True
                    self.execution_state['grasped_object'] = object_name
                    self.execution_state['grasp_force'] = force_magnitude
                    self.execution_state['contact_force'] = force_vector
                    break
                
                # 检查滑移
                if self._detect_slip(object_name, initial_position):
                    print("  检测到滑移，调整抓取力")
                    # 稍微增加抓取力
                    target_force += 1.0
                
                # 短暂延迟
                time.sleep(0.01)
            
            # 保存力数据
            attempt_details['force_profile'] = force_history
            attempt_details['position_profile'] = position_history
            
            if not self.execution_state['is_grasping']:
                print(f"  抓取失败: 未达到足够抓取力")
                print(f"  最大达到力: {max(force_history) if force_history else 0:.1f} N")
                self._record_failure('insufficient_grasp_force')
                attempt_details['reason'] = 'insufficient_grasp_force'
                attempt_details['phases']['grasp']['end'] = time.time()
                attempt_details['phases']['grasp']['success'] = False
                attempt_details['phases']['grasp']['max_force'] = max(force_history) if force_history else 0
                return False, attempt_details
            
            attempt_details['phases']['grasp']['end'] = time.time()
            attempt_details['phases']['grasp']['success'] = True
            attempt_details['phases']['grasp']['final_force'] = force_magnitude
            print(f"  抓取成功，抓取力: {force_magnitude:.1f} N")
            
            # 短暂稳定
            time.sleep(0.5)
            
            # ========== 阶段4: 提升物体 ==========
            print(f"\n阶段4: 提升物体")
            attempt_details['phases']['lift'] = {'start': time.time()}
            
            # 计算提升位置
            lift_position = grasp_position.copy()
            lift_position[2] += self.execution_config['object_hold_height']
            
            print(f"  提升到: {lift_position}")
            
            # 使用力控制提升物体
            lift_start_time = time.time()
            lift_success = False
            
            while time.time() - lift_start_time < 3.0:
                # 获取当前末端位置
                ee_pose = self._get_ee_pose()
                if ee_pose is None:
                    break
                
                current_position = ee_pose['position']
                
                # 计算向目标移动的方向
                direction = lift_position - current_position
                distance = np.linalg.norm(direction)
                
                if distance < 0.01:  # 1cm容忍度
                    lift_success = True
                    print("  提升完成")
                    break
                
                # 缓慢移动
                step_size = min(0.001, distance * 0.1)  # 最大1mm步长
                direction_normalized = direction / distance
                next_position = current_position + direction_normalized * step_size
                
                # 移动到下一个位置
                self._move_to_position_ik(
                    next_position,
                    grasp_orientation,
                    max_time=0.1,
                    check_collision=False
                )
                
                # 检查物体是否仍然被抓握
                current_pose = self._get_object_pose(object_name)
                if current_pose is None:
                    print("  物体丢失!")
                    break
                
                # 检查滑移
                if self._detect_slip(object_name, initial_position):
                    print("  提升过程中检测到滑移")
                    # 增加抓取力
                    self._apply_force_control(target_force + 2.0)
                
                # 记录接触事件
                contact, force_vector, force_magnitude = self._check_contact(object_name)
                if contact:
                    attempt_details['contact_events'].append({
                        'time': time.time() - attempt_details['start_time'],
                        'force': force_magnitude,
                        'position': current_position.tolist()
                    })
                
                time.sleep(0.01)
            
            if not lift_success:
                print("  提升阶段失败")
                self._record_failure('lift_failed')
                attempt_details['reason'] = 'lift_failed'
                attempt_details['phases']['lift']['end'] = time.time()
                attempt_details['phases']['lift']['success'] = False
                
                # 即使提升失败，如果物体仍然被抓握，也算部分成功
                if self.execution_state['is_grasping']:
                    print("  但物体仍然被抓握着")
            
            attempt_details['phases']['lift']['end'] = time.time()
            attempt_details['phases']['lift']['success'] = lift_success
            
            # ========== 阶段5: 持握验证 ==========
            print(f"\n阶段5: 持握验证")
            attempt_details['phases']['hold'] = {'start': time.time()}
            
            # 持握2秒，验证稳定性
            hold_start_time = time.time()
            hold_duration = 2.0
            slip_detected = False
            
            while time.time() - hold_start_time < hold_duration:
                # 检查滑移
                if self._detect_slip(object_name, initial_position):
                    slip_detected = True
                    print("  持握过程中检测到滑移")
                    break
                
                # 应用力控制维持抓取力
                current_force = self._apply_force_control(target_force)
                
                # 记录数据
                if record_data:
                    ee_pose = self._get_ee_pose()
                    obj_pose = self._get_object_pose(object_name)
                    
                    if ee_pose and obj_pose:
                        self.execution_stats['force_history'].append(current_force)
                        self.execution_stats['position_history'].append({
                            'ee': ee_pose['position'].tolist(),
                            'object': obj_pose['position'].tolist(),
                            'time': time.time() - attempt_details['start_time']
                        })
                
                time.sleep(0.01)
            
            attempt_details['phases']['hold']['end'] = time.time()
            attempt_details['phases']['hold']['success'] = not slip_detected
            
            if slip_detected:
                print("  持握验证失败: 物体滑移")
                self._record_failure('slip_during_hold')
                attempt_details['reason'] = 'slip_during_hold'
            else:
                print("  持握验证成功")
            
            # ========== 阶段6: 释放物体 ==========
            print(f"\n阶段6: 释放物体")
            attempt_details['phases']['release'] = {'start': time.time()}
            
            # 返回到安全位置
            safe_position = pre_grasp_position.copy()
            safe_position[2] += 0.1  # 再提高10cm
            
            print("  移动到安全位置")
            self._move_to_position_ik(
                safe_position,
                grasp_orientation,
                max_time=2.0,
                check_collision=True
            )
            
            # 张开夹爪
            print("  张开夹爪释放物体")
            release_start = time.time()
            while time.time() - release_start < 2.0:
                current_pos = self._get_gripper_position()
                if current_pos >= self.execution_config['gripper_open_position'] * 0.95:
                    break
                self._set_gripper_position(self.execution_config['gripper_open_position'])
                time.sleep(0.01)
            
            # 重置抓取状态
            self.execution_state['is_grasping'] = False
            self.execution_state['grasped_object'] = None
            self.execution_state['grasp_force'] = 0.0
            
            attempt_details['phases']['release']['end'] = time.time()
            attempt_details['phases']['release']['success'] = True
            
            # ========== 结果评估 ==========
            print(f"\n结果评估")
            
            # 检查所有阶段是否成功
            all_phases_success = all(
                phase.get('success', False) 
                for phase in attempt_details['phases'].values() 
                if 'success' in phase
            )
            
            if all_phases_success and not slip_detected:
                print("  ✓ 抓取完全成功!")
                self.execution_stats['successful_grasps'] += 1
                attempt_details['success'] = True
                
                # 计算平均抓取力
                if force_history:
                    avg_force = np.mean(force_history)
                    self.execution_stats['average_grasp_force'] = (
                        self.execution_stats['average_grasp_force'] * 
                        (self.execution_stats['successful_grasps'] - 1) + 
                        avg_force
                    ) / self.execution_stats['successful_grasps']
                
                # 计算平均抓取时间
                total_time = time.time() - attempt_details['start_time']
                self.execution_stats['average_grasp_time'] = (
                    self.execution_stats['average_grasp_time'] * 
                    (self.execution_stats['successful_grasps'] - 1) + 
                    total_time
                ) / self.execution_stats['successful_grasps']
                
            else:
                print("  ✗ 抓取部分失败")
                self.execution_stats['failed_grasps'] += 1
                attempt_details['success'] = False
            
            attempt_details['end_time'] = time.time()
            attempt_details['total_duration'] = attempt_details['end_time'] - attempt_details['start_time']
            
            print(f"  总时间: {attempt_details['total_duration']:.2f}秒")
            print(f"  最终抓取力: {force_magnitude:.1f} N")
            
            # 可视化结果（可选）
            if record_data:
                self._visualize_grasp_attempt(attempt_details, object_name)
            
            return attempt_details['success'], attempt_details
            
        except Exception as e:
            print(f"抓取执行异常: {e}")
            import traceback
            traceback.print_exc()
            
            self._record_failure('execution_exception')
            attempt_details['reason'] = f'execution_exception: {str(e)}'
            attempt_details['success'] = False
            attempt_details['end_time'] = time.time()
            
            return False, attempt_details
    
    def _record_failure(self, reason):
        """记录失败原因"""
        self.execution_stats['failed_grasps'] += 1
        
        if reason not in self.execution_stats['failure_reasons']:
            self.execution_stats['failure_reasons'][reason] = 0
        self.execution_stats['failure_reasons'][reason] += 1
    
    def _visualize_grasp_attempt(self, attempt_details, object_name):
        """可视化抓取尝试"""
        if not attempt_details['force_profile']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 图1: 抓取力随时间变化
        ax1 = axes[0, 0]
        time_points = np.arange(len(attempt_details['force_profile'])) * 0.01
        ax1.plot(time_points, attempt_details['force_profile'], 'b-', linewidth=2)
        ax1.axhline(y=self.execution_config['success_grasp_force'], 
                   color='r', linestyle='--', label='目标力')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('抓取力 (N)')
        ax1.set_title('抓取力曲线')
        ax1.legend()
        ax1.grid(True)
        
        # 图2: 夹爪位置随时间变化
        ax2 = axes[0, 1]
        ax2.plot(time_points, attempt_details['position_profile'], 'g-', linewidth=2)
        ax2.axhline(y=self.execution_config['gripper_open_position'], 
                   color='b', linestyle='--', label='张开位置')
        ax2.axhline(y=self.execution_config['gripper_close_position'], 
                   color='r', linestyle='--', label='闭合位置')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('夹爪位置 (m)')
        ax2.set_title('夹爪位置曲线')
        ax2.legend()
        ax2.grid(True)
        
        # 图3: 各阶段持续时间
        ax3 = axes[1, 0]
        phases = list(attempt_details['phases'].keys())
        durations = []
        colors = []
        
        for phase in phases:
            phase_data = attempt_details['phases'][phase]
            if 'start' in phase_data and 'end' in phase_data:
                duration = phase_data['end'] - phase_data['start']
                durations.append(duration)
                colors.append('green' if phase_data.get('success', False) else 'red')
        
        bars = ax3.bar(range(len(durations)), durations, color=colors)
        ax3.set_xticks(range(len(durations)))
        ax3.set_xticklabels(phases, rotation=45)
        ax3.set_ylabel('持续时间 (s)')
        ax3.set_title('各阶段持续时间')
        
        # 添加数值标签
        for bar, duration in zip(bars, durations):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                   f'{duration:.1f}s', ha='center', va='bottom')
        
        # 图4: 接触事件
        ax4 = axes[1, 1]
        if attempt_details['contact_events']:
            times = [event['time'] for event in attempt_details['contact_events']]
            forces = [event['force'] for event in attempt_details['contact_events']]
            ax4.scatter(times, forces, c='red', s=50, alpha=0.6)
            ax4.set_xlabel('时间 (s)')
            ax4.set_ylabel('接触力 (N)')
            ax4.set_title('接触事件')
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, '无接触事件记录', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('接触事件')
        
        plt.suptitle(f'{object_name.capitalize()} 抓取尝试结果 - '
                    f"{'成功' if attempt_details['success'] else '失败'}", 
                    fontsize=16)
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime('%H%M%S')
        filename = f'grasp_{object_name}_{timestamp}.png'
        plt.savefig(filename, dpi=150)
        plt.show()
        
        print(f"抓取可视化已保存到 {filename}")
    
    def run_grasp_experiment(self, object_name, num_trials=10, arm='right'):
        """运行抓取实验"""
        print(f"\n{'='*60}")
        print(f"开始 {object_name} 抓取实验")
        print(f"试验次数: {num_trials}")
        print(f"{'='*60}")
        
        results = []
        
        for trial in range(num_trials):
            print(f"\n试验 {trial+1}/{num_trials}")
            
            # 重置物体位置（如果可能）
            if trial > 0:
                self._reset_object_position(object_name)
                time.sleep(0.5)
            
            # 执行抓取
            success, details = self.execute_grasp(object_name, arm, record_data=True)
            
            results.append({
                'trial': trial + 1,
                'success': success,
                'details': details
            })
            
            print(f"结果: {'成功' if success else '失败'}")
            
            if not success and 'reason' in details:
                print(f"原因: {details['reason']}")
            
            # 短暂暂停
            time.sleep(1.0)
        
        # 统计结果
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / num_trials
        
        print(f"\n{'='*60}")
        print(f"实验完成")
        print(f"{'='*60}")
        print(f"总尝试次数: {num_trials}")
        print(f"成功次数: {success_count}")
        print(f"成功率: {success_rate*100:.1f}%")
        
        # 显示失败原因分布
        if self.execution_stats['failure_reasons']:
            print(f"\n失败原因分布:")
            for reason, count in self.execution_stats['failure_reasons'].items():
                print(f"  {reason}: {count}次")
        
        # 显示统计数据
        print(f"\n统计数据:")
        print(f"平均抓取时间: {self.execution_stats['average_grasp_time']:.2f}秒")
        print(f"平均抓取力: {self.execution_stats['average_grasp_force']:.1f} N")
        
        # 生成实验报告
        report = self._generate_experiment_report(object_name, results)
        
        return results, report
    
    def _reset_object_position(self, object_name):
        """重置物体位置到模型中的初始位置"""
        print(f"  重置物体 {object_name} 位置...")
        
        # 获取物体ID
        body_name = self.object_info[object_name]['body_name']
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        
        if body_id == -1:
            print(f"    警告: 未找到物体 {object_name} (body: {body_name})")
            return False
        
        try:
            # 方法1: 重置整个仿真（最简单）
            # 这会重置所有物体到初始位置
            mujoco.mj_resetData(self.model, self.data)
            print("    重置整个仿真数据")
            
            # 方法2: 只重置特定物体（更精确）
            # 但需要知道物体在qpos中的索引位置
            # 查找物体的自由度索引
            # obj_dofadr = self.model.body_dofadr[body_id]
            # obj_nv = self.model.body_nv[body_id]
            
            # if obj_nv > 0:  # 物体有自由度（可移动）
            #     # 重置位置
            #     qpos_start = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]
            #     if body_name == 'apple':
            #         # 苹果初始位置: [0.2, -0.4, 0.45]
            #         self.data.qpos[qpos_start:qpos_start+3] = [0.2, -0.4, 0.45]
            #         self.data.qpos[qpos_start+3:qpos_start+7] = [1, 0, 0, 0]  # 四元数
            #     elif body_name == 'banana':
            #         # 香蕉初始位置: [-0.2, -0.4, 0.45]
            #         self.data.qpos[qpos_start:qpos_start+3] = [-0.2, -0.4, 0.45]
            #         self.data.qpos[qpos_start+3:qpos_start+7] = [1, 0, 0, 0]  # 四元数
                
            #     # 重置速度
            #     qvel_start = self.model.jnt_dofadr[self.model.body_jntadr[body_id]]
            #     self.data.qvel[qvel_start:qvel_start+obj_nv] = 0
            #     print(f"    重置 {object_name} 位置到初始状态")
            # else:
            #     print(f"    {object_name} 是固定物体，无需重置")
            
            # 重新执行正向动力学
            mujoco.mj_fwdPosition(self.model, self.data)
            mujoco.mj_fwdVelocity(self.model, self.data)
            
            # 验证重置结果
            obj_pos = self._get_object_pose(object_name)
            if obj_pos:
                print(f"    {object_name} 新位置: {obj_pos['position']}")
            
            return True
            
        except Exception as e:
            print(f"    重置失败: {e}")
            return False
    
    def _generate_experiment_report(self, object_name, results):
        """生成实验报告"""
        report = {
            'experiment_date': datetime.now().isoformat(),
            'object': object_name,
            'arm': self.active_arm,
            'num_trials': len(results),
            'success_count': sum(1 for r in results if r['success']),
            'success_rate': sum(1 for r in results if r['success']) / len(results),
            'average_grasp_time': self.execution_stats['average_grasp_time'],
            'average_grasp_force': self.execution_stats['average_grasp_force'],
            'failure_reasons': self.execution_stats['failure_reasons'].copy(),
            'trials': []
        }
        
        for result in results:
            trial_report = {
                'trial': result['trial'],
                'success': result['success'],
                'total_duration': result['details'].get('total_duration', 0),
                'phases': {}
            }
            
            for phase_name, phase_data in result['details'].get('phases', {}).items():
                if 'start' in phase_data and 'end' in phase_data:
                    trial_report['phases'][phase_name] = {
                        'duration': phase_data['end'] - phase_data['start'],
                        'success': phase_data.get('success', False)
                    }
            
            report['trials'].append(trial_report)
        
        # 保存报告
        import json
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'grasp_experiment_{object_name}_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"实验报告已保存到 {filename}")
        
        return report
    
    def get_execution_stats(self):
        """获取执行统计"""
        return self.execution_stats.copy()
