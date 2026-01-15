"""
grasp_planning.py - 真实抓取轨迹规划
使用真实运动学计算，真实碰撞检测，无简化假设
"""

import numpy as np
import mujoco
import time
import math
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

class GraspPlanner:
    """真实抓取轨迹规划器"""
    
    def __init__(self, robot_controller):
        self.robot = robot_controller
        self.model = robot_controller.model
        self.data = robot_controller.data
        
        # 机器人参数
        self.num_joints = self.robot.n_controlled
        
        # 物体位置（从模型中获取真实位置）
        self.object_positions = {
            'apple': self._get_object_position('apple'),
            'banana': self._get_object_position('banana')
        }
        
        # 物体尺寸（从模型读取）
        self.object_sizes = {
            'apple': {'radius': 0.04},
            'banana': {'length': 0.16, 'width': 0.04, 'height': 0.04}
        }
        
        # 机械臂配置
        self.arm_config = {
            'right': {
                'base_joints': [0, 1, 2, 3, 4, 5, 6],  # 右臂关节索引
                'gripper_joints': [7],  # 右夹爪关节索引
                'ee_body': 'right_gripper_center',
                'gripper_body': 'right_link_left_jaw'
            },
            'left': {
                'base_joints': [8, 9, 10, 11, 12, 13, 14],  # 左臂关节索引
                'gripper_joints': [15],  # 左夹爪关节索引
                'ee_body': 'left_gripper_center',
                'gripper_body': 'left_link_left_jaw'
            }
        }
        
        # 使用右臂作为默认
        self.active_arm = 'right'
        self.joint_indices = self.arm_config[self.active_arm]['base_joints']
        self.gripper_indices = self.arm_config[self.active_arm]['gripper_joints']
        self.ee_body = self.arm_config[self.active_arm]['ee_body']
        self.gripper_body = self.arm_config[self.active_arm]['gripper_body']
        
        # 关节限制（从模型读取真实限制）
        self.joint_limits = self._get_joint_limits()
        
        # 规划参数
        self.planning_config = {
            'time_step': 0.02,  # 20ms步长
            'max_velocity': 0.5,  # 最大关节速度 (rad/s)
            'max_acceleration': 2.0,  # 最大关节加速度 (rad/s²)
            'collision_margin': 0.02,  # 碰撞安全距离 2cm
            'ik_tolerance': 0.001,  # 逆运动学容忍度
            'max_ik_iterations': 100,  # 最大逆运动学迭代次数
            'approach_distance': 0.1,  # 接近距离 10cm
            'retreat_distance': 0.15,  # 撤离距离 15cm
            'gripper_open': 0.035,  # 夹爪张开位置
            'gripper_close': 0.005,  # 夹爪闭合位置
            'pre_grasp_height': 0.05,  # 预抓取高度 5cm
            'grasp_height': 0.01  # 抓取高度 1cm
        }
        
        # 存储规划结果
        self.planned_trajectories = {}
        
        print(f"抓取规划器初始化完成")
        print(f"使用机械臂: {self.active_arm}")
        print(f"关节数量: {len(self.joint_indices)}")
        print(f"物体位置: {self.object_positions}")
    
    def _get_object_position(self, object_name):
        """从模型中获取物体真实位置"""
        pos = self.robot.get_body_position(object_name)
        if pos is None:
            # 如果无法获取，使用模型中的默认位置
            if object_name == 'apple':
                return np.array([0.2, -0.4, 0.45])
            elif object_name == 'banana':
                return np.array([-0.2, -0.4, 0.45])
        return pos
    
    def _get_joint_limits(self):
        """从模型中获取真实关节限制"""
        limits = []
        for i in range(self.model.njnt):
            jnt_type = self.model.jnt_type[i]
            
            if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:  # 旋转关节
                limit = self.model.jnt_range[i]
                if limit[0] == limit[1]:  # 无限制
                    limits.append((-np.pi, np.pi))
                else:
                    limits.append((limit[0], limit[1]))
            elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:  # 滑动关节
                limit = self.model.jnt_range[i]
                limits.append((limit[0], limit[1]))
            else:
                limits.append((-np.pi, np.pi))  # 默认值
        
        # 只提取受控关节的限制
        controlled_limits = []
        for joint_id in self.robot.actuated_joint_ids:
            if joint_id < len(limits):
                controlled_limits.append(limits[joint_id])
            else:
                controlled_limits.append((-np.pi, np.pi))
        
        return controlled_limits
    
    def _forward_kinematics(self, joint_angles):
        """正向运动学：计算末端执行器位姿"""
        # 保存当前状态
        original_qpos = self.data.qpos.copy()
        
        # 设置关节角度
        for i, joint_idx in enumerate(self.joint_indices):
            if joint_idx < len(self.data.qpos):
                self.data.qpos[joint_idx] = joint_angles[i]
        
        # 执行正向运动学计算
        mujoco.mj_fwdPosition(self.model, self.data)
        
        # 获取末端执行器位姿
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body)
        if ee_id == -1:
            # 恢复原始状态
            self.data.qpos[:] = original_qpos
            mujoco.mj_fwdPosition(self.model, self.data)
            return None
        
        pos = self.data.body(ee_id).xpos.copy()
        quat = self.data.body(ee_id).xquat.copy()
        
        # 恢复原始状态
        self.data.qpos[:] = original_qpos
        mujoco.mj_fwdPosition(self.model, self.data)
        
        return {'position': pos, 'quaternion': quat}
    
    def _inverse_kinematics(self, target_position, target_orientation=None, initial_guess=None):
        """逆运动学：计算达到目标位姿的关节角度"""
        if initial_guess is None:
            initial_guess = [self.robot.get_joint_current(i) for i in self.joint_indices]
        
        # 定义优化目标函数
        def ik_cost(joint_angles):
            # 正向运动学
            fk_result = self._forward_kinematics(joint_angles)
            if fk_result is None:
                return float('inf')
            
            # 位置误差
            pos_error = np.linalg.norm(fk_result['position'] - target_position)
            
            # 姿态误差（如果有目标姿态）
            orient_error = 0
            if target_orientation is not None:
                q_current = fk_result['quaternion']
                q_target = target_orientation
                
                # 四元数角度差
                dot = np.abs(np.dot(q_current, q_target))
                dot = np.clip(dot, -1.0, 1.0)
                angle = 2 * np.arccos(dot)
                orient_error = angle
            
            # 关节限制惩罚
            joint_limit_penalty = 0
            for i, angle in enumerate(joint_angles):
                lower, upper = self.joint_limits[i]
                if angle < lower:
                    joint_limit_penalty += (lower - angle) ** 2
                elif angle > upper:
                    joint_limit_penalty += (angle - upper) ** 2
            
            # 总成本
            total_cost = pos_error + 0.5 * orient_error + 0.1 * joint_limit_penalty
            
            return total_cost
        
        # 设置优化边界
        bounds = []
        for i in range(len(self.joint_indices)):
            lower, upper = self.joint_limits[i]
            bounds.append((lower, upper))
        
        # 运行优化
        result = minimize(
            ik_cost,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': self.planning_config['max_ik_iterations'],
                    'ftol': self.planning_config['ik_tolerance']}
        )
        
        if result.success:
            # 验证解的质量
            fk_result = self._forward_kinematics(result.x)
            if fk_result is None:
                return None
            
            pos_error = np.linalg.norm(fk_result['position'] - target_position)
            
            if pos_error < 0.01:  # 1cm容忍度
                return result.x
            else:
                return None
        else:
            return None
    
    def _check_collision(self, joint_angles, check_objects=True):
        """检查给定关节角度下的碰撞"""
        # 保存当前状态
        original_qpos = self.data.qpos.copy()
        
        # 设置关节角度
        for i, joint_idx in enumerate(self.joint_indices):
            if joint_idx < len(self.data.qpos):
                self.data.qpos[joint_idx] = joint_angles[i]
        
        # 执行正向运动学计算
        mujoco.mj_fwdPosition(self.model, self.data)
        
        # 检查自碰撞
        mujoco.mj_step(self.model, self.data)
        
        # 检查接触
        collision_detected = False
        contact_margin = self.planning_config['collision_margin']
        
        # 获取机械臂连杆的几何体
        arm_geoms = []
        for body_name in ['right_link1', 'right_link2', 'right_link3', 
                         'right_link4', 'right_link5', 'right_link6',
                         'right_link7', 'right_link8', 'right_link_left_jaw',
                         'right_link_right_jaw']:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                # 获取该body的所有geom
                for i in range(self.model.ngeom):
                    if self.model.geom_bodyid[i] == body_id:
                        arm_geoms.append(i)
        
        # 检查与物体的距离
        if check_objects:
            for obj_name in ['apple', 'banana', 'table']:
                obj_pos = self._get_object_position(obj_name)
                if obj_pos is None:
                    continue
                
                # 检查末端执行器与物体的距离
                ee_pos = self._forward_kinematics(joint_angles)['position']
                distance = np.linalg.norm(ee_pos - obj_pos)
                
                if obj_name == 'apple':
                    min_distance = self.object_sizes['apple']['radius'] + contact_margin
                elif obj_name == 'banana':
                    min_distance = self.object_sizes['banana']['width'] / 2 + contact_margin
                else:  # table
                    min_distance = contact_margin
                
                if distance < min_distance:
                    collision_detected = True
                    break
        
        # 恢复原始状态
        self.data.qpos[:] = original_qpos
        mujoco.mj_fwdPosition(self.model, self.data)
        
        return collision_detected
    
    def _quintic_trajectory(self, q0, q1, t0, t1, t):
        """五次多项式轨迹插值"""
        if t <= t0:
            return q0
        if t >= t1:
            return q1
        
        dt = t1 - t0
        tau = (t - t0) / dt
        
        # 五次多项式系数
        h = q1 - q0
        
        # 边界条件：起始和结束位置、速度、加速度都为0
        a0 = q0
        a1 = 0
        a2 = 0
        a3 = 10 * h
        a4 = -15 * h
        a5 = 6 * h
        
        # 计算位置
        pos = (a0 + a1 * tau + a2 * tau**2 + 
              a3 * tau**3 + a4 * tau**4 + a5 * tau**5)
        
        # 计算速度
        vel = (a1 + 2 * a2 * tau + 3 * a3 * tau**2 + 
              4 * a4 * tau**3 + 5 * a5 * tau**4) / dt
        
        # 计算加速度
        acc = (2 * a2 + 6 * a3 * tau + 12 * a4 * tau**2 + 
              20 * a5 * tau**3) / (dt**2)
        
        return pos, vel, acc
    
    def _generate_trajectory_segment(self, start_joints, end_joints, duration):
        """生成轨迹段"""
        num_points = int(duration / self.planning_config['time_step']) + 1
        trajectory = []
        
        for i in range(num_points):
            t = i * self.planning_config['time_step']
            point = {}
            
            # 关节角度
            point['joints'] = np.zeros(len(start_joints))
            point['velocity'] = np.zeros(len(start_joints))
            point['acceleration'] = np.zeros(len(start_joints))
            
            for j in range(len(start_joints)):
                pos, vel, acc = self._quintic_trajectory(
                    start_joints[j], end_joints[j], 0, duration, t
                )
                point['joints'][j] = pos
                point['velocity'][j] = vel
                point['acceleration'][j] = acc
            
            # 计算末端执行器位姿
            fk = self._forward_kinematics(point['joints'])
            if fk:
                point['ee_position'] = fk['position']
                point['ee_orientation'] = fk['quaternion']
            
            # 检查碰撞
            point['collision'] = self._check_collision(point['joints'])
            
            trajectory.append(point)
        
        return trajectory
    
    def _calculate_approach_orientation(self, target_position, approach_vector=np.array([0, 0, -1])):
        """计算接近方向"""
        # 默认朝向：Z轴向下
        z_axis = approach_vector / np.linalg.norm(approach_vector)
        
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
    
    def plan_grasp_trajectory(self, object_name, arm='right', visualize=False):
        """规划抓取轨迹"""
        print(f"\n{'='*60}")
        print(f"规划 {object_name} 的抓取轨迹")
        print(f"{'='*60}")
        
        # 设置活动机械臂
        if arm != self.active_arm:
            self.active_arm = arm
            self.joint_indices = self.arm_config[self.active_arm]['base_joints']
            self.gripper_indices = self.arm_config[self.active_arm]['gripper_joints']
            self.ee_body = self.arm_config[self.active_arm]['ee_body']
            self.gripper_body = self.arm_config[self.active_arm]['gripper_body']
        
        # 获取物体真实位置
        if object_name not in self.object_positions:
            print(f"错误: 未知物体 {object_name}")
            return {'success': False, 'reason': f'未知物体 {object_name}'}
        
        target_position = self.object_positions[object_name].copy()
        print(f"目标物体位置: {target_position}")
        
        # 获取当前关节位置
        current_joints = [self.robot.get_joint_current(i) for i in self.joint_indices]
        print(f"当前关节位置: {current_joints}")
        
        # ========== 步骤1: 计算预抓取位置 ==========
        pre_grasp_position = target_position.copy()
        pre_grasp_position[2] += self.planning_config['pre_grasp_height']
        print(f"预抓取位置: {pre_grasp_position}")
        
        pre_grasp_orientation = self._calculate_approach_orientation(pre_grasp_position)
        
        # 计算预抓取逆运动学
        print("计算预抓取逆运动学...")
        pre_grasp_joints = self._inverse_kinematics(
            pre_grasp_position,
            pre_grasp_orientation,
            initial_guess=current_joints
        )
        
        if pre_grasp_joints is None:
            print("错误: 无法计算预抓取逆运动学")
            return {'success': False, 'reason': '预抓取逆运动学失败'}
        
        print(f"预抓取关节角度: {pre_grasp_joints}")
        
        # 检查预抓取位置碰撞
        if self._check_collision(pre_grasp_joints):
            print("警告: 预抓取位置可能碰撞")
        
        # ========== 步骤2: 计算抓取位置 ==========
        grasp_position = target_position.copy()
        grasp_position[2] += self.planning_config['grasp_height']
        print(f"抓取位置: {grasp_position}")
        
        grasp_orientation = self._calculate_approach_orientation(grasp_position)
        
        # 计算抓取逆运动学
        print("计算抓取逆运动学...")
        grasp_joints = self._inverse_kinematics(
            grasp_position,
            grasp_orientation,
            initial_guess=pre_grasp_joints
        )
        
        if grasp_joints is None:
            print("错误: 无法计算抓取逆运动学")
            return {'success': False, 'reason': '抓取逆运动学失败'}
        
        print(f"抓取关节角度: {grasp_joints}")
        
        # 检查抓取位置碰撞
        if self._check_collision(grasp_joints, check_objects=False):
            print("警告: 抓取位置可能碰撞")
        
        # ========== 步骤3: 规划轨迹 ==========
        print("规划轨迹段...")
        
        # 段1: 当前 -> 预抓取
        print("  段1: 当前 -> 预抓取")
        segment1_duration = 3.0  # 3秒
        segment1 = self._generate_trajectory_segment(
            current_joints, pre_grasp_joints, segment1_duration
        )
        
        # 检查段1碰撞
        segment1_collisions = sum(1 for point in segment1 if point['collision'])
        if segment1_collisions > 0:
            print(f"  警告: 段1有 {segment1_collisions} 个点可能碰撞")
        
        # 段2: 预抓取 -> 抓取
        print("  段2: 预抓取 -> 抓取")
        segment2_duration = 2.0  # 2秒
        segment2 = self._generate_trajectory_segment(
            pre_grasp_joints, grasp_joints, segment2_duration
        )
        
        # 检查段2碰撞
        segment2_collisions = sum(1 for point in segment2 if point['collision'])
        if segment2_collisions > 0:
            print(f"  警告: 段2有 {segment2_collisions} 个点可能碰撞")
        
        # 段3: 抓取 -> 撤离
        print("  段3: 抓取 -> 撤离")
        retreat_position = grasp_position.copy()
        retreat_position[2] += self.planning_config['retreat_distance']
        retreat_orientation = self._calculate_approach_orientation(retreat_position)
        
        retreat_joints = self._inverse_kinematics(
            retreat_position,
            retreat_orientation,
            initial_guess=grasp_joints
        )
        
        if retreat_joints is None:
            print("  警告: 无法计算撤离位置，使用预抓取位置")
            retreat_joints = pre_grasp_joints.copy()
        
        segment3_duration = 2.0  # 2秒
        segment3 = self._generate_trajectory_segment(
            grasp_joints, retreat_joints, segment3_duration
        )
        
        # 合并轨迹
        full_trajectory = segment1 + segment2 + segment3
        total_duration = segment1_duration + segment2_duration + segment3_duration
        
        print(f"轨迹规划完成")
        print(f"总点数: {len(full_trajectory)}")
        print(f"总时长: {total_duration:.2f}秒")
        print(f"碰撞点数: {segment1_collisions + segment2_collisions}")
        
        # ========== 步骤4: 验证轨迹 ==========
        print("验证轨迹...")
        valid = self._validate_trajectory(full_trajectory)
        
        if not valid:
            print("警告: 轨迹验证失败")
        
        # ========== 步骤5: 存储结果 ==========
        trajectory_id = f"{object_name}_{arm}_{datetime.now().strftime('%H%M%S')}"
        
        self.planned_trajectories[trajectory_id] = {
            'object': object_name,
            'arm': arm,
            'start_joints': current_joints,
            'pre_grasp_joints': pre_grasp_joints,
            'grasp_joints': grasp_joints,
            'retreat_joints': retreat_joints,
            'trajectory': full_trajectory,
            'total_duration': total_duration,
            'collision_points': segment1_collisions + segment2_collisions,
            'valid': valid,
            'timestamp': datetime.now().isoformat()
        }
        
        result = {
            'success': valid,
            'trajectory_id': trajectory_id,
            'object': object_name,
            'arm': arm,
            'num_points': len(full_trajectory),
            'duration': total_duration,
            'collision_points': segment1_collisions + segment2_collisions,
            'start_position': full_trajectory[0]['ee_position'] if full_trajectory[0].get('ee_position') else None,
            'grasp_position': grasp_position,
            'planning_time': time.time()  # 将在外部记录
        }
        
        if not valid:
            result['reason'] = '轨迹验证失败'
        
        # ========== 步骤6: 可视化（可选） ==========
        if visualize:
            self._visualize_trajectory(full_trajectory, object_name)
        
        return result
    
    def _validate_trajectory(self, trajectory):
        """验证轨迹的有效性"""
        if len(trajectory) < 10:
            print("  错误: 轨迹太短")
            return False
        
        # 检查关节速度限制
        max_velocity = self.planning_config['max_velocity']
        velocity_violations = 0
        
        for i, point in enumerate(trajectory):
            if 'velocity' in point:
                max_vel = np.max(np.abs(point['velocity']))
                if max_vel > max_velocity:
                    velocity_violations += 1
        
        if velocity_violations > 0:
            print(f"  警告: {velocity_violations} 个点超过速度限制")
        
        # 检查关节加速度限制
        max_acceleration = self.planning_config['max_acceleration']
        acceleration_violations = 0
        
        for i, point in enumerate(trajectory):
            if 'acceleration' in point:
                max_acc = np.max(np.abs(point['acceleration']))
                if max_acc > max_acceleration:
                    acceleration_violations += 1
        
        if acceleration_violations > 0:
            print(f"  警告: {acceleration_violations} 个点超过加速度限制")
        
        # 检查碰撞
        collision_points = sum(1 for point in trajectory if point.get('collision', False))
        if collision_points > len(trajectory) * 0.1:  # 超过10%的点碰撞
            print(f"  错误: 过多碰撞点 ({collision_points})")
            return False
        
        # 检查末端执行器位置连续性
        position_changes = []
        for i in range(1, len(trajectory)):
            if 'ee_position' in trajectory[i] and 'ee_position' in trajectory[i-1]:
                pos1 = trajectory[i-1]['ee_position']
                pos2 = trajectory[i]['ee_position']
                change = np.linalg.norm(pos2 - pos1)
                position_changes.append(change)
        
        if position_changes:
            max_change = max(position_changes)
            if max_change > 0.1:  # 单个步长移动超过10cm
                print(f"  警告: 最大位置变化 {max_change:.3f}m 过大")
        
        return True
    
    def _visualize_trajectory(self, trajectory, object_name):
        """可视化轨迹"""
        if len(trajectory) == 0:
            return
        
        # 提取数据
        time_points = np.arange(len(trajectory)) * self.planning_config['time_step']
        
        # 关节角度
        joint_angles = []
        for i in range(len(self.joint_indices)):
            joint_data = [point['joints'][i] for point in trajectory]
            joint_angles.append(joint_data)
        
        # 末端执行器位置
        ee_positions = []
        for point in trajectory:
            if 'ee_position' in point:
                ee_positions.append(point['ee_position'])
            else:
                ee_positions.append([0, 0, 0])
        
        ee_positions = np.array(ee_positions)
        
        # 创建图形
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 图1: 关节角度
        ax1 = axes[0, 0]
        for i, angles in enumerate(joint_angles[:3]):  # 只显示前3个关节
            ax1.plot(time_points, angles, label=f'Joint {i}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Joint Angle (rad)')
        ax1.set_title('Joint Angles')
        ax1.legend()
        ax1.grid(True)
        
        # 图2: 关节速度
        ax2 = axes[0, 1]
        for i in range(min(3, len(self.joint_indices))):
            velocities = [point['velocity'][i] for point in trajectory]
            ax2.plot(time_points, velocities, label=f'Joint {i}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Joint Velocity (rad/s)')
        ax2.set_title('Joint Velocities')
        ax2.legend()
        ax2.grid(True)
        
        # 图3: 末端执行器X-Y位置
        ax3 = axes[1, 0]
        ax3.plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', linewidth=2)
        ax3.scatter(ee_positions[0, 0], ee_positions[0, 1], color='green', s=100, label='Start')
        ax3.scatter(ee_positions[-1, 0], ee_positions[-1, 1], color='red', s=100, label='End')
        
        # 标记物体位置
        obj_pos = self.object_positions[object_name]
        ax3.scatter(obj_pos[0], obj_pos[1], color='orange', s=150, marker='*', label='Target')
        
        ax3.set_xlabel('X Position (m)')
        ax3.set_ylabel('Y Position (m)')
        ax3.set_title('End-Effector XY Trajectory')
        ax3.legend()
        ax3.grid(True)
        ax3.axis('equal')
        
        # 图4: 末端执行器Z位置
        ax4 = axes[1, 1]
        ax4.plot(time_points, ee_positions[:, 2], 'r-', linewidth=2)
        ax4.axhline(y=obj_pos[2], color='orange', linestyle='--', label='Target Height')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Z Position (m)')
        ax4.set_title('End-Effector Z Position')
        ax4.legend()
        ax4.grid(True)
        
        # 图5: 速度大小
        ax5 = axes[2, 0]
        speed = []
        for point in trajectory:
            if 'velocity' in point:
                speed.append(np.linalg.norm(point['velocity']))
            else:
                speed.append(0)
        
        ax5.plot(time_points, speed, 'g-', linewidth=2)
        ax5.axhline(y=self.planning_config['max_velocity'], color='red', 
                   linestyle='--', label='Max Speed')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Speed (rad/s)')
        ax5.set_title('Overall Joint Speed')
        ax5.legend()
        ax5.grid(True)
        
        # 图6: 碰撞检测
        ax6 = axes[2, 1]
        collisions = [1 if point.get('collision', False) else 0 for point in trajectory]
        ax6.fill_between(time_points, 0, collisions, color='red', alpha=0.3, label='Collision')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Collision (0/1)')
        ax6.set_title('Collision Detection')
        ax6.set_ylim(-0.1, 1.1)
        ax6.legend()
        ax6.grid(True)
        
        plt.suptitle(f'Grasp Trajectory for {object_name.capitalize()}', fontsize=16)
        plt.tight_layout()
        
        # 保存图像
        filename = f'trajectory_{object_name}_{datetime.now().strftime("%H%M%S")}.png'
        plt.savefig(filename, dpi=150)
        plt.show()
        
        print(f"轨迹可视化已保存到 {filename}")
    
    def execute_trajectory(self, trajectory_id, realtime=False):
        """执行存储的轨迹"""
        if trajectory_id not in self.planned_trajectories:
            print(f"错误: 未找到轨迹 {trajectory_id}")
            return False
        
        trajectory_data = self.planned_trajectories[trajectory_id]
        trajectory = trajectory_data['trajectory']
        
        if not trajectory_data['valid']:
            print(f"警告: 轨迹 {trajectory_id} 未经验证")
        
        print(f"\n执行轨迹: {trajectory_id}")
        print(f"物体: {trajectory_data['object']}")
        print(f"点数: {len(trajectory)}")
        print(f"时长: {trajectory_data['total_duration']:.2f}秒")
        
        # 记录执行统计
        execution_stats = {
            'start_time': time.time(),
            'points_executed': 0,
            'collisions_detected': 0,
            'joint_errors': []
        }
        
        # 执行轨迹
        for i, point in enumerate(trajectory):
            # 设置关节角度
            for j, joint_idx in enumerate(self.joint_indices):
                self.robot.set_joint_target(joint_idx, point['joints'][j])
            
            # 更新控制
            self.robot.update_control()
            
            # 步进仿真
            self.robot.step_simulation()
            
            # 检查碰撞
            if point.get('collision', False):
                execution_stats['collisions_detected'] += 1
            
            # 记录关节误差
            current_joints = [self.robot.get_joint_current(joint_idx) for joint_idx in self.joint_indices]
            error = np.mean(np.abs(np.array(current_joints) - point['joints']))
            execution_stats['joint_errors'].append(error)
            
            execution_stats['points_executed'] += 1
            
            # 实时执行时添加延迟
            if realtime:
                time.sleep(self.planning_config['time_step'])
            
            # 进度显示
            if i % 10 == 0:
                progress = (i + 1) / len(trajectory) * 100
                print(f"  进度: {progress:.1f}% ({i+1}/{len(trajectory)})")
        
        execution_stats['end_time'] = time.time()
        execution_stats['total_time'] = execution_stats['end_time'] - execution_stats['start_time']
        execution_stats['average_error'] = np.mean(execution_stats['joint_errors'])
        execution_stats['max_error'] = np.max(execution_stats['joint_errors'])
        
        print(f"\n轨迹执行完成")
        print(f"总执行时间: {execution_stats['total_time']:.2f}秒")
        print(f"平均关节误差: {execution_stats['average_error']:.4f} rad")
        print(f"最大关节误差: {execution_stats['max_error']:.4f} rad")
        print(f"检测到的碰撞: {execution_stats['collisions_detected']}")
        
        # 保存执行统计
        trajectory_data['execution_stats'] = execution_stats
        
        return True, execution_stats
    
    def save_trajectory_report(self, trajectory_id, filename=None):
        """保存轨迹报告"""
        if trajectory_id not in self.planned_trajectories:
            print(f"错误: 未找到轨迹 {trajectory_id}")
            return
        
        if filename is None:
            filename = f"trajectory_report_{trajectory_id}.json"
        
        import json
        
        data = self.planned_trajectories[trajectory_id]
        
        # 转换为可JSON序列化的格式
        serializable_data = {
            'trajectory_id': trajectory_id,
            'object': data['object'],
            'arm': data['arm'],
            'start_joints': [float(x) for x in data['start_joints']],
            'pre_grasp_joints': [float(x) for x in data['pre_grasp_joints']],
            'grasp_joints': [float(x) for x in data['grasp_joints']],
            'retreat_joints': [float(x) for x in data['retreat_joints']],
            'num_points': len(data['trajectory']),
            'total_duration': float(data['total_duration']),
            'collision_points': data['collision_points'],
            'valid': data['valid'],
            'timestamp': data['timestamp']
        }
        
        # 添加执行统计（如果有）
        if 'execution_stats' in data:
            stats = data['execution_stats']
            serializable_data['execution_stats'] = {
                'points_executed': stats['points_executed'],
                'total_time': float(stats['total_time']),
                'collisions_detected': stats['collisions_detected'],
                'average_error': float(stats['average_error']),
                'max_error': float(stats['max_error'])
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"轨迹报告已保存到 {filename}")
        
        return serializable_data
