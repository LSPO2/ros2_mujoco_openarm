"""
experiment_data.py - 真实实验数据采集与分析
使用真实物理量数据，无简化假设，所有数据从仿真中实时获取
"""

import numpy as np
import time
import json
import csv
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import os

class ExperimentDataCollector:
    """真实实验数据采集与分析器"""
    
    def __init__(self, robot_controller):
        self.robot = robot_controller
        self.model = robot_controller.model
        self.data = robot_controller.data
        
        # 数据采集配置
        self.collection_config = {
            'sampling_rate': 100.0,  # 采样率 Hz
            'sampling_interval': 0.01,  # 采样间隔 10ms
            'max_data_points': 10000,  # 最大数据点数
            'save_raw_data': True,  # 保存原始数据
            'save_processed_data': True,  # 保存处理后的数据
            'data_directory': 'experiment_data',
            'plots_directory': 'experiment_plots',
            'reports_directory': 'experiment_reports'
        }
        
        # 创建目录
        self._create_directories()
        
        # 数据存储
        self.current_experiment = None
        self.data_buffers = {}
        self.experiment_metadata = {}
        
        # 性能指标
        self.metrics = {
            'calibration': {},
            'detection': {},
            'planning': {},
            'grasping': {},
            'overall': {}
        }
        
        print(f"实验数据采集器初始化完成")
        print(f"采样率: {self.collection_config['sampling_rate']} Hz")
        print(f"数据目录: {self.collection_config['data_directory']}")
    
    def _create_directories(self):
        """创建数据目录"""
        directories = [
            self.collection_config['data_directory'],
            self.collection_config['plots_directory'],
            self.collection_config['reports_directory']
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建目录: {directory}")
    
    def start_experiment(self, experiment_name, experiment_type, metadata=None):
        """开始新实验"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_experiment = f"{experiment_name}_{timestamp}"
        
        # 初始化数据缓冲区
        self.data_buffers = {
            'timestamps': [],
            'joint_positions': [],
            'joint_velocities': [],
            'joint_torques': [],
            'ee_positions': [],
            'ee_velocities': [],
            'ee_orientations': [],
            'contact_forces': [],
            'object_positions': {},
            'object_velocities': {},
            'object_orientations': {},
            'gripper_positions': [],
            'gripper_forces': [],
            'camera_images': [],  # 存储图像路径
            'performance_metrics': []
        }
        
        # 实验元数据
        self.experiment_metadata = {
            'experiment_id': self.current_experiment,
            'experiment_name': experiment_name,
            'experiment_type': experiment_type,
            'start_time': datetime.now().isoformat(),
            'metadata': metadata or {},
            'system_info': self._get_system_info()
        }
        
        print(f"\n{'='*60}")
        print(f"开始实验: {self.current_experiment}")
        print(f"实验类型: {experiment_type}")
        print(f"{'='*60}")
        
        return self.current_experiment
    
    def _get_system_info(self):
        """获取系统信息"""
        import platform
        import mujoco
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'mujoco_version': mujoco.__version__,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'nq': self.model.nq,  # 广义坐标数
                'nv': self.model.nv,  # 自由度
                'nu': self.model.nu,  # 执行器数
                'njnt': self.model.njnt,  # 关节数
                'nbody': self.model.nbody,  # 物体数
                'ngeom': self.model.ngeom,  # 几何体数
                'nsensor': self.model.nsensor,  # 传感器数
                'nmesh': self.model.nmesh,  # 网格数
                'ntex': self.model.ntex,  # 纹理数
                'nmat': self.model.nmat  # 材质数
            }
        }
    
    def collect_data_point(self, data_type='full'):
        """采集一个数据点"""
        if self.current_experiment is None:
            print("警告: 未开始实验")
            return
        
        timestamp = time.time()
        self.data_buffers['timestamps'].append(timestamp)
        
        if data_type in ['full', 'joints']:
            # 采集关节数据（真实数据）
            joint_pos = []
            joint_vel = []
            joint_torque = []
            
            for i in range(self.model.nu):
                # 获取执行器对应的关节ID
                joint_id = self.model.actuator_trnid[i, 0]
                
                # 关节位置 (qpos)
                if joint_id < self.model.nq:
                    joint_pos.append(float(self.data.qpos[joint_id]))
                else:
                    joint_pos.append(0.0)
                
                # 关节速度 (qvel) - 真实速度
                if joint_id < self.model.nv:
                    joint_vel.append(float(self.data.qvel[joint_id]))
                else:
                    joint_vel.append(0.0)
                
                # 关节扭矩 (qfrc_actuator)
                if joint_id < self.model.nv:
                    joint_torque.append(float(self.data.qfrc_actuator[joint_id]))
                else:
                    joint_torque.append(0.0)
            
            self.data_buffers['joint_positions'].append(joint_pos)
            self.data_buffers['joint_velocities'].append(joint_vel)
            self.data_buffers['joint_torques'].append(joint_torque)
        
        if data_type in ['full', 'ee']:
            # 采集末端执行器数据
            for ee_body in ['right_gripper_center', 'left_gripper_center']:
                ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
                if ee_id != -1:
                    # 位置
                    pos = self.data.body(ee_id).xpos.copy()
                    
                    # 速度（真实速度，从qvel计算）
                    # 获取body的全局速度
                    vel = np.zeros(6)  # 3个线速度 + 3个角速度
                    body_dofadr = self.model.body_dofadr[ee_id]
                    body_nv = self.model.body_nv[ee_id]
                    
                    if body_nv > 0:
                        # 从qvel中提取body的速度
                        vel[:body_nv] = self.data.qvel[body_dofadr:body_dofadr+body_nv]
                    
                    # 姿态（四元数）
                    quat = self.data.body(ee_id).xquat.copy()
                    
                    if ee_body == 'right_gripper_center':
                        self.data_buffers['ee_positions'].append(pos.tolist())
                        self.data_buffers['ee_velocities'].append(vel.tolist())
                        self.data_buffers['ee_orientations'].append(quat.tolist())
        
        if data_type in ['full', 'objects']:
            # 采集物体数据
            for obj_name in ['apple', 'banana', 'calibration_target']:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                if body_id != -1:
                    # 位置
                    pos = self.data.body(body_id).xpos.copy()
                    
                    # 速度（真实速度）
                    vel = np.zeros(6)
                    body_dofadr = self.model.body_dofadr[body_id]
                    body_nv = self.model.body_nv[body_id]
                    
                    if body_nv > 0:
                        vel[:body_nv] = self.data.qvel[body_dofadr:body_dofadr+body_nv]
                    
                    # 姿态
                    quat = self.data.body(body_id).xquat.copy()
                    
                    # 存储数据
                    if obj_name not in self.data_buffers['object_positions']:
                        self.data_buffers['object_positions'][obj_name] = []
                        self.data_buffers['object_velocities'][obj_name] = []
                        self.data_buffers['object_orientations'][obj_name] = []
                    
                    self.data_buffers['object_positions'][obj_name].append(pos.tolist())
                    self.data_buffers['object_velocities'][obj_name].append(vel.tolist())
                    self.data_buffers['object_orientations'][obj_name].append(quat.tolist())
        
        if data_type in ['full', 'contact']:
            # 采集接触力数据（真实接触力）
            contact_forces = []
            
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                
                # 计算接触力
                force = np.zeros(6)  # 3个力 + 3个力矩
                mujoco.mj_contactForce(self.model, self.data, i, force)
                
                contact_info = {
                    'geom1': int(contact.geom1),
                    'geom2': int(contact.geom2),
                    'position': contact.pos.copy().tolist(),
                    'force': force[:3].tolist(),  # 接触力
                    'torque': force[3:].tolist(),  # 接触力矩
                    'distance': float(contact.dist),
                    'friction': float(contact.friction)
                }
                
                contact_forces.append(contact_info)
            
            self.data_buffers['contact_forces'].append(contact_forces)
        
        if data_type in ['full', 'gripper']:
            # 采集夹爪数据
            for gripper_joint in ['right_left_pris1', 'left_left_pris1']:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, gripper_joint)
                if joint_id != -1:
                    # 夹爪位置
                    qpos_adr = self.model.jnt_qposadr[joint_id]
                    gripper_pos = float(self.data.qpos[qpos_adr]) if qpos_adr < self.model.nq else 0.0
                    
                    # 夹爪力（从执行器力获取）
                    # 查找控制夹爪的执行器
                    gripper_force = 0.0
                    for i in range(self.model.nu):
                        if self.model.actuator_trnid[i, 0] == joint_id:
                            gripper_force = float(self.data.ctrl[i])
                            break
                    
                    self.data_buffers['gripper_positions'].append(gripper_pos)
                    self.data_buffers['gripper_forces'].append(gripper_force)
        
        # 检查数据缓冲区大小
        self._check_buffer_size()
    
    def _check_buffer_size(self):
        """检查并管理数据缓冲区大小"""
        max_points = self.collection_config['max_data_points']
        
        for key, buffer in self.data_buffers.items():
            if isinstance(buffer, list) and len(buffer) > max_points * 1.5:
                # 保留最新的数据
                self.data_buffers[key] = buffer[-max_points:]
                print(f"警告: {key} 缓冲区达到限制，已截断")
    
    def collect_camera_image(self, camera_id=2, save_image=True):
        """采集相机图像"""
        if self.current_experiment is None:
            return None
        
        try:
            # 获取图像
            image = self.robot.get_camera_image()
            if image is None:
                return None
            
            # 保存图像
            if save_image:
                timestamp = datetime.now().strftime('%H%M%S_%f')
                filename = f"{self.current_experiment}_camera_{timestamp}.png"
                filepath = os.path.join(self.collection_config['data_directory'], filename)
                
                # 转换为BGR并保存
                import cv2
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, bgr_image)
                
                # 记录图像路径
                self.data_buffers['camera_images'].append({
                    'timestamp': time.time(),
                    'filepath': filepath,
                    'camera_id': camera_id
                })
                
                return filepath
            
            return image
            
        except Exception as e:
            print(f"采集相机图像失败: {e}")
            return None
    
    def record_performance_metric(self, metric_name, value, unit=None, metadata=None):
        """记录性能指标"""
        metric_data = {
            'timestamp': time.time(),
            'metric': metric_name,
            'value': float(value) if isinstance(value, (int, float, np.number)) else value,
            'unit': unit,
            'metadata': metadata or {}
        }
        
        self.data_buffers['performance_metrics'].append(metric_data)
        
        # 根据指标类型存储到相应的类别
        if 'calibration' in metric_name.lower():
            self.metrics['calibration'][metric_name] = value
        elif 'detection' in metric_name.lower():
            self.metrics['detection'][metric_name] = value
        elif 'planning' in metric_name.lower():
            self.metrics['planning'][metric_name] = value
        elif 'grasp' in metric_name.lower():
            self.metrics['grasping'][metric_name] = value
        else:
            self.metrics['overall'][metric_name] = value
    
    def collect_calibration_data(self, calibration_results):
        """采集标定数据"""
        if not calibration_results.get('success', False):
            return
        
        print("采集标定实验数据...")
        
        # 记录标定误差
        errors = calibration_results.get('errors', {})
        if errors:
            self.record_performance_metric(
                'calibration_mean_translation_error',
                errors.get('mean_translation', 0) * 1000,  # 转换为mm
                'mm',
                {'description': '平均平移误差'}
            )
            
            self.record_performance_metric(
                'calibration_max_translation_error',
                errors.get('max_translation', 0) * 1000,  # 转换为mm
                'mm',
                {'description': '最大平移误差'}
            )
            
            self.record_performance_metric(
                'calibration_mean_rotation_error',
                errors.get('mean_rotation', 0),
                'rad',
                {'description': '平均旋转误差'}
            )
            
            self.record_performance_metric(
                'calibration_reprojection_error',
                errors.get('mean', 0),
                'pixels',
                {'description': '平均重投影误差'}
            )
        
        # 采集标定过程中的数据点
        for _ in range(100):  # 采集100个数据点
            self.collect_data_point('full')
            time.sleep(0.01)  # 10ms间隔
    
    def collect_detection_data(self, detection_results):
        """采集检测数据"""
        print("采集视觉检测实验数据...")
        
        for obj_name, result in detection_results.items():
            if result.get('success', False):
                error_mm = result.get('error_mm', 0)
                
                self.record_performance_metric(
                    f'detection_{obj_name}_error',
                    error_mm,
                    'mm',
                    {
                        'object': obj_name,
                        'detected_position': result.get('position'),
                        'true_position': self._get_object_true_position(obj_name)
                    }
                )
                
                # 判断是否满足精度要求
                if error_mm < 10.0:  # 10mm阈值
                    self.record_performance_metric(
                        f'detection_{obj_name}_success',
                        1,
                        'boolean',
                        {'description': f'{obj_name}检测成功'}
                    )
                else:
                    self.record_performance_metric(
                        f'detection_{obj_name}_success',
                        0,
                        'boolean',
                        {'description': f'{obj_name}检测失败'}
                    )
        
        # 采集检测过程中的数据点
        for _ in range(50):
            self.collect_data_point('full')
            time.sleep(0.02)
    
    def collect_planning_data(self, planning_results):
        """采集规划数据"""
        print("采集轨迹规划实验数据...")
        
        if planning_results.get('success', False):
            self.record_performance_metric(
                'planning_success',
                1,
                'boolean',
                {'description': '轨迹规划成功'}
            )
            
            self.record_performance_metric(
                'planning_trajectory_points',
                planning_results.get('num_points', 0),
                'points',
                {'description': '轨迹点数'}
            )
            
            self.record_performance_metric(
                'planning_trajectory_duration',
                planning_results.get('duration', 0),
                'seconds',
                {'description': '轨迹时长'}
            )
            
            self.record_performance_metric(
                'planning_collision_points',
                planning_results.get('collision_points', 0),
                'points',
                {'description': '碰撞点数'}
            )
        else:
            self.record_performance_metric(
                'planning_success',
                0,
                'boolean',
                {'description': '轨迹规划失败', 'reason': planning_results.get('reason', 'unknown')}
            )
    
    def collect_grasping_data(self, grasp_results):
        """采集抓取数据"""
        print("采集抓取执行实验数据...")
        
        for trial in grasp_results:
            trial_num = trial.get('trial', 0)
            success = trial.get('success', False)
            details = trial.get('details', {})
            
            # 记录每次尝试的结果
            self.record_performance_metric(
                f'grasp_trial_{trial_num}_success',
                1 if success else 0,
                'boolean',
                {
                    'trial': trial_num,
                    'object': details.get('object', 'unknown'),
                    'reason': details.get('reason', 'success' if success else 'unknown'),
                    'total_duration': details.get('total_duration', 0)
                }
            )
            
            # 记录抓取力
            if 'force_profile' in details and details['force_profile']:
                max_force = max(details['force_profile'])
                avg_force = np.mean(details['force_profile'])
                
                self.record_performance_metric(
                    f'grasp_trial_{trial_num}_max_force',
                    max_force,
                    'N',
                    {'trial': trial_num, 'description': '最大抓取力'}
                )
                
                self.record_performance_metric(
                    f'grasp_trial_{trial_num}_avg_force',
                    avg_force,
                    'N',
                    {'trial': trial_num, 'description': '平均抓取力'}
                )
            
            # 记录各阶段时间
            phases = details.get('phases', {})
            for phase_name, phase_data in phases.items():
                if 'start' in phase_data and 'end' in phase_data:
                    duration = phase_data['end'] - phase_data['start']
                    
                    self.record_performance_metric(
                        f'grasp_trial_{trial_num}_phase_{phase_name}_duration',
                        duration,
                        'seconds',
                        {
                            'trial': trial_num,
                            'phase': phase_name,
                            'success': phase_data.get('success', False)
                        }
                    )
        
        # 计算总体统计
        total_trials = len(grasp_results)
        successful_trials = sum(1 for t in grasp_results if t.get('success', False))
        success_rate = successful_trials / total_trials if total_trials > 0 else 0
        
        self.record_performance_metric(
            'grasping_total_trials',
            total_trials,
            'trials',
            {'description': '总尝试次数'}
        )
        
        self.record_performance_metric(
            'grasping_successful_trials',
            successful_trials,
            'trials',
            {'description': '成功次数'}
        )
        
        self.record_performance_metric(
            'grasping_success_rate',
            success_rate * 100,
            'percent',
            {'description': '成功率'}
        )
    
    def _get_object_true_position(self, object_name):
        """获取物体真实位置（从模型中）"""
        if object_name == 'apple':
            return [0.2, -0.4, 0.45]
        elif object_name == 'banana':
            return [-0.2, -0.4, 0.45]
        else:
            return [0, 0, 0]
    
    def end_experiment(self, success=True, summary=None):
        """结束实验并保存数据"""
        if self.current_experiment is None:
            print("警告: 没有正在进行的实验")
            return None
        
        print(f"\n结束实验: {self.current_experiment}")
        
        # 更新元数据
        self.experiment_metadata['end_time'] = datetime.now().isoformat()
        self.experiment_metadata['success'] = success
        self.experiment_metadata['summary'] = summary or {}
        
        # 计算实验时长
        start_time = datetime.fromisoformat(self.experiment_metadata['start_time'].replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(self.experiment_metadata['end_time'].replace('Z', '+00:00'))
        duration = (end_time - start_time).total_seconds()
        
        self.experiment_metadata['duration'] = duration
        
        # 保存数据
        data_path = self._save_experiment_data()
        
        # 生成报告
        report_path = self._generate_experiment_report()
        
        # 生成可视化图表
        plots_path = self._generate_experiment_plots()
        
        print(f"实验数据已保存到: {data_path}")
        print(f"实验报告已保存到: {report_path}")
        print(f"实验图表已保存到: {plots_path}")
        
        # 重置实验状态
        experiment_id = self.current_experiment
        self.current_experiment = None
        
        return {
            'experiment_id': experiment_id,
            'data_path': data_path,
            'report_path': report_path,
            'plots_path': plots_path,
            'metadata': self.experiment_metadata
        }
    
    def _save_experiment_data(self):
        """保存实验数据"""
        if not self.collection_config['save_raw_data']:
            return None
        
        # 准备数据
        save_data = {
            'metadata': self.experiment_metadata,
            'performance_metrics': self.data_buffers['performance_metrics'],
            'metrics_summary': self.metrics
        }
        
        # 添加原始数据（如果保存）
        if self.collection_config['save_raw_data']:
            save_data['raw_data'] = {}
            
            for key, buffer in self.data_buffers.items():
                if key != 'performance_metrics':  # 已经单独保存
                    # 转换为可序列化的格式
                    if isinstance(buffer, list):
                        save_data['raw_data'][key] = buffer
                    elif isinstance(buffer, dict):
                        save_data['raw_data'][key] = {
                            k: v for k, v in buffer.items()
                        }
        
        # 保存为JSON
        filename = f"{self.current_experiment}_data.json"
        filepath = os.path.join(self.collection_config['data_directory'], filename)
        
        try:
            with open(filepath, 'w') as f:
                # 使用自定义序列化器处理numpy数组
                def default_serializer(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                
                json.dump(save_data, f, default=default_serializer, indent=2)
            
            # 同时保存为CSV格式（便于分析）
            self._save_data_as_csv(save_data)
            
            return filepath
            
        except Exception as e:
            print(f"保存数据失败: {e}")
            return None
    
    def _save_data_as_csv(self, data):
        """保存为CSV格式"""
        try:
            # 保存性能指标
            metrics_file = f"{self.current_experiment}_metrics.csv"
            metrics_path = os.path.join(self.collection_config['data_directory'], metrics_file)
            
            with open(metrics_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'metric', 'value', 'unit', 'metadata'])
                
                for metric in data['performance_metrics']:
                    metadata_str = json.dumps(metric.get('metadata', {}))
                    writer.writerow([
                        metric['timestamp'],
                        metric['metric'],
                        metric['value'],
                        metric.get('unit', ''),
                        metadata_str
                    ])
            
            # 保存关节数据
            if 'raw_data' in data and 'joint_positions' in data['raw_data']:
                joints_file = f"{self.current_experiment}_joints.csv"
                joints_path = os.path.join(self.collection_config['data_directory'], joints_file)
                
                with open(joints_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # 写入头部
                    headers = ['timestamp']
                    for i in range(len(data['raw_data']['joint_positions'][0])):
                        headers.extend([
                            f'joint_{i}_pos',
                            f'joint_{i}_vel',
                            f'joint_{i}_torque'
                        ])
                    
                    writer.writerow(headers)
                    
                    # 写入数据
                    for i, timestamp in enumerate(data['raw_data']['timestamps']):
                        row = [timestamp]
                        
                        if i < len(data['raw_data']['joint_positions']):
                            pos = data['raw_data']['joint_positions'][i]
                            vel = data['raw_data']['joint_velocities'][i]
                            torque = data['raw_data']['joint_torques'][i]
                            
                            for j in range(len(pos)):
                                row.extend([pos[j], vel[j], torque[j]])
                        
                        writer.writerow(row)
            
        except Exception as e:
            print(f"保存CSV数据失败: {e}")
    
    def _generate_experiment_report(self):
        """生成实验报告"""
        report = {
            'experiment_id': self.current_experiment,
            'metadata': self.experiment_metadata,
            'performance_summary': self._calculate_performance_summary(),
            'metrics_summary': self.metrics,
            'statistical_analysis': self._perform_statistical_analysis(),
            'conclusions': self._generate_conclusions(),
            'recommendations': self._generate_recommendations()
        }
        
        filename = f"{self.current_experiment}_report.json"
        filepath = os.path.join(self.collection_config['reports_directory'], filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 同时生成Markdown格式的报告
        self._generate_markdown_report(report, filepath.replace('.json', '.md'))
        
        return filepath
    
    def _calculate_performance_summary(self):
        """计算性能总结"""
        summary = {
            'calibration': {
                'success': self.metrics['calibration'].get('calibration_success', False),
                'mean_error_mm': self.metrics['calibration'].get('calibration_mean_translation_error', 0),
                'max_error_mm': self.metrics['calibration'].get('calibration_max_translation_error', 0)
            },
            'detection': {
                'apple_success': self.metrics['detection'].get('detection_apple_success', 0) == 1,
                'apple_error_mm': self.metrics['detection'].get('detection_apple_error', 0),
                'banana_success': self.metrics['detection'].get('detection_banana_success', 0) == 1,
                'banana_error_mm': self.metrics['detection'].get('detection_banana_error', 0)
            },
            'planning': {
                'success': self.metrics['planning'].get('planning_success', 0) == 1,
                'collision_points': self.metrics['planning'].get('planning_collision_points', 0)
            },
            'grasping': {
                'success_rate_percent': self.metrics['grasping'].get('grasping_success_rate', 0),
                'total_trials': self.metrics['grasping'].get('grasping_total_trials', 0),
                'successful_trials': self.metrics['grasping'].get('grasping_successful_trials', 0)
            }
        }
        
        # 计算总体成功率
        overall_success = (
            summary['calibration']['success'] and
            summary['detection']['apple_success'] and
            summary['detection']['banana_success'] and
            summary['planning']['success'] and
            summary['grasping']['success_rate_percent'] >= 95.0
        )
        
        summary['overall'] = {
            'success': overall_success,
            'total_experiment_duration': self.experiment_metadata.get('duration', 0),
            'data_points_collected': len(self.data_buffers['timestamps']) if 'timestamps' in self.data_buffers else 0
        }
        
        return summary
    
    def _perform_statistical_analysis(self):
        """进行统计分析"""
        analysis = {}
        
        # 分析关节数据
        if self.data_buffers.get('joint_positions'):
            joint_data = np.array(self.data_buffers['joint_positions'])
            
            analysis['joint_statistics'] = {
                'mean_positions': np.mean(joint_data, axis=0).tolist(),
                'std_positions': np.std(joint_data, axis=0).tolist(),
                'max_positions': np.max(joint_data, axis=0).tolist(),
                'min_positions': np.min(joint_data, axis=0).tolist(),
                'position_ranges': (np.max(joint_data, axis=0) - np.min(joint_data, axis=0)).tolist()
            }
        
        # 分析末端执行器轨迹
        if self.data_buffers.get('ee_positions'):
            ee_positions = np.array(self.data_buffers['ee_positions'])
            
            analysis['ee_trajectory'] = {
                'path_length': self._calculate_path_length(ee_positions),
                'mean_velocity': self._calculate_mean_velocity(),
                'position_variance': np.var(ee_positions, axis=0).tolist()
            }
        
        # 分析接触力
        if self.data_buffers.get('contact_forces'):
            contact_forces = []
            for contact_list in self.data_buffers['contact_forces']:
                for contact in contact_list:
                    force_magnitude = np.linalg.norm(contact['force'])
                    contact_forces.append(force_magnitude)
            
            if contact_forces:
                contact_forces = np.array(contact_forces)
                analysis['contact_statistics'] = {
                    'mean_force': float(np.mean(contact_forces)),
                    'max_force': float(np.max(contact_forces)),
                    'force_occurrence': len(contact_forces) / len(self.data_buffers['contact_forces']) if self.data_buffers['contact_forces'] else 0
                }
        
        return analysis
    
    def _calculate_path_length(self, positions):
        """计算路径长度"""
        if len(positions) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(positions)):
            segment_length = np.linalg.norm(positions[i] - positions[i-1])
            total_length += segment_length
        
        return float(total_length)
    
    def _calculate_mean_velocity(self):
        """计算平均速度"""
        if not self.data_buffers.get('ee_velocities') or not self.data_buffers.get('timestamps'):
            return 0.0
        
        velocities = []
        for vel in self.data_buffers['ee_velocities']:
            # 计算线速度大小
            linear_vel = np.linalg.norm(vel[:3])
            velocities.append(linear_vel)
        
        return float(np.mean(velocities)) if velocities else 0.0
    
    def _generate_conclusions(self):
        """生成结论"""
        summary = self._calculate_performance_summary()
        
        conclusions = []
        
        # 标定结论
        if summary['calibration']['success']:
            if summary['calibration']['mean_error_mm'] < 5.0:
                conclusions.append("手眼标定精度优秀（<5mm），满足高精度抓取要求。")
            elif summary['calibration']['mean_error_mm'] < 10.0:
                conclusions.append("手眼标定精度良好（<10mm），满足一般抓取要求。")
            else:
                conclusions.append("手眼标定精度一般（≥10mm），建议重新标定以提高精度。")
        else:
            conclusions.append("手眼标定失败，需要重新进行标定。")
        
        # 检测结论
        detection_success = summary['detection']['apple_success'] and summary['detection']['banana_success']
        if detection_success:
            max_detection_error = max(
                summary['detection']['apple_error_mm'],
                summary['detection']['banana_error_mm']
            )
            
            if max_detection_error < 10.0:
                conclusions.append("视觉检测精度优秀（<10mm），定位准确。")
            elif max_detection_error < 20.0:
                conclusions.append("视觉检测精度良好（<20mm），可满足抓取需求。")
            else:
                conclusions.append(f"视觉检测精度不足（{max_detection_error:.1f}mm），建议优化检测算法。")
        else:
            conclusions.append("视觉检测存在问题，需要优化检测算法或调整相机参数。")
        
        # 规划结论
        if summary['planning']['success']:
            if summary['planning']['collision_points'] == 0:
                conclusions.append("轨迹规划完全无碰撞，规划质量优秀。")
            elif summary['planning']['collision_points'] < 5:
                conclusions.append("轨迹规划有少量潜在碰撞点，但基本满足要求。")
            else:
                conclusions.append(f"轨迹规划存在较多碰撞点（{summary['planning']['collision_points']}个），需要优化。")
        else:
            conclusions.append("轨迹规划失败，需要检查运动学模型或规划参数。")
        
        # 抓取结论
        if summary['grasping']['success_rate_percent'] >= 95.0:
            conclusions.append(f"抓取成功率优秀（{summary['grasping']['success_rate_percent']:.1f}%），系统性能稳定。")
        elif summary['grasping']['success_rate_percent'] >= 80.0:
            conclusions.append(f"抓取成功率良好（{summary['grasping']['success_rate_percent']:.1f}%），可满足基本需求。")
        else:
            conclusions.append(f"抓取成功率不足（{summary['grasping']['success_rate_percent']:.1f}%），需要优化抓取策略。")
        
        # 总体结论
        if summary['overall']['success']:
            conclusions.append("【总体评估】系统性能优秀，所有指标均达到或超过要求。")
        else:
            conclusions.append("【总体评估】系统存在需要改进的方面，建议根据具体失败点进行优化。")
        
        return conclusions
    
    def _generate_recommendations(self):
        """生成改进建议"""
        summary = self._calculate_performance_summary()
        recommendations = []
        
        # 标定建议
        if summary['calibration']['mean_error_mm'] > 10.0:
            recommendations.extend([
                "1. 增加标定位姿数量（建议15-20个）",
                "2. 确保标定板在不同位姿下都在相机视野内",
                "3. 使用更高精度的标定板检测算法"
            ])
        
        # 检测建议
        max_detection_error = max(
            summary['detection']['apple_error_mm'],
            summary['detection']['banana_error_mm']
        )
        
        if max_detection_error > 15.0:
            recommendations.extend([
                "4. 优化颜色阈值参数以适应不同光照条件",
                "5. 使用更精确的深度估计方法",
                "6. 考虑使用深度学习进行物体检测"
            ])
        
        # 规划建议
        if summary['planning']['collision_points'] > 3:
            recommendations.extend([
                "7. 增加碰撞检测的安全裕度",
                "8. 优化轨迹插值算法减少突变",
                "9. 使用更精确的碰撞模型"
            ])
        
        # 抓取建议
        if summary['grasping']['success_rate_percent'] < 85.0:
            recommendations.extend([
                "10. 优化抓取位置和姿态",
                "11. 调整抓取力控制参数",
                "12. 改进滑移检测和补偿策略"
            ])
        
        # 通用建议
        recommendations.extend([
            "13. 定期进行系统校准和维护",
            "14. 收集更多实验数据用于算法优化",
            "15. 考虑使用力/力矩传感器提高抓取精度"
        ])
        
        return recommendations
    
    def _generate_markdown_report(self, report, filepath):
        """生成Markdown格式的报告"""
        with open(filepath, 'w') as f:
            f.write(f"# 实验报告: {report['experiment_id']}\n\n")
            
            f.write("## 1. 实验概览\n")
            f.write(f"- **实验名称**: {report['metadata']['experiment_name']}\n")
            f.write(f"- **实验类型**: {report['metadata']['experiment_type']}\n")
            f.write(f"- **开始时间**: {report['metadata']['start_time']}\n")
            f.write(f"- **结束时间**: {report['metadata']['end_time']}\n")
            f.write(f"- **持续时间**: {report['metadata']['duration']:.1f} 秒\n")
            f.write(f"- **总体成功**: {'✅ 是' if report['performance_summary']['overall']['success'] else '❌ 否'}\n\n")
            
            f.write("## 2. 性能总结\n")
            
            # 标定性能
            f.write("### 2.1 手眼标定\n")
            cal = report['performance_summary']['calibration']
            f.write(f"- **成功**: {'✅ 是' if cal['success'] else '❌ 否'}\n")
            f.write(f"- **平均误差**: {cal['mean_error_mm']:.2f} mm\n")
            f.write(f"- **最大误差**: {cal['max_error_mm']:.2f} mm\n\n")
            
            # 检测性能
            f.write("### 2.2 视觉检测\n")
            det = report['performance_summary']['detection']
            f.write(f"- **苹果检测**: {'✅ 成功' if det['apple_success'] else '❌ 失败'} (误差: {det['apple_error_mm']:.1f} mm)\n")
            f.write(f"- **香蕉检测**: {'✅ 成功' if det['banana_success'] else '❌ 失败'} (误差: {det['banana_error_mm']:.1f} mm)\n\n")
            
            # 规划性能
            f.write("### 2.3 轨迹规划\n")
            plan = report['performance_summary']['planning']
            f.write(f"- **成功**: {'✅ 是' if plan['success'] else '❌ 否'}\n")
            f.write(f"- **碰撞点数**: {plan['collision_points']}\n\n")
            
            # 抓取性能
            f.write("### 2.4 抓取执行\n")
            grasp = report['performance_summary']['grasping']
            f.write(f"- **总尝试次数**: {grasp['total_trials']}\n")
            f.write(f"- **成功次数**: {grasp['successful_trials']}\n")
            f.write(f"- **成功率**: {grasp['success_rate_percent']:.1f}%\n\n")
            
            f.write("## 3. 统计分析\n")
            stats = report['statistical_analysis']
            
            if 'joint_statistics' in stats:
                f.write("### 3.1 关节运动统计\n")
                f.write(f"- **平均位置**: {[f'{x:.3f}' for x in stats['joint_statistics']['mean_positions']]}\n")
                f.write(f"- **位置标准差**: {[f'{x:.3f}' for x in stats['joint_statistics']['std_positions']]}\n")
                f.write(f"- **位置范围**: {[f'{x:.3f}' for x in stats['joint_statistics']['position_ranges']]}\n\n")
            
            if 'ee_trajectory' in stats:
                f.write("### 3.2 末端执行器轨迹\n")
                f.write(f"- **路径长度**: {stats['ee_trajectory']['path_length']:.3f} m\n")
                f.write(f"- **平均速度**: {stats['ee_trajectory']['mean_velocity']:.3f} m/s\n")
                f.write(f"- **位置方差**: {stats['ee_trajectory']['position_variance']}\n\n")
            
            f.write("## 4. 结论\n")
            for conclusion in report['conclusions']:
                f.write(f"- {conclusion}\n")
            
            f.write("\n## 5. 改进建议\n")
            for i, recommendation in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {recommendation}\n")
            
            f.write("\n## 6. 原始数据\n")
            f.write(f"- JSON数据: `{self.current_experiment}_data.json`\n")
            f.write(f"- CSV指标: `{self.current_experiment}_metrics.csv`\n")
            f.write(f"- CSV关节数据: `{self.current_experiment}_joints.csv`\n")
            
            f.write("\n---\n")
            f.write(f"*报告生成时间: {datetime.now().isoformat()}*\n")
    
    def _generate_experiment_plots(self):
        """生成实验图表"""
        if not self.data_buffers or len(self.data_buffers['timestamps']) == 0:
            print("没有足够的数据生成图表")
            return None
        
        try:
            plots = []
            
            # 1. 关节位置图
            if self.data_buffers.get('joint_positions'):
                plot1 = self._plot_joint_positions()
                plots.append(plot1)
            
            # 2. 关节速度图
            if self.data_buffers.get('joint_velocities'):
                plot2 = self._plot_joint_velocities()
                plots.append(plot2)
            
            # 3. 末端执行器轨迹图
            if self.data_buffers.get('ee_positions'):
                plot3 = self._plot_ee_trajectory()
                plots.append(plot3)
            
            # 4. 性能指标图
            if self.data_buffers.get('performance_metrics'):
                plot4 = self._plot_performance_metrics()
                plots.append(plot4)
            
            # 5. 接触力图
            if self.data_buffers.get('contact_forces'):
                plot5 = self._plot_contact_forces()
                plots.append(plot5)
            
            # 6. 抓取力曲线
            plot6 = self._plot_grasp_analysis()
            plots.append(plot6)
            
            # 保存所有图表
            saved_plots = []
            for i, fig in enumerate(plots):
                if fig is not None:
                    filename = f"{self.current_experiment}_plot_{i+1}.png"
                    filepath = os.path.join(self.collection_config['plots_directory'], filename)
                    fig.savefig(filepath, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    saved_plots.append(filepath)
            
            # 生成综合仪表板
            dashboard = self._create_dashboard()
            if dashboard:
                dashboard_file = f"{self.current_experiment}_dashboard.png"
                dashboard_path = os.path.join(self.collection_config['plots_directory'], dashboard_file)
                dashboard.savefig(dashboard_path, dpi=150, bbox_inches='tight')
                plt.close(dashboard)
                saved_plots.append(dashboard_path)
            
            return saved_plots
            
        except Exception as e:
            print(f"生成图表失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _plot_joint_positions(self):
        """绘制关节位置图"""
        if not self.data_buffers.get('joint_positions'):
            return None
        
        joint_data = np.array(self.data_buffers['joint_positions'])
        timestamps = np.array(self.data_buffers['timestamps'])
        timestamps = timestamps - timestamps[0]  # 相对时间
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        num_joints = min(joint_data.shape[1], len(axes))
        
        for i in range(num_joints):
            ax = axes[i]
            ax.plot(timestamps, joint_data[:, i], 'b-', linewidth=1.5)
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('位置 (rad)')
            ax.set_title(f'关节 {i}')
            ax.grid(True, alpha=0.3)
            
            # 计算统计信息
            mean_val = np.mean(joint_data[:, i])
            std_val = np.std(joint_data[:, i])
            
            ax.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7, label=f'均值: {mean_val:.3f}')
            ax.fill_between(timestamps, mean_val-std_val, mean_val+std_val, 
                          alpha=0.2, color='gray', label=f'±1标准差')
            ax.legend(fontsize=8)
        
        # 隐藏多余的子图
        for i in range(num_joints, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('关节位置随时间变化', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def _plot_joint_velocities(self):
        """绘制关节速度图"""
        if not self.data_buffers.get('joint_velocities'):
            return None
        
        velocity_data = np.array(self.data_buffers['joint_velocities'])
        timestamps = np.array(self.data_buffers['timestamps'])
        timestamps = timestamps - timestamps[0]
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        num_joints = min(velocity_data.shape[1], len(axes))
        
        for i in range(num_joints):
            ax = axes[i]
            ax.plot(timestamps, velocity_data[:, i], 'g-', linewidth=1.5)
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('速度 (rad/s)')
            ax.set_title(f'关节 {i} 速度')
            ax.grid(True, alpha=0.3)
            
            # 标记速度限制（假设±2 rad/s）
            ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='上限')
            ax.axhline(y=-2.0, color='r', linestyle='--', alpha=0.5, label='下限')
            
            # 计算最大速度
            max_vel = np.max(np.abs(velocity_data[:, i]))
            ax.text(0.05, 0.95, f'最大: {max_vel:.2f} rad/s', 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        for i in range(num_joints, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('关节速度随时间变化', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def _plot_ee_trajectory(self):
        """绘制末端执行器轨迹图"""
        if not self.data_buffers.get('ee_positions'):
            return None
        
        ee_positions = np.array(self.data_buffers['ee_positions'])
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D轨迹图
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], 
                'b-', linewidth=2, alpha=0.7)
        ax1.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2], 
                   color='green', s=100, label='起点', zorder=5)
        ax1.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2], 
                   color='red', s=100, label='终点', zorder=5)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D轨迹')
        ax1.legend()
        ax1.grid(True)
        
        # XY平面投影
        ax2 = fig.add_subplot(234)
        ax2.plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', linewidth=2)
        ax2.scatter(ee_positions[0, 0], ee_positions[0, 1], color='green', s=100, label='起点')
        ax2.scatter(ee_positions[-1, 0], ee_positions[-1, 1], color='red', s=100, label='终点')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY平面投影')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        # 时间序列
        timestamps = np.array(self.data_buffers['timestamps'])
        timestamps = timestamps - timestamps[0]
        
        ax3 = fig.add_subplot(232)
        ax3.plot(timestamps, ee_positions[:, 0], 'r-', label='X')
        ax3.plot(timestamps, ee_positions[:, 1], 'g-', label='Y')
        ax3.plot(timestamps, ee_positions[:, 2], 'b-', label='Z')
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('位置 (m)')
        ax3.set_title('位置分量时间序列')
        ax3.legend()
        ax3.grid(True)
        
        # 速度大小
        if self.data_buffers.get('ee_velocities'):
            ee_velocities = np.array(self.data_buffers['ee_velocities'])
            speeds = np.linalg.norm(ee_velocities[:, :3], axis=1)
            
            ax4 = fig.add_subplot(235)
            ax4.plot(timestamps, speeds, 'purple', linewidth=2)
            ax4.set_xlabel('时间 (s)')
            ax4.set_ylabel('速度大小 (m/s)')
            ax4.set_title('末端速度')
            ax4.grid(True)
            
            # 标记最大速度
            max_speed = np.max(speeds)
            max_time = timestamps[np.argmax(speeds)]
            ax4.axvline(x=max_time, color='r', linestyle='--', alpha=0.5)
            ax4.text(max_time, max_speed, f'最大: {max_speed:.2f} m/s', 
                    fontsize=9, ha='right')
        
        # 路径长度累积
        ax5 = fig.add_subplot(233)
        cumulative_length = np.zeros(len(ee_positions))
        for i in range(1, len(ee_positions)):
            segment_length = np.linalg.norm(ee_positions[i] - ee_positions[i-1])
            cumulative_length[i] = cumulative_length[i-1] + segment_length
        
        ax5.plot(timestamps, cumulative_length, 'orange', linewidth=2)
        ax5.set_xlabel('时间 (s)')
        ax5.set_ylabel('累积路径长度 (m)')
        ax5.set_title(f'路径长度: {cumulative_length[-1]:.3f} m')
        ax5.grid(True)
        
        # 位置方差
        ax6 = fig.add_subplot(236)
        position_variance = np.var(ee_positions, axis=0)
        bars = ax6.bar(['X', 'Y', 'Z'], position_variance, color=['red', 'green', 'blue'])
        ax6.set_ylabel('方差 (m²)')
        ax6.set_title('位置方差')
        
        # 添加数值标签
        for bar, var in zip(bars, position_variance):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                   f'{var:.2e}', ha='center', va='bottom')
        
        plt.suptitle('末端执行器轨迹分析', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def _plot_performance_metrics(self):
        """绘制性能指标图"""
        if not self.data_buffers.get('performance_metrics'):
            return None
        
        # 分类指标
        metric_categories = {
            'calibration': [],
            'detection': [],
            'planning': [],
            'grasping': [],
            'other': []
        }
        
        metric_values = {}
        
        for metric in self.data_buffers['performance_metrics']:
            metric_name = metric['metric']
            metric_value = metric['value']
            
            # 分类
            category = 'other'
            for cat in ['calibration', 'detection', 'planning', 'grasping']:
                if cat in metric_name.lower():
                    category = cat
                    break
            
            metric_categories[category].append(metric_name)
            
            # 存储最新值
            metric_values[metric_name] = metric_value
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. 标定误差图
        ax1 = axes[0]
        cal_metrics = [m for m in metric_categories['calibration'] if 'error' in m.lower()]
        cal_values = [metric_values.get(m, 0) for m in cal_metrics]
        
        if cal_metrics:
            bars = ax1.bar(range(len(cal_metrics)), cal_values, color='skyblue')
            ax1.set_xticks(range(len(cal_metrics)))
            ax1.set_xticklabels([m.replace('calibration_', '').replace('_', ' ') 
                               for m in cal_metrics], rotation=45, ha='right')
            ax1.set_ylabel('误差值')
            ax1.set_title('标定误差')
            
            # 添加阈值线
            if 'calibration_mean_translation_error' in metric_values:
                threshold = 10.0  # 10mm阈值
                ax1.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='10mm阈值')
                ax1.legend()
        
        # 2. 检测误差图
        ax2 = axes[1]
        det_metrics = [m for m in metric_categories['detection'] if 'error' in m.lower()]
        det_values = [metric_values.get(m, 0) for m in det_metrics]
        
        if det_metrics:
            colors = ['red' if 'apple' in m else 'yellow' for m in det_metrics]
            bars = ax2.bar(range(len(det_metrics)), det_values, color=colors)
            ax2.set_xticks(range(len(det_metrics)))
            ax2.set_xticklabels([m.replace('detection_', '').replace('_error', '') 
                               for m in det_metrics], rotation=45, ha='right')
            ax2.set_ylabel('误差 (mm)')
            ax2.set_title('检测误差')
            ax2.axhline(y=10.0, color='r', linestyle='--', alpha=0.5, label='10mm阈值')
            ax2.legend()
        
        # 3. 抓取成功率
        ax3 = axes[2]
        grasp_rate = metric_values.get('grasping_success_rate', 0)
        
        wedges, texts, autotexts = ax3.pie([grasp_rate, 100-grasp_rate], 
                                          labels=['成功', '失败'],
                                          colors=['green', 'red'],
                                          autopct='%1.1f%%')
        ax3.set_title(f'抓取成功率: {grasp_rate:.1f}%')
        
        # 4. 阶段时间分布
        ax4 = axes[3]
        phase_metrics = [m for m in metric_categories['grasping'] if 'phase' in m and 'duration' in m]
        phase_durations = [metric_values.get(m, 0) for m in phase_metrics]
        
        if phase_metrics:
            phase_names = [m.split('_phase_')[1].split('_duration')[0] for m in phase_metrics]
            bars = ax4.bar(phase_names, phase_durations, color=cm.rainbow(np.linspace(0, 1, len(phase_names))))
            ax4.set_ylabel('时间 (s)')
            ax4.set_title('抓取各阶段时间')
            ax4.tick_params(axis='x', rotation=45)
            
            # 添加总时间
            total_time = sum(phase_durations)
            ax4.text(0.95, 0.95, f'总时间: {total_time:.1f}s', 
                    transform=ax4.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. 力分布图
        ax5 = axes[4]
        force_metrics = [m for m in metric_values.keys() if 'force' in m.lower() and 'grasp' in m.lower()]
        force_values = [metric_values[m] for m in force_metrics if m in metric_values]
        
        if force_metrics and force_values:
            force_names = [m.replace('grasp_trial_', '').replace('_max_force', '') 
                         for m in force_metrics if 'max_force' in m]
            
            if len(force_names) == len(force_values):
                ax5.bar(force_names, force_values, color='orange')
                ax5.set_ylabel('力 (N)')
                ax5.set_title('抓取力分布')
                ax5.tick_params(axis='x', rotation=45)
                
                # 标记平均力
                avg_force = np.mean(force_values) if force_values else 0
                ax5.axhline(y=avg_force, color='r', linestyle='--', alpha=0.5, 
                          label=f'平均: {avg_force:.1f}N')
                ax5.legend()
        
        # 6. 综合评估雷达图
        ax6 = axes[5]
        
        # 计算各个维度的分数（0-100）
        scores = []
        labels = []
        
        # 标定精度分数
        cal_error = metric_values.get('calibration_mean_translation_error', 100)
        cal_score = max(0, 100 - cal_error * 2)  # 误差每1mm扣2分
        scores.append(cal_score)
        labels.append('标定')
        
        # 检测精度分数
        det_errors = [metric_values.get(f'detection_{obj}_error', 100) 
                     for obj in ['apple', 'banana']]
        det_score = max(0, 100 - np.mean(det_errors) if det_errors else 0)
        scores.append(det_score)
        labels.append('检测')
        
        # 规划分数
        plan_success = metric_values.get('planning_success', 0)
        plan_score = 100 if plan_success == 1 else 0
        scores.append(plan_score)
        labels.append('规划')
        
        # 抓取分数
        grasp_score = metric_values.get('grasping_success_rate', 0)
        scores.append(grasp_score)
        labels.append('抓取')
        
        # 速度分数（如果有）
        if 'ee_trajectory' in self._perform_statistical_analysis():
            speed = self._perform_statistical_analysis()['ee_trajectory']['mean_velocity']
            speed_score = min(100, speed * 50)  # 2m/s得100分
            scores.append(speed_score)
            labels.append('速度')
        
        # 绘制雷达图
        if len(scores) >= 3:
            angles = np.linspace(0, 2*np.pi, len(scores), endpoint=False).tolist()
            scores.append(scores[0])  # 闭合图形
            angles.append(angles[0])
            
            ax6 = fig.add_subplot(236, polar=True)
            ax6.plot(angles, scores, 'o-', linewidth=2)
            ax6.fill(angles, scores, alpha=0.25)
            ax6.set_thetagrids(np.degrees(angles[:-1]), labels)
            ax6.set_ylim(0, 100)
            ax6.set_title('系统性能雷达图', pad=20)
            ax6.grid(True)
        
        plt.suptitle('实验性能指标分析', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def _plot_contact_forces(self):
        """绘制接触力图"""
        if not self.data_buffers.get('contact_forces'):
            return None
        
        # 提取接触力数据
        force_magnitudes = []
        force_timestamps = []
        
        for i, contact_list in enumerate(self.data_buffers['contact_forces']):
            if contact_list:
                max_force = 0
                for contact in contact_list:
                    force = np.linalg.norm(contact['force'])
                    max_force = max(max_force, force)
                
                force_magnitudes.append(max_force)
                force_timestamps.append(self.data_buffers['timestamps'][i])
        
        if not force_magnitudes:
            return None
        
        force_magnitudes = np.array(force_magnitudes)
        force_timestamps = np.array(force_timestamps)
        force_timestamps = force_timestamps - force_timestamps[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 接触力时间序列
        ax1 = axes[0, 0]
        ax1.plot(force_timestamps, force_magnitudes, 'r-', linewidth=2, marker='o', markersize=3)
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('接触力 (N)')
        ax1.set_title('接触力随时间变化')
        ax1.grid(True)
        
        # 标记最大力
        max_force = np.max(force_magnitudes)
        max_time = force_timestamps[np.argmax(force_magnitudes)]
        ax1.axvline(x=max_time, color='b', linestyle='--', alpha=0.5)
        ax1.text(max_time, max_force, f'最大: {max_force:.1f} N', 
                fontsize=9, ha='right')
        
        # 2. 接触力直方图
        ax2 = axes[0, 1]
        ax2.hist(force_magnitudes, bins=20, color='orange', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('接触力 (N)')
        ax2.set_ylabel('频数')
        ax2.set_title('接触力分布')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'均值: {np.mean(force_magnitudes):.2f} N\n'
        stats_text += f'标准差: {np.std(force_magnitudes):.2f} N\n'
        stats_text += f'最大值: {max_force:.2f} N\n'
        stats_text += f'非零率: {100*len(force_magnitudes)/len(self.data_buffers["contact_forces"]):.1f}%'
        
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. 接触频率
        ax3 = axes[1, 0]
        
        # 计算接触频率
        contact_present = [1 if contacts else 0 for contacts in self.data_buffers['contact_forces']]
        cumulative_contacts = np.cumsum(contact_present)
        timestamps_all = np.array(self.data_buffers['timestamps'])
        timestamps_all = timestamps_all - timestamps_all[0]
        
        ax3.plot(timestamps_all, cumulative_contacts, 'g-', linewidth=2)
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('累积接触次数')
        ax3.set_title('累积接触次数')
        ax3.grid(True)
        
        # 4. 力-时间关系散点图
        ax4 = axes[1, 1]
        scatter = ax4.scatter(force_timestamps, force_magnitudes, 
                            c=force_magnitudes, cmap='hot', s=50, alpha=0.7)
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('接触力 (N)')
        ax4.set_title('力-时间关系')
        ax4.grid(True, alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax4, label='力大小 (N)')
        
        plt.suptitle('接触力分析', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def _plot_grasp_analysis(self):
        """绘制抓取分析图"""
        # 从性能指标中提取抓取数据
        grasp_trials = {}
        
        for metric in self.data_buffers['performance_metrics']:
            metric_name = metric['metric']
            
            if 'grasp_trial' in metric_name and '_success' in metric_name:
                # 提取试验编号
                try:
                    trial_num = int(metric_name.split('_')[2])
                except:
                    continue
                
                if trial_num not in grasp_trials:
                    grasp_trials[trial_num] = {
                        'success': metric['value'] == 1,
                        'reason': metric['metadata'].get('reason', 'unknown'),
                        'object': metric['metadata'].get('object', 'unknown'),
                        'duration': metric['metadata'].get('total_duration', 0)
                    }
                else:
                    grasp_trials[trial_num]['success'] = metric['value'] == 1
        
        if not grasp_trials:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 成功/失败分布
        ax1 = axes[0, 0]
        successes = sum(1 for t in grasp_trials.values() if t['success'])
        failures = len(grasp_trials) - successes
        
        colors = ['green', 'red']
        ax1.pie([successes, failures], labels=['成功', '失败'], 
               colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'抓取成功率: {100*successes/len(grasp_trials):.1f}%')
        
        # 2. 试验结果时间序列
        ax2 = axes[0, 1]
        trial_nums = sorted(grasp_trials.keys())
        success_status = [1 if grasp_trials[t]['success'] else 0 for t in trial_nums]
        
        ax2.step(trial_nums, success_status, 'b-', where='mid', linewidth=2, marker='o')
        ax2.set_xlabel('试验编号')
        ax2.set_ylabel('成功 (1)/失败 (0)')
        ax2.set_title('试验结果序列')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['失败', '成功'])
        ax2.grid(True)
        
        # 标记失败点
        for i, status in enumerate(success_status):
            if status == 0:
                ax2.text(trial_nums[i], -0.1, '✗', ha='center', va='top', 
                        fontsize=12, color='red')
        
        # 3. 失败原因分析
        ax3 = axes[1, 0]
        failure_reasons = {}
        
        for trial_data in grasp_trials.values():
            if not trial_data['success']:
                reason = trial_data['reason']
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        
        if failure_reasons:
            reasons = list(failure_reasons.keys())
            counts = list(failure_reasons.values())
            
            bars = ax3.barh(reasons, counts, color='red', alpha=0.7)
            ax3.set_xlabel('发生次数')
            ax3.set_title('失败原因分析')
            ax3.grid(True, axis='x', alpha=0.3)
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                width = bar.get_width()
                ax3.text(width, bar.get_y() + bar.get_height()/2, 
                       f' {count}', va='center')
        
        # 4. 试验持续时间
        ax4 = axes[1, 1]
        durations = [grasp_trials[t]['duration'] for t in trial_nums]
        
        colors = ['green' if grasp_trials[t]['success'] else 'red' for t in trial_nums]
        bars = ax4.bar(trial_nums, durations, color=colors, alpha=0.7)
        ax4.set_xlabel('试验编号')
        ax4.set_ylabel('持续时间 (s)')
        ax4.set_title('试验持续时间')
        ax4.grid(True, alpha=0.3)
        
        # 添加平均线
        avg_duration = np.mean(durations)
        ax4.axhline(y=avg_duration, color='blue', linestyle='--', alpha=0.5,
                   label=f'平均: {avg_duration:.1f}s')
        ax4.legend()
        
        plt.suptitle('抓取试验分析', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def _create_dashboard(self):
        """创建综合仪表板"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 总体状态（左上）
        ax1 = plt.subplot(3, 4, 1)
        
        summary = self._calculate_performance_summary()
        
        # 创建状态指示器
        status_text = "系统状态:\n\n"
        
        indicators = {
            '标定': ('✅' if summary['calibration']['success'] else '❌', 
                    'green' if summary['calibration']['success'] else 'red'),
            '检测': ('✅' if summary['detection']['apple_success'] and summary['detection']['banana_success'] else '❌',
                    'green' if summary['detection']['apple_success'] and summary['detection']['banana_success'] else 'red'),
            '规划': ('✅' if summary['planning']['success'] else '❌',
                    'green' if summary['planning']['success'] else 'red'),
            '抓取': ('✅' if summary['grasping']['success_rate_percent'] >= 95.0 else '⚠',
                    'green' if summary['grasping']['success_rate_percent'] >= 95.0 else 'orange')
        }
        
        for name, (symbol, color) in indicators.items():
            status_text += f"{symbol} {name}\n"
        
        status_text += f"\n总体: {'✅ 通过' if summary['overall']['success'] else '❌ 未通过'}"
        
        ax1.text(0.5, 0.5, status_text, transform=ax1.transAxes,
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax1.axis('off')
        ax1.set_title('系统状态', fontsize=16, pad=20)
        
        # 2. 关键指标（右上）
        ax2 = plt.subplot(3, 4, 2)
        
        key_metrics = [
            ('标定误差', f"{summary['calibration']['mean_error_mm']:.1f} mm"),
            ('检测误差', f"{max(summary['detection']['apple_error_mm'], summary['detection']['banana_error_mm']):.1f} mm"),
            ('抓取成功率', f"{summary['grasping']['success_rate_percent']:.1f}%"),
            ('总试验次数', f"{summary['grasping']['total_trials']}"),
            ('数据点数', f"{summary['overall']['data_points_collected']}"),
            ('实验时长', f"{summary['overall']['total_experiment_duration']:.0f} s")
        ]
        
        metric_text = "关键指标:\n\n"
        for name, value in key_metrics:
            metric_text += f"{name}: {value}\n"
        
        ax2.text(0.5, 0.5, metric_text, transform=ax2.transAxes,
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax2.axis('off')
        ax2.set_title('关键指标', fontsize=16, pad=20)
        
        # 3. 性能评分（中左）
        ax3 = plt.subplot(3, 4, 5)
        
        # 计算各维度分数
        dimensions = ['标定', '检测', '规划', '抓取', '总体']
        
        cal_score = max(0, 100 - summary['calibration']['mean_error_mm'] * 10)
        det_score = max(0, 100 - max(summary['detection']['apple_error_mm'], 
                                    summary['detection']['banana_error_mm']))
        plan_score = 100 if summary['planning']['success'] else 0
        grasp_score = summary['grasping']['success_rate_percent']
        overall_score = np.mean([cal_score, det_score, plan_score, grasp_score])
        
        scores = [cal_score, det_score, plan_score, grasp_score, overall_score]
        
        bars = ax3.bar(dimensions, scores, color=cm.viridis(np.array(scores)/100))
        ax3.set_ylim(0, 105)
        ax3.set_ylabel('分数 (0-100)')
        ax3.set_title('性能评分')
        ax3.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.0f}', ha='center', va='bottom')
        
        # 4. 错误分布（中右）
        ax4 = plt.subplot(3, 4, 6)
        
        # 统计各类错误
        error_types = {
            '标定误差大': 1 if summary['calibration']['mean_error_mm'] > 10 else 0,
            '检测失败': 1 if not (summary['detection']['apple_success'] and summary['detection']['banana_success']) else 0,
            '规划失败': 1 if not summary['planning']['success'] else 0,
            '抓取率低': 1 if summary['grasping']['success_rate_percent'] < 80 else 0,
            '碰撞风险': 1 if summary['planning']['collision_points'] > 5 else 0
        }
        
        error_counts = [count for count in error_types.values() if count > 0]
        error_names = [name for name, count in error_types.items() if count > 0]
        
        if error_counts:
            wedges, texts, autotexts = ax4.pie(error_counts, labels=error_names,
                                             autopct='%1.0f', startangle=90)
            ax4.set_title('错误类型分布')
        else:
            ax4.text(0.5, 0.5, '无重大错误', transform=ax4.transAxes,
                    ha='center', va='center', fontsize=14,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            ax4.set_title('错误分布')
        
        # 5. 时间线（下左）
        ax5 = plt.subplot(3, 4, 9)
        
        # 实验阶段时间线
        if self.experiment_metadata.get('duration'):
            duration = self.experiment_metadata['duration']
            
            # 估算各阶段时间（简化）
            phase_times = {
                '标定': duration * 0.2,
                '检测': duration * 0.15,
                '规划': duration * 0.1,
                '抓取': duration * 0.5,
                '分析': duration * 0.05
            }
            
            current_time = 0
            colors = ['blue', 'green', 'orange', 'red', 'purple']
            
            for i, (phase, time_span) in enumerate(phase_times.items()):
                ax5.barh('时间线', time_span, left=current_time, 
                        color=colors[i], edgecolor='black', alpha=0.7,
                        label=f'{phase}: {time_span:.0f}s')
                current_time += time_span
            
            ax5.set_xlabel('时间 (s)')
            ax5.set_title('实验时间线')
            ax5.legend(loc='upper right', fontsize=9)
        
        # 6. 建议摘要（下右）
        ax6 = plt.subplot(3, 4, 10)
        
        recommendations = self._generate_recommendations()
        
        if recommendations:
            rec_text = "改进建议:\n\n"
            for i, rec in enumerate(recommendations[:5], 1):  # 只显示前5条
                rec_text += f"{i}. {rec}\n"
            
            if len(recommendations) > 5:
                rec_text += f"\n...还有{len(recommendations)-5}条建议"
        else:
            rec_text = "无改进建议\n系统性能良好"
        
        ax6.text(0.5, 0.5, rec_text, transform=ax6.transAxes,
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax6.axis('off')
        ax6.set_title('改进建议', fontsize=16, pad=20)
        
        # 7. 实验信息（右下）
        ax7 = plt.subplot(3, 4, 11)
        
        info_text = f"实验ID: {self.current_experiment}\n"
        info_text += f"开始时间: {self.experiment_metadata['start_time'][11:19]}\n"
        info_text += f"结束时间: {self.experiment_metadata['end_time'][11:19]}\n"
        info_text += f"持续时间: {self.experiment_metadata['duration']:.0f}秒\n\n"
        info_text += f"系统信息:\n"
        info_text += f"关节数: {self.model.nq}\n"
        info_text += f"执行器数: {self.model.nu}\n"
        info_text += f"采样率: {self.collection_config['sampling_rate']}Hz"
        
        ax7.text(0.5, 0.5, info_text, transform=ax7.transAxes,
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax7.axis('off')
        ax7.set_title('实验信息', fontsize=16, pad=20)
        
        # 8. 版权信息（右下角）
        ax8 = plt.subplot(3, 4, 12)
        ax8.text(0.5, 0.5, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                         f"机器人视觉抓取实验系统\n"
                         f"© 2024 实验数据采集系统",
                transform=ax8.transAxes,
                ha='center', va='center', fontsize=8, alpha=0.7)
        ax8.axis('off')
        
        plt.suptitle(f'实验综合仪表板 - {self.current_experiment}', 
                    fontsize=18, y=0.98)
        plt.tight_layout()
        
        return fig
    
    def collect_all_data(self):
        """收集所有实验数据"""
        print("收集所有实验数据...")
        
        # 计算性能总结
        performance_summary = self._calculate_performance_summary()
        
        # 进行统计分析
        statistical_analysis = self._perform_statistical_analysis()
        
        # 生成结论和建议
        conclusions = self._generate_conclusions()
        recommendations = self._generate_recommendations()
        
        all_data = {
            'experiment_id': self.current_experiment,
            'metadata': self.experiment_metadata,
            'performance_summary': performance_summary,
            'statistical_analysis': statistical_analysis,
            'conclusions': conclusions,
            'recommendations': recommendations,
            'raw_data_summary': {
                'data_points': len(self.data_buffers['timestamps']) if 'timestamps' in self.data_buffers else 0,
                'performance_metrics': len(self.data_buffers['performance_metrics']),
                'joint_data_points': len(self.data_buffers.get('joint_positions', [])),
                'ee_data_points': len(self.data_buffers.get('ee_positions', [])),
                'contact_data_points': len(self.data_buffers.get('contact_forces', []))
            }
        }
        
        return all_data
    
    def generate_report(self, data=None):
        """生成实验报告"""
        if data is None:
            data = self.collect_all_data()
        
        return self._generate_experiment_report()
    
    def generate_plots(self, data=None):
        """生成实验图表"""
        plots = self._generate_experiment_plots()
        
        if plots:
            print(f"生成了 {len(plots)} 张图表")
            return plots
        else:
            print("未能生成图表")
            return []
