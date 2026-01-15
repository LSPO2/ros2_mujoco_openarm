"""
open_robot.py - 集成实验功能的完整机器人控制与实验系统
集成了：手眼标定、视觉检测、轨迹规划、抓取执行、数据采集
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import threading
import json
import pickle
import numpy as np
import cv2
from datetime import datetime

# 设置工作目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
print(f"工作目录: {os.getcwd()}")

# 设置MuJoCo环境
os.environ['MUJOCO_GL'] = 'glfw'
import mujoco
import mujoco.viewer

# 导入实验模块
sys.path.append(SCRIPT_DIR)
try:
    from handeye_calibration import HandEyeCalibrator
    from vision_detection import VisionDetector
    from grasp_planning import GraspPlanner
    from grasp_execution import GraspExecutor
    from experiment_data import ExperimentDataCollector
    print("实验模块加载成功")
except ImportError as e:
    print(f"警告: 实验模块导入失败 - {e}")
    print("请确保所有模块文件在同一目录")

# ====================== 全局配置 ======================
class Config:
    """全局配置参数"""
    # 文件路径
    MODEL_PATH = 'openarm_bimanual.mjcf.xml'
    CALIBRATION_PATH = 'calibration_matrix.npy'
    RESULTS_DIR = 'experiment_results'
    VIDEOS_DIR = 'videos'
    
    # 相机配置
    CAMERA_ID = 2  # depth_estimation_camera
    CAMERA_NAME = 'depth_estimation_camera'
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FOVY = 90.0  # 度
    
    # 物体配置
    APPLE_POS = np.array([0.2, -0.4, 0.45])  # 世界坐标系
    BANANA_POS = np.array([-0.2, -0.4, 0.45])
    APPLE_SIZE = 0.04  # 半径
    BANANA_SIZE = 0.02  # 厚度
    
    # 标定配置
    CALIBRATION_POSES = 12  # 标定位姿数量
    CALIBRATION_ERROR_THRESHOLD = 0.005  # 5mm
    
    # 视觉检测配置
    DETECTION_ERROR_THRESHOLD = 0.01  # 10mm
    
    # 抓取配置
    GRASP_SUCCESS_RATE_TARGET = 0.95  # 95%成功率
    NUM_GRASP_TRIALS = 10  # 抓取实验次数
    
    # 实验配置
    EXPERIMENT_NAME = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# 确保目录存在
os.makedirs(Config.RESULTS_DIR, exist_ok=True)
os.makedirs(Config.VIDEOS_DIR, exist_ok=True)

# ====================== 核心机器人控制器 ======================
class RobotController:
    """机器人核心控制器"""
    
    def __init__(self, model_path=Config.MODEL_PATH):
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        
        print(f"模型加载完成 - 关节: {self.model.nq}, 执行器: {self.model.nu}")
        
        # 获取执行器控制的关节ID
        self.actuated_joint_ids = []
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            self.actuated_joint_ids.append(joint_id)
        
        self.n_controlled = len(self.actuated_joint_ids)
        self.target_positions = np.zeros(self.n_controlled)
        
        # 初始化目标位置
        self._update_current()
        self.target_positions = self.current_positions.copy()
        
        # PD参数
        self.kp = np.full(self.n_controlled, 10000.0)
        self.kd = np.full(self.n_controlled, 80.0)
        
        # 状态标志
        self.is_connected = True
        self.last_update_time = time.time()
        
        # 创建渲染器（用于视觉）
        self.renderer = mujoco.Renderer(self.model, Config.CAMERA_HEIGHT, Config.CAMERA_WIDTH)
        
        print(f"控制器初始化完成 - 受控关节: {self.n_controlled}")
    
    def _update_current(self):
        """更新当前关节位置和速度"""
        self.current_positions = np.array([
            self.data.qpos[joint_id] for joint_id in self.actuated_joint_ids
        ])
        self.current_velocities = np.array([
            self.data.qvel[joint_id] for joint_id in self.actuated_joint_ids
        ])
    
    def set_joint_target(self, idx, angle):
        """设置关节目标位置"""
        if 0 <= idx < self.n_controlled:
            # 限制在合理范围内
            angle = max(-np.pi, min(np.pi, angle))
            self.target_positions[idx] = angle
            return angle
        return None
    
    def adjust_joint_target(self, idx, delta):
        """调整关节目标位置"""
        if 0 <= idx < self.n_controlled:
            new_angle = self.target_positions[idx] + delta
            return self.set_joint_target(idx, new_angle)
        return None
    
    def get_joint_target(self, idx):
        """获取关节目标位置"""
        if 0 <= idx < self.n_controlled:
            return self.target_positions[idx]
        return 0.0
    
    def get_joint_current(self, idx):
        """获取关节当前位置"""
        if 0 <= idx < self.n_controlled:
            return self.current_positions[idx]
        return 0.0
    
    def set_all_joints(self, angles):
        """设置所有关节目标位置"""
        for i in range(min(len(angles), self.n_controlled)):
            self.set_joint_target(i, angles[i])
    
    def get_all_joints(self):
        """获取所有关节目标位置"""
        return self.target_positions.copy()
    
    def reset_to_current(self):
        """重置为当前位置"""
        self._update_current()
        self.target_positions = self.current_positions.copy()
    
    def reset_to_zero(self):
        """重置为零位置"""
        self.target_positions = np.zeros(self.n_controlled)
        for i in range(self.n_controlled):
            self.set_joint_target(i, 0.0)
    
    def update_control(self):
        """更新控制信号"""
        if not self.is_connected:
            return np.zeros(self.n_controlled)
        
        self._update_current()
        
        # 计算误差
        errors = self.target_positions - self.current_positions
        
        # PD控制
        torques = self.kp * errors - self.kd * self.current_velocities
        
        # 应用控制
        for i in range(min(self.model.nu, len(torques))):
            self.data.ctrl[i] = torques[i]
        
        self.last_update_time = time.time()
        return torques
    
    def get_body_position(self, body_name):
        """获取指定物体在世界坐标系中的位置"""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                print(f"警告: 未找到物体 '{body_name}'")
                return None
            return self.data.body(body_id).xpos.copy()
        except Exception as e:
            print(f"获取物体位置错误: {e}")
            return None
    
    def get_body_orientation(self, body_name):
        """获取指定物体在世界坐标系中的姿态（四元数）"""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                return None
            return self.data.body(body_id).xquat.copy()
        except:
            return None
    
    def get_camera_image(self):
        """获取相机RGB图像"""
        try:
            self.renderer.update_scene(self.data, camera=Config.CAMERA_ID)
            image = self.renderer.render()
            return image.copy() if image is not None else None
        except Exception as e:
            print(f"获取相机图像错误: {e}")
            return None
    
    def get_camera_depth(self):
        """获取相机深度图像（简化版本）"""
        image = self.get_camera_image()
        if image is None:
            return None
        
        # 使用绿色通道作为伪深度
        depth = image[:, :, 1].astype(np.float32) / 255.0
        
        # 添加简单的距离梯度
        height, width = depth.shape
        y, x = np.mgrid[0:height, 0:width]
        center_x, center_y = width // 2, height // 2
        
        # 创建距离梯度（0.3-1.0范围）
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        gradient = 0.3 + 0.7 * (distance / max_dist)
        
        return gradient
    
    def get_camera_params(self):
        """获取相机内参矩阵"""
        # 根据FOV和分辨率计算内参
        fov_rad = np.radians(Config.CAMERA_FOVY)
        f = Config.CAMERA_HEIGHT / (2 * np.tan(fov_rad / 2))
        
        K = np.array([
            [f, 0, Config.CAMERA_WIDTH / 2],
            [0, f, Config.CAMERA_HEIGHT / 2],
            [0, 0, 1]
        ])
        return K
    
    def move_to_position(self, target_positions, max_time=5.0):
        """移动到指定关节位置（阻塞）"""
        original_targets = self.target_positions.copy()
        start_time = time.time()
        
        # 设置目标
        self.set_all_joints(target_positions)
        
        # 等待到达或超时
        while time.time() - start_time < max_time:
            self._update_current()
            errors = np.abs(self.target_positions - self.current_positions)
            
            if np.all(errors < 0.01):  # 0.01弧度阈值
                print("到达目标位置")
                return True
            
            time.sleep(0.01)
        
        print("移动超时")
        return False
    
    def step_simulation(self):
        """执行一步仿真"""
        self.update_control()
        mujoco.mj_step(self.model, self.data)

# ====================== 实验管理器 ======================
class ExperimentManager:
    """实验流程管理器"""
    
    def __init__(self, robot_controller):
        self.robot = robot_controller
        
        # 初始化实验模块
        self.calibrator = HandEyeCalibrator(self.robot)
        self.detector = VisionDetector(self.robot)
        self.planner = GraspPlanner(self.robot)
        self.executor = GraspExecutor(self.robot)
        self.data_collector = ExperimentDataCollector(self.robot)
        
        # 实验状态
        self.current_experiment = Config.EXPERIMENT_NAME
        self.current_step = 0
        self.is_calibrated = False
        self.calibration_matrix = None
        
        print("实验管理器初始化完成")
    
    def run_handeye_calibration(self, num_poses=12):
        """运行手眼标定"""
        print("=" * 50)
        print("步骤 1: 手眼标定")
        print("=" * 50)
        
        success, matrix, errors = self.calibrator.perform_calibration()
        
        if success:
            self.calibration_matrix = matrix
            self.is_calibrated = True
            self.detector.set_calibration_matrix(matrix)
            
            # 保存标定结果
            np.save(Config.CALIBRATION_PATH, matrix)
            
            print(f"标定成功! 平均误差: {errors['mean_translation']*1000:.2f} mm")
            print(f"最大误差: {errors['max_translation']*1000:.2f} mm")
            print(f"旋转误差: {errors['mean_rotation']:.4f} rad")
            
            if errors['max_translation'] < Config.CALIBRATION_ERROR_THRESHOLD:
                print("✓ 标定精度满足要求 (≤5mm)")
            else:
                print("⚠ 标定精度略高，但可继续")
        else:
            print("标定失败，请重试")
        
        return success
    
    def run_vision_detection(self):
        """运行视觉检测与定位"""
        if not self.is_calibrated:
            print("请先完成手眼标定")
            return False
        
        print("=" * 50)
        print("步骤 2: 视觉检测与定位")
        print("=" * 50)
        
        # 检测苹果
        print("\n检测苹果...")
        apple_success, apple_pos, apple_error = self.detector.detect_object('apple')
        
        if apple_success:
            print(f"苹果检测位置: {apple_pos}")
            print(f"实际位置: {Config.APPLE_POS}")
            print(f"定位误差: {apple_error*1000:.2f} mm")
            
            if apple_error < Config.DETECTION_ERROR_THRESHOLD:
                print("✓ 苹果定位精度满足要求 (≤10mm)")
            else:
                print("⚠ 苹果定位精度略高")
        else:
            print("苹果检测失败")
        
        # 检测香蕉
        print("\n检测香蕉...")
        banana_success, banana_pos, banana_error = self.detector.detect_object('banana')
        
        if banana_success:
            print(f"香蕉检测位置: {banana_pos}")
            print(f"实际位置: {Config.BANANA_POS}")
            print(f"定位误差: {banana_error*1000:.2f} mm")
            
            if banana_error < Config.DETECTION_ERROR_THRESHOLD:
                print("✓ 香蕉定位精度满足要求 (≤10mm)")
            else:
                print("⚠ 香蕉定位精度略高")
        else:
            print("香蕉检测失败")
        
        return apple_success and banana_success
    
    def run_grasp_planning(self):
        """运行抓取轨迹规划"""
        print("=" * 50)
        print("步骤 3: 抓取轨迹规划")
        print("=" * 50)
        
        # 规划苹果抓取
        print("\n规划苹果抓取轨迹...")
        apple_plan = self.planner.plan_grasp_trajectory('apple')
        
        if apple_plan['success']:
            print(f"苹果抓取轨迹规划成功")
            print(f"轨迹点数: {len(apple_plan['trajectory'])}")
            print(f"规划时间: {apple_plan['planning_time']:.2f}秒")
        else:
            print(f"苹果抓取轨迹规划失败: {apple_plan['reason']}")
        
        # 规划香蕉抓取
        print("\n规划香蕉抓取轨迹...")
        banana_plan = self.planner.plan_grasp_trajectory('banana')
        
        if banana_plan['success']:
            print(f"香蕉抓取轨迹规划成功")
            print(f"轨迹点数: {len(banana_plan['trajectory'])}")
            print(f"规划时间: {banana_plan['planning_time']:.2f}秒")
        else:
            print(f"香蕉抓取轨迹规划失败: {banana_plan['reason']}")
        
        return apple_plan['success'] and banana_plan['success']
    
    def run_grasp_execution(self, num_trials=10):
        """运行抓取执行实验"""
        print("=" * 50)
        print("步骤 4: 抓取执行与控制")
        print("=" * 50)
        
        results = []
        
        for i in range(num_trials):
            print(f"\n实验 {i+1}/{num_trials}")
            
            # 随机选择抓取物体
            target = 'apple' if np.random.random() > 0.5 else 'banana'
            print(f"目标物体: {target}")
            
            # 执行抓取
            success, details = self.executor.execute_grasp(target)
            
            results.append({
                'trial': i+1,
                'target': target,
                'success': success,
                'details': details
            })
            
            print(f"结果: {'成功' if success else '失败'}")
            
            if not success and 'reason' in details:
                print(f"失败原因: {details['reason']}")
            
            # 短暂暂停
            time.sleep(1.0)
        
        # 统计结果
        successes = sum(1 for r in results if r['success'])
        success_rate = successes / num_trials
        
        print(f"\n抓取实验完成")
        print(f"总尝试次数: {num_trials}")
        print(f"成功次数: {successes}")
        print(f"成功率: {success_rate*100:.1f}%")
        
        if success_rate >= Config.GRASP_SUCCESS_RATE_TARGET:
            print(f"✓ 抓取成功率满足要求 (≥{Config.GRASP_SUCCESS_RATE_TARGET*100:.0f}%)")
        else:
            print(f"⚠ 抓取成功率未达到目标")
        
        return success_rate >= Config.GRASP_SUCCESS_RATE_TARGET
    
    def collect_experiment_data(self):
        """收集实验数据并生成报告"""
        print("=" * 50)
        print("步骤 5: 实验数据采集与分析")
        print("=" * 50)
        
        # 收集所有数据
        data = self.data_collector.collect_all_data()
        
        # 保存数据
        report_path = self.data_collector.generate_report(data)
        
        print(f"实验数据已保存到: {report_path}")
        
        # 生成可视化图表
        plots_path = self.data_collector.generate_plots(data)
        print(f"可视化图表已保存到: {plots_path}")
        
        return True
    
    def run_full_experiment(self):
        """运行完整实验流程"""
        print("\n" + "=" * 60)
        print("开始完整实验流程")
        print("=" * 60)
        
        steps = [
            ("手眼标定", self.run_handeye_calibration),
            ("视觉检测", self.run_vision_detection),
            ("轨迹规划", self.run_grasp_planning),
            ("抓取执行", lambda: self.run_grasp_execution(Config.NUM_GRASP_TRIALS)),
            ("数据采集", self.collect_experiment_data)
        ]
        
        results = {}
        
        for step_name, step_func in steps:
            print(f"\n>>> 执行: {step_name}")
            try:
                success = step_func()
                results[step_name] = success
                
                if success:
                    print(f"✓ {step_name} 成功")
                else:
                    print(f"✗ {step_name} 失败")
                    print("是否继续? (y/n)")
                    # 这里可以添加用户输入，简化版本默认继续
                    
            except Exception as e:
                print(f"✗ {step_name} 出错: {e}")
                results[step_name] = False
        
        # 总结
        print("\n" + "=" * 60)
        print("实验完成总结")
        print("=" * 60)
        
        for step_name, success in results.items():
            status = "✓ 成功" if success else "✗ 失败"
            print(f"{step_name}: {status}")
        
        success_count = sum(1 for s in results.values() if s)
        total_count = len(results)
        
        print(f"\n总成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        return results

# ====================== 主GUI界面 ======================
def create_experiment_gui(robot, experiment_manager):
    """创建集成实验功能的GUI界面"""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, scrolledtext
        import threading as gui_threading
        
        # 创建主窗口
        root = tk.Tk()
        root.title("机器人视觉抓取实验系统")
        root.geometry("900x750")
        
        # 标题
        title_frame = tk.Frame(root)
        title_frame.pack(pady=10)
        
        tk.Label(title_frame, text="基于MuJoCo的机器人视觉抓取实验系统", 
                font=("Arial", 16, "bold")).pack()
        tk.Label(title_frame, text=f"实验名称: {Config.EXPERIMENT_NAME}", 
                font=("Arial", 10)).pack()
        
        # 状态显示
        status_frame = tk.Frame(root)
        status_frame.pack(pady=10, fill=tk.X, padx=20)
        
        status_var = tk.StringVar(value="就绪")
        tk.Label(status_frame, text="状态:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Label(status_frame, textvariable=status_var, 
                font=("Arial", 10), fg="blue").pack(side=tk.LEFT, padx=10)
        
        # 日志输出
        log_frame = tk.LabelFrame(root, text="实验日志", padx=10, pady=10)
        log_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        log_text.pack(fill=tk.BOTH, expand=True)
        
        def log_message(message):
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            log_text.see(tk.END)
            root.update_idletasks()
        
        # 实验控制按钮
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)
        
        def run_in_thread(func, *args):
            """在新线程中运行实验函数"""
            def wrapper():
                status_var.set("运行中...")
                root.update()
                try:
                    result = func(*args)
                    status_var.set("完成")
                    return result
                except Exception as e:
                    status_var.set("错误")
                    log_message(f"错误: {e}")
                    return None
            
            thread = gui_threading.Thread(target=wrapper, daemon=True)
            thread.start()
        
        # 单个步骤按钮
        steps_frame = tk.LabelFrame(root, text="实验步骤", padx=10, pady=10)
        steps_frame.pack(pady=10, padx=20, fill=tk.X)
        
        btn_calibrate = tk.Button(steps_frame, text="1. 手眼标定", width=15,
                                command=lambda: run_in_thread(experiment_manager.run_handeye_calibration))
        btn_calibrate.grid(row=0, column=0, padx=5, pady=5)
        
        btn_detect = tk.Button(steps_frame, text="2. 视觉检测", width=15,
                             command=lambda: run_in_thread(experiment_manager.run_vision_detection))
        btn_detect.grid(row=0, column=1, padx=5, pady=5)
        
        btn_plan = tk.Button(steps_frame, text="3. 轨迹规划", width=15,
                           command=lambda: run_in_thread(experiment_manager.run_grasp_planning))
        btn_plan.grid(row=0, column=2, padx=5, pady=5)
        
        btn_execute = tk.Button(steps_frame, text="4. 抓取执行", width=15,
                              command=lambda: run_in_thread(experiment_manager.run_grasp_execution, 5))
        btn_execute.grid(row=0, column=3, padx=5, pady=5)
        
        btn_collect = tk.Button(steps_frame, text="5. 数据采集", width=15,
                              command=lambda: run_in_thread(experiment_manager.collect_experiment_data))
        btn_collect.grid(row=0, column=4, padx=5, pady=5)
        
        # 完整实验按钮
        full_exp_frame = tk.Frame(root)
        full_exp_frame.pack(pady=10)
        
        btn_full = tk.Button(full_exp_frame, text="运行完整实验", width=20, height=2,
                           font=("Arial", 12, "bold"),
                           command=lambda: run_in_thread(experiment_manager.run_full_experiment))
        btn_full.pack()
        
        # 关节控制面板
        joint_frame = tk.LabelFrame(root, text="手动关节控制", padx=10, pady=10)
        joint_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # 创建关节控制滑条
        joint_sliders = []
        joint_labels = []
        
        for i in range(min(8, robot.n_controlled)):  # 只显示前8个关节
            joint_row = tk.Frame(joint_frame)
            joint_row.pack(fill=tk.X, pady=2)
            
            tk.Label(joint_row, text=f"关节{i}", width=8).pack(side=tk.LEFT)
            
            slider = ttk.Scale(joint_row, from_=-np.pi, to=np.pi, 
                             orient=tk.HORIZONTAL, length=200)
            slider.set(robot.get_joint_target(i))
            slider.pack(side=tk.LEFT, padx=10)
            
            def make_slider_callback(idx, slider_obj):
                def callback(val):
                    robot.set_joint_target(idx, float(val))
                return callback
            
            slider.configure(command=make_slider_callback(i, slider))
            joint_sliders.append(slider)
            
            value_label = tk.Label(joint_row, text=f"{robot.get_joint_target(i):.3f}", width=10)
            value_label.pack(side=tk.LEFT)
            joint_labels.append(value_label)
        
        # 全局控制按钮
        global_frame = tk.Frame(joint_frame)
        global_frame.pack(pady=10)
        
        btn_reset = tk.Button(global_frame, text="重置到零位", 
                            command=robot.reset_to_zero)
        btn_reset.pack(side=tk.LEFT, padx=5)
        
        btn_home = tk.Button(global_frame, text="归位", 
                           command=robot.reset_to_current)
        btn_home.pack(side=tk.LEFT, padx=5)
        
        # 更新关节显示
        def update_joint_display():
            for i, label in enumerate(joint_labels):
                if i < robot.n_controlled:
                    current = robot.get_joint_current(i)
                    label.config(text=f"{current:.3f}")
            root.after(100, update_joint_display)
        
        update_joint_display()
        
        # 系统信息
        info_frame = tk.Frame(root)
        info_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(info_frame, text="系统信息:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        tk.Label(info_frame, text=f"受控关节: {robot.n_controlled} | 相机: {Config.CAMERA_NAME}").pack(anchor=tk.W)
        tk.Label(info_frame, text="使用说明: 1) 先运行手眼标定 2) 运行完整实验或分步执行").pack(anchor=tk.W)
        
        # 重定向print到日志
        import sys
        class LogWriter:
            def __init__(self, text_widget):
                self.text_widget = text_widget
            
            def write(self, message):
                if message.strip():
                    log_message(message.strip())
            
            def flush(self):
                pass
        
        sys.stdout = LogWriter(log_text)
        sys.stderr = LogWriter(log_text)
        
        log_message("系统启动完成")
        log_message("点击按钮开始实验")
        
        # 退出处理
        def on_closing():
            if messagebox.askokcancel("退出", "确定要退出吗?"):
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        return root
        
    except ImportError as e:
        print(f"GUI创建失败: {e}")
        print("需要安装tkinter: sudo apt-get install python3-tk")
        return None

# ====================== 主程序 ======================
def main():
    print("=" * 60)
    print("机器人视觉抓取实验系统")
    print("=" * 60)
    
    # 初始化机器人控制器
    print("初始化机器人控制器...")
    robot = RobotController()
    
    # 初始化实验管理器
    print("初始化实验管理器...")
    experiment_manager = ExperimentManager(robot)
    
    # 创建GUI界面
    print("创建GUI界面...")
    root = create_experiment_gui(robot, experiment_manager)
    
    if root is None:
        print("GUI创建失败，使用命令行模式")
        print("输入 'full' 运行完整实验，或输入步骤编号运行单个步骤")
        print("1: 手眼标定, 2: 视觉检测, 3: 轨迹规划, 4: 抓取执行, 5: 数据采集")
        
        # 命令行交互
        while True:
            cmd = input("\n请输入命令 (1-5, full, quit): ").strip().lower()
            
            if cmd == 'quit':
                break
            elif cmd == 'full':
                experiment_manager.run_full_experiment()
            elif cmd == '1':
                experiment_manager.run_handeye_calibration()
            elif cmd == '2':
                experiment_manager.run_vision_detection()
            elif cmd == '3':
                experiment_manager.run_grasp_planning()
            elif cmd == '4':
                experiment_manager.run_grasp_execution(5)
            elif cmd == '5':
                experiment_manager.collect_experiment_data()
    
    else:
        # 启动MuJoCo viewer
        print("启动MuJoCo viewer...")
        
        def run_viewer():
            with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.distance = 2.5
                viewer.cam.azimuth = 45
                viewer.cam.elevation = -20
                
                print("MuJoCo viewer已启动")
                
                # 仿真循环
                while viewer.is_running():
                    try:
                        # 更新机器人控制
                        robot.update_control()
                        
                        # 执行仿真步骤
                        mujoco.mj_step(robot.model, robot.data)
                        
                        # 同步viewer
                        viewer.sync()
                        
                        # 短暂延迟
                        time.sleep(0.001)
                        
                    except Exception as e:
                        print(f"Viewer错误: {e}")
                        break
        
        # 在新线程中运行viewer
        viewer_thread = threading.Thread(target=run_viewer, daemon=True)
        viewer_thread.start()
        
        # 运行GUI主循环
        print("启动GUI主循环...")
        try:
            root.mainloop()
        except Exception as e:
            print(f"GUI错误: {e}")
    
    print("系统退出")

if __name__ == "__main__":
    main()
