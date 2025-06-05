import numpy as np
from utils import calculate_cost, calculate_gradient, calculate_all_times

class GradientDescentOptimizer:
    """梯度下降优化器，用于找到最佳碰头地点"""
    
    def __init__(self, learning_rate=0.01, max_iterations=5000, convergence_threshold=1e-8):
        """
        初始化优化器
        
        参数:
            learning_rate: 学习率，控制每次迭代的步长
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值，当代价函数变化小于此值时停止迭代
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
    def optimize(self, friend_positions, speeds):
        """
        使用梯度下降寻找最佳碰头地点
        
        参数:
            friend_positions: 朋友位置列表 [[x1, y1], [x2, y2], ...]
            speeds: 朋友速度列表 [v1, v2, ...]
            
        返回:
            optimal_point: 最佳碰头地点 [x, y]
            times: 各朋友到达最佳碰头地点所需的时间列表
            iterations: 实际迭代次数
            costs: 每次迭代的代价函数值
        """
        # 初始化会面点为所有朋友位置的加权平均值（按速度的倒数加权）
        weights = [1.0/s for s in speeds]
        total_weight = sum(weights)
        weighted_positions = np.array([[p[0]*w/total_weight, p[1]*w/total_weight] 
                                     for p, w in zip(friend_positions, weights)])
        meeting_point = np.sum(weighted_positions, axis=0).tolist()
        
        # 存储迭代过程中的代价值
        previous_cost = float('inf')
        costs = []
        
        # 自适应学习率参数
        adaptive_lr = self.learning_rate
        patience = 5  # 容忍连续增加的次数
        consecutive_increases = 0
        
        for i in range(self.max_iterations):
            # 计算当前代价
            current_cost = calculate_cost(meeting_point, friend_positions, speeds)
            costs.append(current_cost)
            
            # 检查收敛性
            if abs(previous_cost - current_cost) < self.convergence_threshold:
                break
            
            # 检查代价是否增加
            if current_cost > previous_cost:
                consecutive_increases += 1
                if consecutive_increases > patience:
                    # 如果连续多次增加，减小学习率
                    adaptive_lr *= 0.5
                    consecutive_increases = 0
            else:
                consecutive_increases = 0
                
            previous_cost = current_cost
            
            # 计算梯度
            gradient = calculate_gradient(meeting_point, friend_positions, speeds)
            
            # 更新会面点
            meeting_point[0] -= adaptive_lr * gradient[0]
            meeting_point[1] -= adaptive_lr * gradient[1]
        
        # 计算各朋友到达最佳碰头地点所需的时间
        times = calculate_all_times(meeting_point, friend_positions, speeds)
        
        return meeting_point, times, i+1, costs
        
    def find_optimal_meeting_point(self, friend_positions, speeds, learning_rate=None):
        """
        对外提供的接口函数，用于寻找最佳碰头地点
        
        参数:
            friend_positions: 朋友位置列表 [[x1, y1], [x2, y2], ...]
            speeds: 朋友速度列表 [v1, v2, ...]
            learning_rate: 可选参数，学习率
            
        返回:
            result: 包含最佳碰头地点和各朋友所需时间的字典
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
            
        optimal_point, times, iterations, costs = self.optimize(friend_positions, speeds)
        
        result = {
            "meeting_point": optimal_point,
            "times": times,
            "iterations": iterations,
            "final_cost": costs[-1]
        }
        
        return result 