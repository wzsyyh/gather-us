import numpy as np

def calculate_distance(point1, point2):
    """计算两点之间的欧几里得距离"""
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def calculate_time(distance, speed):
    """计算给定距离和速度下的时间"""
    return distance / speed

def calculate_cost(meeting_point, friend_positions, speeds):
    """
    计算代价函数：所有人到达会面点所需时间的平方和
    
    参数:
        meeting_point: 会面点坐标 [x, y]
        friend_positions: 朋友位置列表 [[x1, y1], [x2, y2], ...]
        speeds: 朋友速度列表 [v1, v2, ...]
        
    返回:
        cost: 所有人到达会面点所需时间的平方和
    """
    total_cost = 0
    for i, position in enumerate(friend_positions):
        distance = calculate_distance(meeting_point, position)
        time = calculate_time(distance, speeds[i])
        total_cost += time ** 2
    
    return total_cost

def calculate_gradient(meeting_point, friend_positions, speeds):
    """
    计算代价函数相对于会面点坐标的梯度
    
    参数:
        meeting_point: 会面点坐标 [x, y]
        friend_positions: 朋友位置列表 [[x1, y1], [x2, y2], ...]
        speeds: 朋友速度列表 [v1, v2, ...]
        
    返回:
        gradient: [dJ/dx, dJ/dy]，代价函数相对于x和y的梯度
    """
    grad_x = 0
    grad_y = 0
    
    for i, position in enumerate(friend_positions):
        distance = calculate_distance(meeting_point, position)
        if distance > 1e-10:  # 避免除以零
            # 计算时间 t = d/v
            time = calculate_time(distance, speeds[i])
            
            # 计算梯度：d(t^2)/dx = 2t * dt/dx, 其中 dt/dx = (1/v) * (x-x_i)/d
            # 正确的梯度计算
            dx = meeting_point[0] - position[0]
            dy = meeting_point[1] - position[1]
            
            grad_x += 2 * time * dx / (speeds[i] * distance)
            grad_y += 2 * time * dy / (speeds[i] * distance)
    
    return [grad_x, grad_y]

def calculate_all_times(meeting_point, friend_positions, speeds):
    """
    计算所有朋友到达会面点所需的时间
    
    参数:
        meeting_point: 会面点坐标 [x, y]
        friend_positions: 朋友位置列表 [[x1, y1], [x2, y2], ...]
        speeds: 朋友速度列表 [v1, v2, ...]
        
    返回:
        times: 所有朋友到达会面点所需的时间列表
    """
    times = []
    for i, position in enumerate(friend_positions):
        distance = calculate_distance(meeting_point, position)
        time = calculate_time(distance, speeds[i])
        times.append(time)
    
    return times 