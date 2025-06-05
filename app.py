import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from optimizer import GradientDescentOptimizer

# 创建优化器实例
optimizer = GradientDescentOptimizer()

def plot_map(friend_positions, meeting_point=None):
    """
    绘制朋友位置和会面点的地图
    
    参数:
        friend_positions: 朋友位置列表 [[x1, y1], [x2, y2], ...]
        meeting_point: 可选，会面点坐标 [x, y]
        
    返回:
        fig: matplotlib图像
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制朋友位置
    friend_positions = np.array(friend_positions)
    ax.scatter(friend_positions[:, 0], friend_positions[:, 1], color='blue', s=100, label='朋友位置')
    
    # 为每个朋友位置添加标签
    for i, pos in enumerate(friend_positions):
        ax.annotate(f"朋友{i+1}", (pos[0], pos[1]), textcoords="offset points", 
                    xytext=(0, 10), ha='center')
    
    # 如果有会面点，绘制会面点
    if meeting_point is not None:
        ax.scatter(meeting_point[0], meeting_point[1], color='red', s=200, marker='*', label='最佳碰头地点')
        
        # 绘制从朋友位置到会面点的路线
        for pos in friend_positions:
            ax.plot([pos[0], meeting_point[0]], [pos[1], meeting_point[1]], 'k--', alpha=0.3)
    
    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    ax.set_title('朋友位置和最佳碰头地点')
    ax.legend()
    ax.grid(True)
    
    return fig

def add_friend(friends_data, x_coord, y_coord, speed):
    """添加一个朋友到列表中"""
    if not friends_data:
        friends_data = []
    
    try:
        x = float(x_coord)
        y = float(y_coord)
        s = float(speed)
        
        if s <= 0:
            return friends_data, "错误：速度必须大于0"
            
        friends_data.append({"id": len(friends_data) + 1, "x": x, "y": y, "speed": s})
        return friends_data, f"已添加朋友{len(friends_data)}：位置({x}, {y})，速度{s}"
    except ValueError:
        return friends_data, "错误：请输入有效的数值"

def remove_friend(friends_data, friend_id):
    """从列表中移除一个朋友"""
    if not friends_data:
        return [], "没有朋友可以移除"
    
    try:
        friend_id = int(friend_id)
        if friend_id < 1 or friend_id > len(friends_data):
            return friends_data, f"错误：朋友ID必须在1到{len(friends_data)}之间"
        
        removed_friend = friends_data.pop(friend_id - 1)
        
        # 重新分配ID
        for i, friend in enumerate(friends_data):
            friend["id"] = i + 1
            
        return friends_data, f"已移除朋友{friend_id}：位置({removed_friend['x']}, {removed_friend['y']})，速度{removed_friend['speed']}"
    except ValueError:
        return friends_data, "错误：请输入有效的朋友ID"

def clear_friends(friends_data):
    """清空朋友列表"""
    return [], "已清空所有朋友"

def format_friends_display(friends_data):
    """格式化朋友列表显示"""
    if not friends_data:
        return "没有添加朋友"
    
    result = "已添加的朋友：\n"
    for friend in friends_data:
        result += f"朋友{friend['id']}：位置({friend['x']}, {friend['y']})，速度{friend['speed']}\n"
    
    return result

def calculate_meeting_point(friends_data, learning_rate_str):
    """计算最佳碰头地点"""
    if not friends_data or len(friends_data) < 2:
        return None, "错误：请至少添加两个朋友"
    
    try:
        # 提取朋友位置和速度
        positions = [[friend["x"], friend["y"]] for friend in friends_data]
        speeds = [friend["speed"] for friend in friends_data]
        
        # 解析学习率（如果提供）
        learning_rate = None
        if learning_rate_str.strip():
            learning_rate = float(learning_rate_str)
        
        # 找到最佳碰头地点
        result = optimizer.find_optimal_meeting_point(positions, speeds, learning_rate)
        
        # 生成结果文本
        meeting_point = result["meeting_point"]
        times = result["times"]
        
        result_str = f"最佳碰头地点：({meeting_point[0]:.4f}, {meeting_point[1]:.4f})\n\n"
        result_str += "各朋友到达碰头地点所需时间：\n"
        
        for i, time in enumerate(times):
            result_str += f"朋友{i+1}：{time:.4f} 时间单位\n"
            
        result_str += f"\n总迭代次数：{result['iterations']}\n"
        result_str += f"最终代价值（时间平方和）：{result['final_cost']:.4f}"
        
        # 绘制地图
        map_fig = plot_map(positions, meeting_point)
        
        return map_fig, result_str
    
    except Exception as e:
        return None, f"错误：{str(e)}"

def import_data(positions_str, speeds_str):
    """从字符串导入朋友数据"""
    try:
        # 解析朋友位置
        positions = []
        for pos_str in positions_str.strip().split(';'):
            if pos_str:
                x, y = map(float, pos_str.split(','))
                positions.append([x, y])
        
        # 解析朋友速度
        speeds = []
        for speed_str in speeds_str.strip().split(';'):
            if speed_str:
                speed = float(speed_str)
                if speed <= 0:
                    return [], "错误：速度必须大于0"
                speeds.append(speed)
        
        # 检查输入是否有效
        if len(positions) != len(speeds):
            return [], "错误：朋友位置和速度的数量不匹配！"
        
        if len(positions) == 0:
            return [], "错误：请至少输入一个朋友的位置和速度！"
        
        # 创建朋友数据
        friends_data = []
        for i, (pos, speed) in enumerate(zip(positions, speeds)):
            friends_data.append({"id": i + 1, "x": pos[0], "y": pos[1], "speed": speed})
        
        return friends_data, f"成功导入{len(friends_data)}个朋友的数据"
    
    except Exception as e:
        return [], f"错误：{str(e)}"

def export_data(friends_data):
    """导出朋友数据为字符串格式"""
    if not friends_data:
        return "", "", "没有朋友数据可导出"
    
    positions_str = ";".join([f"{friend['x']},{friend['y']}" for friend in friends_data])
    speeds_str = ";".join([str(friend['speed']) for friend in friends_data])
    
    return positions_str, speeds_str, f"成功导出{len(friends_data)}个朋友的数据"

def load_example(example_id):
    """加载示例数据"""
    examples = [
        # 示例1: 三角形位置，相同速度
        {
            "positions": [[0, 0], [10, 0], [5, 8]],
            "speeds": [1, 1, 1]
        },
        # 示例2: 三角形位置，不同速度
        {
            "positions": [[0, 0], [10, 0], [5, 8]],
            "speeds": [1, 2, 0.5]
        },
        # 示例3: 四个点形成正方形，相同速度
        {
            "positions": [[0, 0], [0, 10], [10, 10], [10, 0]],
            "speeds": [1, 1, 1, 1]
        },
        # 示例4: 8个朋友在城市中的不同位置
        {
            "positions": [[5, 10], [15, 5], [10, 0], [0, 5], [3, 8], [13, 8], [13, 2], [3, 2]],
            "speeds": [1, 2, 1.5, 1, 2, 1.5, 1, 2]
        },
        # 示例5: 10个朋友分布在不规则位置
        {
            "positions": [[20, 30], [15, 45], [30, 40], [40, 25], [50, 40], [45, 15], [30, 5], [10, 10], [5, 25], [25, 25]],
            "speeds": [1.2, 1.8, 2.2, 1.5, 1.0, 2.0, 1.7, 1.4, 1.9, 2.5]
        }
    ]
    
    try:
        example_id = int(example_id) - 1
        if example_id < 0 or example_id >= len(examples):
            return [], f"错误：示例ID必须在1到{len(examples)}之间"
        
        example = examples[example_id]
        friends_data = []
        
        for i, (pos, speed) in enumerate(zip(example["positions"], example["speeds"])):
            friends_data.append({"id": i + 1, "x": pos[0], "y": pos[1], "speed": speed})
        
        return friends_data, f"已加载示例{example_id + 1}，包含{len(friends_data)}个朋友"
    except ValueError:
        return [], "错误：请输入有效的示例ID"

# 创建Gradio界面
with gr.Blocks(title="Gather-Us：寻找最佳碰头地点", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🌟 Gather-Us：寻找最佳碰头地点 🌟
    
    这个应用可以帮助你找到一群朋友聚会的最佳碰头地点，考虑每个人的不同移动速度，并最小化总时间平方和。
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 添加朋友")
                
                with gr.Row():
                    x_input = gr.Number(label="X坐标", precision=2)
                    y_input = gr.Number(label="Y坐标", precision=2)
                    speed_input = gr.Number(label="速度", precision=2, value=1.0)
                
                with gr.Row():
                    add_btn = gr.Button("添加朋友", variant="primary")
                    remove_btn = gr.Button("移除朋友")
                    clear_btn = gr.Button("清空所有", variant="stop")
                
                friend_id_input = gr.Number(label="朋友ID（用于移除）", precision=0, value=1)
                
                friends_state = gr.State([])
                friends_display = gr.Textbox(label="朋友列表", value="没有添加朋友", lines=8, interactive=False)
                
                add_status = gr.Textbox(label="状态信息", interactive=False)
            
            with gr.Group():
                gr.Markdown("### 批量导入/导出")
                
                with gr.Row():
                    positions_input = gr.Textbox(
                        label="朋友位置 (格式: x1,y1;x2,y2;...)",
                        placeholder="例如：0,0;10,0;5,8",
                        lines=2
                    )
                
                with gr.Row():
                    speeds_input = gr.Textbox(
                        label="朋友速度 (格式: v1;v2;...)",
                        placeholder="例如：1;1.5;0.8",
                        lines=1
                    )
                
                with gr.Row():
                    import_btn = gr.Button("导入数据")
                    export_btn = gr.Button("导出数据")
            
            with gr.Group():
                gr.Markdown("### 示例数据")
                
                with gr.Row():
                    example_id_input = gr.Number(label="示例ID", precision=0, value=1, minimum=1, maximum=5)
                    load_example_btn = gr.Button("加载示例")
                
                gr.Markdown("""
                示例1: 三角形位置，相同速度
                示例2: 三角形位置，不同速度
                示例3: 四个点形成正方形，相同速度
                示例4: 8个朋友在城市中的不同位置
                示例5: 10个朋友分布在不规则位置
                """)
            
            with gr.Group():
                gr.Markdown("### 计算最佳碰头地点")
                learning_rate_input = gr.Textbox(
                    label="学习率 (可选，默认为0.01)",
                    placeholder="例如：0.01",
                    value="0.01"
                )
                calculate_btn = gr.Button("计算最佳碰头地点", variant="primary", size="large")
        
        with gr.Column(scale=1):
            map_output = gr.Plot(label="位置地图")
            result_output = gr.Textbox(label="计算结果", lines=10, interactive=False)
    
    # 设置事件处理
    add_btn.click(
        add_friend,
        inputs=[friends_state, x_input, y_input, speed_input],
        outputs=[friends_state, add_status]
    ).then(
        format_friends_display,
        inputs=[friends_state],
        outputs=[friends_display]
    )
    
    remove_btn.click(
        remove_friend,
        inputs=[friends_state, friend_id_input],
        outputs=[friends_state, add_status]
    ).then(
        format_friends_display,
        inputs=[friends_state],
        outputs=[friends_display]
    )
    
    clear_btn.click(
        clear_friends,
        inputs=[friends_state],
        outputs=[friends_state, add_status]
    ).then(
        format_friends_display,
        inputs=[friends_state],
        outputs=[friends_display]
    )
    
    import_btn.click(
        import_data,
        inputs=[positions_input, speeds_input],
        outputs=[friends_state, add_status]
    ).then(
        format_friends_display,
        inputs=[friends_state],
        outputs=[friends_display]
    )
    
    export_btn.click(
        export_data,
        inputs=[friends_state],
        outputs=[positions_input, speeds_input, add_status]
    )
    
    load_example_btn.click(
        load_example,
        inputs=[example_id_input],
        outputs=[friends_state, add_status]
    ).then(
        format_friends_display,
        inputs=[friends_state],
        outputs=[friends_display]
    )
    
    calculate_btn.click(
        calculate_meeting_point,
        inputs=[friends_state, learning_rate_input],
        outputs=[map_output, result_output]
    )

if __name__ == "__main__":
    app.launch() 