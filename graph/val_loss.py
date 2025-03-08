import matplotlib.pyplot as plt
import re
import os
import matplotlib.font_manager as fm

def extract_losses_from_log(log_file):
    """从日志文件中提取损失值"""
    epochs = []
    val_losses = []
    baseline_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # 匹配验证损失行
            match = re.search(r'Val Loss - MHA: ([\d.]+), Baseline: ([\d.]+)', line)
            if match:
                val_loss = float(match.group(1))
                baseline_loss = float(match.group(2))
                epochs.append(len(val_losses) + 1)
                val_losses.append(val_loss)
                baseline_losses.append(baseline_loss)
    
    return epochs, val_losses, baseline_losses

def plot_losses(epochs, val_losses, baseline_losses, save_path):
    """绘制损失曲线"""
    # 设置中文字体
    font_path = '/data/home/Jianxin/MyProject/ContextCache/graph/ttf/SimHei.ttf'  # Ubuntu系统中的宋体路径
    font_prop = fm.FontProperties(fname=font_path)
    
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_losses, 'b-', marker='o', label='Model')
    # plt.plot(epochs, baseline_losses, 'r--', label='Only Mean for context vector')
    
    # 设置图表属性
    plt.title('Context vector Model Val Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend(prop=font_prop)
    
    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图表
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    # 设置日志文件路径和图表保存路径
    log_file = '/data/home/Jianxin/MyProject/ContextCache/results/exp_20250102_002005/training.log'
    save_path = '/data/home/Jianxin/MyProject/ContextCache/graph/pic/val_loss_new2.png'
    
    # 从日志中提取数据
    epochs, val_losses, baseline_losses = extract_losses_from_log(log_file)
    
    # 打印提取的数据
    # print("提取的验证损失数据：")
    # for e, v, b in zip(epochs, val_losses, baseline_losses):
    #     print(f"Epoch {e}: MHA Loss = {v:.4f}, Baseline Loss = {b:.4f}")
    
    # 绘制并保存图表
    plot_losses(epochs, val_losses, baseline_losses, save_path)
    print(f"\n图表已保存至：{save_path}")

if __name__ == "__main__":
    main()