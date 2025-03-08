import os
import subprocess
from multiprocessing import Process

def run_training_on_gpu(gpu_id, margin,layers):
    """
    在指定的 GPU 上运行训练程序，传入 margin 参数
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 设置当前进程可见的 GPU
    command = [
        "python",  # 使用 Python 命令运行
        "train.py",  # 替换为你的脚本文件名，如 context_trainer.py
        "--margin", str(margin),  # 设置 margin 参数
        "--gpu_id", str(gpu_id),  # 设置 GPU ID 参数
        "--num_layers", str(layers)  # 设置 GPU ID 参数
    ]
    # 使用 subprocess 执行命令
    subprocess.run(command)

def main():
    """
    测试不同 margin 参数，从 1 到 4.5，并分配到 8 个显卡上运行
    """
    margins = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]  # 定义 margin 参数
    num_gpus = 4  # 显卡数量
    processes = []

    # for gpu_id,margin in enumerate(margins):
    for gpu_id in range(num_gpus):
        # 为每个 margin 创建一个进程
        process = Process(target=run_training_on_gpu, args=(gpu_id, 2.5, 2))
        processes.append(process)

    # 启动所有进程
    for process in processes:
        process.start()

    # 等待所有进程结束
    for process in processes:
        process.join()

if __name__ == "__main__":
    main()
