import json
import random
from pathlib import Path

# def merge_jsonl_dataset(output_file,input_file1,input_file2):

#     data = []
#     with open(input_file1, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip(): 
#                 data.append(json.loads(line.strip()))
#     with open(input_file2, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():
#                 data.append(json.loads(line.strip()))
                
#     with open(output_file, 'w') as f:
#         for item in data:
#             f.write(json.dumps(item, ensure_ascii=False) + '\n')
#     print(f"数据集合并完成！")
#     print(f"总数据量: {len(data):,d}")

# if __name__ == "__main__":
#     merge_jsonl_dataset(
#         output_file='/data/home/Jianxin/MyProject/ContextCache/data/final/pos_dataset.jsonl',
#         input_file1='/data/home/Jianxin/MyProject/ContextCache/data/final/val_pos.jsonl',
#         input_file2='/data/home/Jianxin/MyProject/ContextCache/data/final/train_pos.jsonl'
#     )
#     merge_jsonl_dataset(
#         output_file='/data/home/Jianxin/MyProject/ContextCache/data/final/neg_dataset.jsonl',
#         input_file1='/data/home/Jianxin/MyProject/ContextCache/data/final/val_neg.jsonl',
#         input_file2='/data/home/Jianxin/MyProject/ContextCache/data/final/train_neg.jsonl'
#     )
    

def merge_datasets(pos_dataset_path: str, neg_dataset_path: str, output_path: str) -> None:
    """
    合并两个JSONL数据集，将neg_dataset中的neg字段添加到pos_dataset中相同original的项中

    Args:
        pos_dataset_path: 正样本数据集路径，包含"original", "variations", "simplified", "expanded"
        neg_dataset_path: 负样本数据集路径，包含"original"和"neg"
        output_path: 合并后数据集的输出路径
    """
    # 1. 读取负样本数据集，构建字典 {original: neg}
    neg_dict = {}
    with open(neg_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            original = data['original']
            neg = data['neg']
            neg_dict[original] = neg  # 若original重复，后者会覆盖前者

    # 2. 处理正样本数据集并合并字段
    with open(pos_dataset_path, 'r', encoding='utf-8') as pos_f, \
         open(output_path, 'w', encoding='utf-8') as out_f:
        for line in pos_f:
            pos_data = json.loads(line)
            original = pos_data['original']
            
            # 若当前original在负样本中存在，则添加neg字段
            if original in neg_dict:
                pos_data['neg'] = neg_dict[original]
            
            # 写入合并后的数据
            out_line = json.dumps(pos_data, ensure_ascii=False)
            out_f.write(out_line + '\n')
            

def split_dataset(
    merged_path: str,
    output_dir: str,
    split_ratio: float = 0.8,
    seed: int = 42
) -> None:
    """
    将合并后的数据集分割为训练集和验证集

    Args:
        merged_path: 合并后的数据集路径
        output_dir: 输出目录路径
        split_ratio: 训练集比例 (默认0.8)
        seed: 随机种子 (默认42)
    """
    # 读取所有数据
    with open(merged_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 打乱数据顺序（保证可重复性）
    random.seed(seed)
    random.shuffle(data)

    # 计算分割点
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]

    # 写入训练集
    with open(f"{output_dir}/train.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 写入验证集
    with open(f"{output_dir}/val.jsonl", 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"数据集分割完成：")
    print(f"- 训练集: {len(train_data)} 条")
    print(f"- 验证集: {len(val_data)} 条")

# 使用示例
if __name__ == "__main__":
    output_path = "/data/home/Jianxin/MyProject/ContextCache/data/new/merged_dataset.jsonl"
    output_dir = "/data/home/Jianxin/MyProject/ContextCache/data/new"
    split_dataset(
        merged_path=output_path,
        output_dir=output_dir,
        split_ratio=0.8,
        seed=42
    )