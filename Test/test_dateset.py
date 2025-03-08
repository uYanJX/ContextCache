import torch
from dataset import QueryDataset

def test_dataset():
    # 创建测试数据
    test_data = [
        {
            "original": "1. 这是第一个查询\n2. 包含一个例子",
            "variations": "1. 这是相似的查询\n2. 包含一个相似的例子",
            "simplified": "1. 简单的查询\n2. 简单的例子",
            "expanded": "1. 这是更详细的第一个查询说明\n2. 包含一个详细的例子说明"
        },
        {
            "original": "1. 这是第二个查询\n2. 完全不同的内容",
            "variations": "1. 第二个查询的变体\n2. 不同的内容变化",
            "simplified": "1. 简单查询2\n2. 不同内容",
            "expanded": "1. 第二个查询的详细解释\n2. 完全不同内容的详细说明"
        },
        {
            "original": "1. 这是第一个查询\n2. 包含一个例子",
            "variations": "1. 这是相似的查询\n2. 包含一个相似的例子",
            "simplified": "1. 简单的查询\n2. 简单的例子",
            "expanded": "1. 这是更详细的第一个查询说明\n2. 包含一个详细的例子说明"
        },
        {
            "original": "1. 这是第二个查询\n2. 完全不同的内容",
            "variations": "1. 第二个查询的变体\n2. 不同的内容变化",
            "simplified": "1. 简单查询2\n2. 不同内容",
            "expanded": "1. 第二个查询的详细解释\n2. 完全不同内容的详细说明"
        }
    ]
    
    # 保存测试数据
    import json
    with open("temp_test.json", "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 初始化数据集
    dataset = QueryDataset(
        file_path="temp_test.json",
        max_seq_length=5,
        rate=0.7,
        batch_size=2
    )

    # 测试数据集的基本属性
    print(f"数据集大小: {len(dataset)}")

    # 获取第一个样本
    s1, s2, label = dataset[0]
    
    # 打印样本信息
    print("\n样本信息:")
    print(f"输入序列1形状: {s1.shape}")
    print(f"输入序列2形状: {s2.shape}")
    print(f"标签: {label}")
    print(f"设备: {s1.device}")

    # 清理临时文件
    import os
    os.remove("temp_test.json")

if __name__ == "__main__":
    test_dataset()