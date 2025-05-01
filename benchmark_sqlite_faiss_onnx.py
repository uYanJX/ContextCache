import json
import os
import time
import argparse

from gptcache.adapter import openai
from gptcache import cache, Config
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.onnx import OnnxModelEvaluation
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def set_seed(seed=42):
    """设置所有随机种子保证可复现性"""
    random.seed(seed)       # Python内置随机数生成器
    np.random.seed(seed)    # NumPy随机数生成器

set_seed()

def evaluate_metrix(records):
    precision = records[0] / (records[0] + records[1]) if (records[0] + records[1]) > 0 else 0
    recall = records[0] / (records[0] + records[2]) if (records[0] + records[2]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def clean_numbering(origin):
    sentences = re.split(r'^\d+\.\s|(?:\n)\d+\.\s', origin, flags=re.MULTILINE)
    return [re.sub(r'^\d+\.\s*', '', sentence) for sentence in sentences if sentence.strip()]

def run_single(
    test_model = True,
    single_threshold = 0.75,
    single_threshold_rerank = 0.8,
):
    with open("/data/home/Jianxin/similiar_qqp.json", "r") as mock_file:
        mock_data = json.load(mock_file)

    test_data = mock_data[:100]
    embedding_onnx = EmbeddingOnnx()

    sqlite_file = "sqlite.db"
    faiss_file = "faiss.index"
    has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)
    cache_base = CacheBase("sqlite")
    vector_base = VectorBase("faiss", dimension=embedding_onnx.dimension)
    data_manager = get_data_manager(cache_base, vector_base, max_size=100000)
    if test_model:
        cache.init(
            embedding_func=embedding_onnx.to_embeddings,
            data_manager=data_manager,
            # similarity_evaluation=OnnxModelEvaluation(),
            config=Config(
                similarity_threshold=single_threshold,
                similarity_threshold_rerank=single_threshold_rerank,
                test_mymodel = True,
            ),
        )
    else:
        cache.init(
            embedding_func=embedding_onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=OnnxModelEvaluation(),
            config=Config(
                similarity_threshold_rerank=0.7,
                test_mymodel = False,
            ),
        )  
    cache.set_openai_key()
    i = 0

    for pair in test_data:
        pair["id"] = str(i)
        i += 1

    duplicate_data = test_data[:30]
    
    if not has_data:
        print("insert data")
        start_time = time.time()
        
        questions, answers = map(
            list, zip(*((pair["origin"], pair["id"]) for pair in duplicate_data))
        )
        
        cache.import_data(questions=questions, answers=answers)
        print(
            "end insert data, time consuming: {:.2f}s".format(time.time() - start_time)
        )

    all_time = 0.0
    hit_cache_positive, hit_cache_negative = 0, 0
    fail_count_pos,fail_count_neg = 0,0
    i=0

    random.seed(42)
    random.shuffle(test_data)
    
    for pair in test_data:
        i+=1
        mock_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": pair["similar"]},
        ]
        start_time = time.time()
        res = openai.ChatCompletion.create(
            model="qwen0.5B",
            messages=mock_messages,
        )

        if isinstance(res, str):
            if int(pair['id'])>=len(duplicate_data):
                if res == pair["similar"]:
                    fail_count_pos += 1
                else:
                    hit_cache_negative += 1
                    print("fail", pair["id"], res)
            else:
                if res == pair["similar"]:
                    fail_count_neg += 1
                    print("fail_count_neg", pair["id"], res)
                else:
                    hit_cache_negative += 1
                    print("fail", pair["id"], res)                
            continue
        
        ## 命中缓存
        res_text = openai.get_message_from_openai_answer(res)
        
        if res_text == pair["id"]:
            hit_cache_positive += 1
        else:
            print("fail", pair["id"], res_text)
            hit_cache_negative += 1
            
        consume_time = time.time() - start_time
        all_time += consume_time
        print("cache hint time consuming: {:.2f}s".format(consume_time))

    print("average time: {:.2f}s".format(all_time / len(test_data)))
    print("cache_hint_positive:", hit_cache_positive)
    print("hit_cache_negative:", hit_cache_negative)
    print("fail_count_pos:", fail_count_pos)
    print("fail_count_neg:", fail_count_neg)
    print("average embedding time: ", cache.report.average_embedding_time())
    print("average search time: ", cache.report.average_search_time())

    print("precision, recall, f1")
    print(
        evaluate_metrix(
            [hit_cache_positive, hit_cache_negative, fail_count_neg, fail_count_pos]
        )
    )
    
    
def run_multi(
    test_model = True,
    single_threshold = 0.75,
    single_threshold_rerank = 0.8,
    multi_threshold = 0.5,
    multi_threshold_rerank = 0.7,
):


    limit = 1000
    duplicate_limit = 300

    
    embedding_onnx = EmbeddingOnnx()

    sqlite_file = "sqlite.db"
    faiss_file = "faiss.index"
    has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)
    cache_base = CacheBase("sqlite")
    vector_base = VectorBase("faiss", dimension=embedding_onnx.dimension)
    data_manager = get_data_manager(cache_base, vector_base, max_size=100000)
    if test_model:
        cache.init(
            embedding_func=embedding_onnx.to_embeddings,
            data_manager=data_manager,
            # similarity_evaluation=OnnxModelEvaluation(),
            config=Config(
                similarity_threshold=single_threshold,
                similarity_threshold_rerank=single_threshold_rerank,
                dialuoge_threshold=multi_threshold,
                dialuoge_threshold_rerank=multi_threshold_rerank,
                test_mymodel = True,
            ),
        )
    else:
        cache.init(
            embedding_func=embedding_onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=OnnxModelEvaluation(),
            config=Config(
                similarity_threshold_rerank=0.7,
                test_mymodel = False,
            ),
        )  
    cache.set_openai_key()
    random.seed(42)
    i = 0

    # 加载数据并预处理
    pos_file_path = "/data/home/Jianxin/MyProject/ContextCache/data/new/val.jsonl"
    dialuoges = [json.loads(line) for line in open(pos_file_path)]
    dialuoges_new = []
    for item in dialuoges:
        if len(item['original'])>=512 or len(item['variations']) >=512 or len(item['simplified']) >= 512 or len(item['expanded']) >= 512:
            continue
        else:
            dialuoges_new.append(item)
    dialuoges = dialuoges_new
    print("dialuoges length", len(dialuoges))
    
    random.shuffle(dialuoges)
    dialuoges = dialuoges[:limit]
    
    
    multi_data = []
    for item in dialuoges:
        sim_select = random.choice(["variations", "simplified", "expanded"])
        multi_data.append(
            {
                "origin": clean_numbering(item["original"]),
                "similar": clean_numbering(item[sim_select]),
                "id": str(i),
            })
        i += 1
    
    duplicate_data_multi = multi_data[:duplicate_limit] 
    
    print("!!!insert multi turn data")
    start_time = time.time()
    questions, answers = map(
        list, zip(*((pair["origin"], pair["id"]) for pair in duplicate_data_multi))
    )

    cache.import_data(questions=questions, answers=answers)
    print(
        "end insert data, time consuming: {:.2f}s".format(time.time() - start_time)
    )


    all_time = 0.0
    hit_cache_positive, hit_cache_negative = 0, 0
    fail_count_pos,fail_count_neg = 0,0
    i=0
    
    test_data = multi_data
    random.shuffle(test_data)
    
    for pair in test_data:
        i+=1
        mock_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": pair["similar"]},
        ]
        start_time = time.time()
        res = openai.ChatCompletion.create(
            model="qwen0.5B",
            messages=mock_messages,
        )

        if isinstance(res, str):
            if isinstance(pair["similar"], list):
                ans = pair["similar"][-1]
            else:
                ans  = pair["similar"]
            
            tag = int(pair['id'])
            if tag >= duplicate_limit and tag < limit or tag >= limit+duplicate_limit and tag < 2*limit:
                if res == ans:
                    fail_count_pos += 1
                else:
                    hit_cache_negative += 1
            else: 
                if res == ans:
                    fail_count_neg += 1
                else:
                    hit_cache_negative += 1    
            continue
        
        ## 命中缓存
        res_text = openai.get_message_from_openai_answer(res)
        
        if res_text == pair["id"]:
            hit_cache_positive += 1
        else:
            print("fail", pair["id"], res_text)
            hit_cache_negative += 1
            
        consume_time = time.time() - start_time
        all_time += consume_time
        print("cache hint time consuming: {:.2f}s".format(consume_time))

    print("average time: {:.2f}s".format(all_time / len(test_data)))
    print("cache_hint_positive:", hit_cache_positive)
    print("hit_cache_negative:", hit_cache_negative)
    print("fail_count_pos:", fail_count_pos)
    print("fail_count_neg:", fail_count_neg)
    print("average embedding time: ", cache.report.average_embedding_time())
    print("average search time: ", cache.report.average_search_time())

    print("precision, recall, f1")
    print(
        evaluate_metrix(
            [hit_cache_positive, hit_cache_negative, fail_count_neg, fail_count_pos]
        )
    )


def run_mixed(
    test_model = True,
    single_threshold = 0.75,
    single_threshold_rerank = 0.8,
    multi_threshold = 0.5,
    multi_threshold_rerank = 0.7,
):


    limit = 500
    duplicate_limit = 150

    with open("/data/home/Jianxin/similiar_qqp.json", "r") as mock_file:
        mock_data = json.load(mock_file)

    test_data = mock_data[:limit]
    
    embedding_onnx = EmbeddingOnnx()

    sqlite_file = "sqlite.db"
    faiss_file = "faiss.index"
    has_data = os.path.isfile(sqlite_file) and os.path.isfile(faiss_file)
    cache_base = CacheBase("sqlite")
    vector_base = VectorBase("faiss", dimension=embedding_onnx.dimension)
    data_manager = get_data_manager(cache_base, vector_base, max_size=100000)
    if test_model:
        cache.init(
            embedding_func=embedding_onnx.to_embeddings,
            data_manager=data_manager,
            # similarity_evaluation=OnnxModelEvaluation(),
            config=Config(
                similarity_threshold=single_threshold,
                similarity_threshold_rerank=single_threshold_rerank,
                dialuoge_threshold=multi_threshold,
                dialuoge_threshold_rerank=multi_threshold_rerank,
                test_mymodel = True,
            ),
        )
    else:
        cache.init(
            embedding_func=embedding_onnx.to_embeddings,
            data_manager=data_manager,
            similarity_evaluation=OnnxModelEvaluation(),
            config=Config(
                similarity_threshold_rerank=0.7,
                dialuoge_threshold_rerank=0.7,
                test_mymodel = False,
            ),
        )  
    cache.set_openai_key()
    random.seed(42)
    i = 0

    for pair in test_data:
        pair["id"] = str(i)
        i += 1

    duplicate_data = test_data[:duplicate_limit]

    # 加载数据并预处理
    pos_file_path = "/data/home/Jianxin/MyProject/ContextCache/data/new/val.jsonl"
    dialuoges = [json.loads(line) for line in open(pos_file_path)]
    dialuoges_new = []
    for item in dialuoges:
        if len(item['original'])>=512 or len(item['variations']) >=512 or len(item['simplified']) >= 512 or len(item['expanded']) >= 512:
            continue
        else:
            dialuoges_new.append(item)
    dialuoges = dialuoges_new
    print("dialuoges length", len(dialuoges))
    
    random.shuffle(dialuoges)
    dialuoges = dialuoges[:limit]
    
    multi_data = []
    for k,item in enumerate(dialuoges):
        if k>=duplicate_limit and k<2*duplicate_limit:
            item = dialuoges[k-duplicate_limit]
            sim_select = random.choice(['reverse'])
            if sim_select == 'reverse':
                final = clean_numbering(item["variations"])
                if len(final) > 1:
                    idx = random.randint(0, len(final)-2)
                    final[-1], final[idx] = final[idx], final[-1]
            else:
                final = clean_numbering(item[sim_select])
            multi_data.append(
                {
                    "origin": clean_numbering(item["original"]),
                    "similar": final,
                    "id": str(i),
                })
        else:
            # sim_select = random.choice(["variations", "simplified", "expanded"])
            multi_data.append(
                {
                    "origin": clean_numbering(item["original"]),
                    "similar": clean_numbering(item["variations"]),
                    "id": str(i),
                })
        i += 1
    
    duplicate_data_multi = duplicate_data + multi_data[:duplicate_limit]
    # duplicate_data_multi = multi_data[:duplicate_limit]
    
    print("!!!insert multi turn data")
    start_time = time.time()
    questions, answers = map(
        list, zip(*((pair["origin"], pair["id"]) for pair in duplicate_data_multi))
    )

    cache.import_data(questions=questions, answers=answers)
    print(
        "end insert data, time consuming: {:.2f}s".format(time.time() - start_time)
    )

    all_time = 0.0
    hit_cache_positive, hit_cache_negative = 0, 0
    fail_count_pos,fail_count_neg = 0,0
    i=0
    
    test_data = multi_data + test_data
    random.shuffle(test_data)
    
    analysis_pos = []
    analysis_neg = []
    
    for pair in test_data:
        # print()
        # print(pair)
        
        i+=1
        mock_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": pair["similar"]},
        ]
        start_time = time.time()
        res,score = openai.ChatCompletion.create(
            model="qwen0.5B",
            messages=mock_messages,
        )
        
        tag = int(pair['id'])
        
        if tag<duplicate_limit or tag>=limit and tag<limit+duplicate_limit:
            analysis_pos.extend(score)
        else:
            analysis_neg.extend(score)
        
        
        if isinstance(res, str):
            if isinstance(pair["similar"], list):
                ans = pair["similar"][-1]
            else:
                ans  = pair["similar"]
            
            if tag >= duplicate_limit and tag < limit or tag >= limit+duplicate_limit and tag < 2*limit:
                if res == ans:
                    fail_count_pos += 1
                else:
                    hit_cache_negative += 1
            else: 
                if res == ans:
                    fail_count_neg += 1
                else:
                    hit_cache_negative += 1    
            continue
        
        ## 命中缓存
        res_text = openai.get_message_from_openai_answer(res)
        
        if res_text == pair["id"]:
            hit_cache_positive += 1
        else:
            print("fail", pair["id"], res_text)
            hit_cache_negative += 1
            
        consume_time = time.time() - start_time
        all_time += consume_time
        print("cache hint time consuming: {:.2f}s".format(consume_time))

    print("average time: {:.2f}s".format(all_time / len(test_data)))
    print("cache_hint_positive:", hit_cache_positive)
    print("hit_cache_negative:", hit_cache_negative)
    print("fail_count_pos:", fail_count_pos)
    print("fail_count_neg:", fail_count_neg)
    print("average embedding time: ", cache.report.average_embedding_time())
    print("average search time: ", cache.report.average_search_time())

    print("precision, recall, f1")
    print(
        evaluate_metrix(
            [hit_cache_positive, hit_cache_negative, fail_count_neg, fail_count_pos]
        )
    )
    print(len(analysis_pos), len(analysis_neg))
    visualize_score_distributions(analysis_pos, analysis_neg)

def visualize_score_distributions(analysis_pos, analysis_neg):
    """
    可视化正样本和负样本的分数分布
    
    参数:
    analysis_pos: 包含(分数,类别)元组的列表
    analysis_neg: 包含(分数,类别)元组的列表
    """
    # 导入必要的库
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    
    # 初始化不同类别的分数列表
    pos_cat0_scores = []
    pos_cat1_scores = []
    neg_cat0_scores = []
    neg_cat1_scores = []
    
    # 分类收集分数
    for score, category in analysis_pos:
        if category == 0:
            pos_cat0_scores.append(score)
        elif category == 1:
            pos_cat1_scores.append(score)
    
    for score, category in analysis_neg:
        if category == 0:
            neg_cat0_scores.append(score)
        elif category == 1:
            neg_cat1_scores.append(score)
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 可视化正样本分布
    ax1 = axes[0]
    if pos_cat0_scores:
        sns.histplot(pos_cat0_scores, kde=True, color='blue', alpha=0.6, 
                     label='Category 0', ax=ax1)
    if pos_cat1_scores:
        sns.histplot(pos_cat1_scores, kde=True, color='red', alpha=0.6, 
                     label='Category 1', ax=ax1)
    ax1.set_title('Positive Samples Score Distribution by Category')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(left=-1)
    
    # 设置更细致的横坐标刻度 - 正样本图表
    ax1.xaxis.set_major_locator(MultipleLocator(0.05))  # 每0.05一个刻度
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 显示两位小数
    ax1.grid(axis='x', which='major', linestyle='--', alpha=0.7)  # 添加垂直网格线
    ax1.legend()
    
    # 可视化负样本分布
    ax2 = axes[1]
    if neg_cat0_scores:
        sns.histplot(neg_cat0_scores, kde=True, color='blue', alpha=0.6, 
                     label='Category 0', ax=ax2)
    if neg_cat1_scores:
        sns.histplot(neg_cat1_scores, kde=True, color='red', alpha=0.6, 
                     label='Category 1', ax=ax2)
    ax2.set_title('Negative Samples Score Distribution by Category')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Frequency')
    ax2.set_xlim(left=-1)
    
    # 设置更细致的横坐标刻度 - 负样本图表
    ax2.xaxis.set_major_locator(MultipleLocator(0.05))  # 每0.05一个刻度
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 显示两位小数
    ax2.grid(axis='x', which='major', linestyle='--', alpha=0.7)  # 添加垂直网格线
    ax2.legend()
    
    # 如果标签太拥挤，可以尝试这种替代方案 (取消注释以使用)
    # ax1.xaxis.set_major_locator(MultipleLocator(0.25))  # 主刻度为0.25
    # ax1.xaxis.set_minor_locator(MultipleLocator(0.05))  # 次刻度为0.05
    # ax1.grid(which='minor', linestyle=':', alpha=0.4)  # 次要网格线
    # ax2.xaxis.set_major_locator(MultipleLocator(0.25))
    # ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
    # ax2.grid(which='minor', linestyle=':', alpha=0.4)
    
    # 自动调整标签，避免重叠
    fig.autofmt_xdate(rotation=45)  # 旋转标签，避免重叠
    
    # 添加整体标题
    plt.suptitle('Score Distribution Analysis by Category (0 vs 1)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，避免标题重叠
    
    # 显示图表
    plt.savefig("score_distribution_analysis.png", dpi=300)  # 增加DPI以提高图像质量
    
    # 输出一些统计信息
    print("统计信息:")
    if len(pos_cat0_scores)>0:
        print(f"正样本 - 类别0: {len(pos_cat0_scores)} 项, 平均分: {np.mean(pos_cat0_scores):.4f}") 
    if len(pos_cat1_scores)>0:
        print(f"正样本 - 类别1: {len(pos_cat1_scores)} 项, 平均分: {np.mean(pos_cat1_scores):.4f}")
    if len(neg_cat0_scores)>0:      
        print(f"负样本 - 类别0: {len(neg_cat0_scores)} 项, 平均分: {np.mean(neg_cat0_scores):.4f}")
    if len(neg_cat1_scores)>0:
        print(f"负样本 - 类别1: {len(neg_cat1_scores)} 项, 平均分: {np.mean(neg_cat1_scores):.4f}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run cache testing functions.")
    parser.add_argument('--function', type=str, choices=['run_single', 'run_multi' ,'run_mixed'], default="run_mixed", help="Choose the function to run: 'run' or 'run_mixed'.")
    parser.add_argument('--test_model', type=bool, default=True, help="Whether to test the model.")
    parser.add_argument('--single_threshold', type=float, default=0.7, help="Single threshold value.")
    parser.add_argument('--single_threshold_rerank', type=float, default=-1, help="Single threshold rerank value.")
    parser.add_argument('--multi_threshold', type=float, default=0.65, help="Multi threshold value.")
    parser.add_argument('--multi_threshold_rerank', type=float, default=-1, help="Multi threshold rerank value.")

## best  (0.7514619883040936, 0.8741496598639455, 0.8081761006289307) 

## average time: 0.20s
# cache_hint_positive: 249
# hit_cache_negative: 82
# fail_count_pos: 625
# fail_count_neg: 44
# average embedding time:  0.0818
# average search time:  0.0074
# precision, recall, f1
# (0.7522658610271903, 0.8498293515358362, 0.798076923076923)

# average time: 0.20s
# cache_hint_positive: 260
# hit_cache_negative: 88
# fail_count_pos: 618
# fail_count_neg: 34
# average embedding time:  0.0842
# average search time:  0.0083
# precision, recall, f1
# (0.7471264367816092, 0.8843537414965986, 0.809968847352025)

# average time: 0.23s
# cache_hint_positive: 265
# hit_cache_negative: 99
# fail_count_pos: 608
# fail_count_neg: 28
# average embedding time:  0.087
# average search time:  0.0091
# precision, recall, f1
# (0.728021978021978, 0.9044368600682594, 0.8066971080669711)

# average time: 0.18s
# cache_hint_positive: 251
# hit_cache_negative: 79
# fail_count_pos: 625
# fail_count_neg: 45
# average embedding time:  0.0873
# average search time:  0.0085
# precision, recall, f1
# (0.7606060606060606, 0.847972972972973, 0.8019169329073483)


# average time: 0.21s
# cache_hint_positive: 263
# hit_cache_negative: 96
# fail_count_pos: 610
# fail_count_neg: 31
# average embedding time:  0.0843
# average search time:  0.0088
# precision, recall, f1
# (0.7325905292479109, 0.8945578231292517, 0.8055130168453292)

# average time: 0.19s
# cache_hint_positive: 270
# hit_cache_negative: 106
# fail_count_pos: 599
# fail_count_neg: 25
# average embedding time:  0.0857
# average search time:  0.009
# precision, recall, f1
# (0.7180851063829787, 0.9152542372881356, 0.8047690014903129)

# average time: 0.19s
# cache_hint_positive: 232
# hit_cache_negative: 84
# fail_count_pos: 626
# fail_count_neg: 58
# average embedding time:  0.0783
# average search time:  0.0054
# precision, recall, f1
# (0.7341772151898734, 0.8, 0.7656765676567656)

    args = parser.parse_args()

    if args.function == 'run_single':
        run_single(
            test_model=args.test_model,
            single_threshold=args.single_threshold,
            single_threshold_rerank=args.single_threshold_rerank,
        )
    elif args.function == 'run_mixed':
        run_mixed(
            test_model=args.test_model,
            single_threshold=args.single_threshold,
            single_threshold_rerank=args.single_threshold_rerank,
            multi_threshold=args.multi_threshold,
            multi_threshold_rerank=args.multi_threshold_rerank,
        )
    else:
        run_multi(
            test_model=args.test_model,
            single_threshold=args.single_threshold,
            single_threshold_rerank=args.single_threshold_rerank,
            multi_threshold=args.multi_threshold,
            multi_threshold_rerank=args.multi_threshold_rerank
        )


