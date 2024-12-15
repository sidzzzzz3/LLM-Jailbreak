import datasets

# 加载数据集
dataset = datasets.load_dataset("walledai/HarmBench", 'contextual', split="train")

# 过滤掉以 "which" 开头的问题
def filter_which_questions(example):
    return not example["prompt"].strip().lower().startswith("which")

filtered_dataset = dataset.filter(filter_which_questions)

# 为剩余的样本添加索引列
def add_index(example, idx):
    example["index"] = idx
    return example

indexed_dataset = filtered_dataset.map(add_index, with_indices=True)

# 检查过滤后和添加索引列后的数据集
print(indexed_dataset)

# 保存为本地文件，方便检查
indexed_dataset.to_csv("filtered_harmbench_dataset.csv")

# 上传到 Hugging Face (确保已经通过 huggingface-cli login 登录)
indexed_dataset.push_to_hub("slz0106/indexed_harmbench_dataset")
