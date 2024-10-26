import torch

def sample_top_p(probs, p):
    """这种采样方法是为了在文本生成过程中增加多样性，防止模型仅生成概率最高的词，同时也能防止生成概率极低的词。"""
    # 将概率probs降序排序，并返回排序后的值probs_sort和对应的索引probs_idx
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # 计算累积的概率
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 标记出累加和大于阈值p的元素
    mask = probs_sum - probs_sort > p
    # 用0填充probs sort中所有mask为True的位置，这意味着所有累积概率超过p的词都不会被考虑。
    probs_sort[mask]=0.0
    probs_sort.div(probs_sort.sum(dim=-1, keepdim=True))
    # 从修改后的probs_sort中随机抽样，得到下一个token的索引，torch.multinomial函数可以从给定的多项分布中抽取样本。
    next_token = torch.multinomial(probs_sort, num_samples = 1)
    # 根据索引从排序之前的概率分布中找到对应的token
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token