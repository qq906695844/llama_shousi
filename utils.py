import torch

def build_word_2_index(path):
    with open(path, "r", encoding="utf-8") as f:
        all_text = f.read().split("\n")
    word_2_index = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<SEP>": 2,
    }

    for text in all_text:
        if len(text) < 1:
            continue
        for chair in text:
            if chair not in word_2_index:
                word_2_index[chair] = len(word_2_index)

    return word_2_index, list(word_2_index)


def fill_padding_mask(score, padding_mask):
    padding_mask_sum = torch.sum(padding_mask, dim=-1)
    for i in range(padding_mask_sum.shape[0]):
        score[i, :, padding_mask_sum[i]:] = -torch.inf
    return score


def fill_look_ahead_mask(score):
    sentence_lens = score.shape[-1]
    for i in range(sentence_lens):
        score[:, i, i+1:] = -torch.inf
    return score

