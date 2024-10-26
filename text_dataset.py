import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from utils import build_word_2_index

def read_data(path):
    with open(path, "r", encoding="utf-8") as f:
        all_data = f.read().split("\n\n")
    all_new_data = [i.split("\n") for i in all_data]
    lens = [len(i) for i in all_data]
    avg_lens = sum(lens) / len(lens)
    max_lens = int(avg_lens * 1.5) + 1
    max_lens = max(max_lens, 100)
    return all_new_data[:1], max_lens


class TextDataset(Dataset):
    def __init__(self, all_data, max_lens, word_2_index):
        self.all_data = all_data
        self.max_lens = max_lens
        self.word_2_index = word_2_index

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        data = self.all_data[index]
        content_index = []
        for content in data:
            per_content_index = [self.word_2_index[i] for i in content] + [2]
            content_index.extend(per_content_index)

        if len(content_index) > self.max_lens:
            content_index = content_index[:self.max_lens]
        content_index += [0] * (self.max_lens - len(content_index))
        content_index_x = content_index[:-1]
        content_index_y = content_index[1:]
        return content_index_x, content_index_y

    def process_batch(self, batch_data):
        batch_x, batch_y= zip(*batch_data)
        return torch.Tensor(batch_x).long(), torch.Tensor(batch_y).long()


if __name__ == "__main__":
    path = os.path.join("data", "train.txt")
    all_data, max_lens = read_data(path)
    word_2_index, index_2_word = build_word_2_index(path)
    train_dataset = TextDataset(all_data, max_lens, word_2_index)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=train_dataset.process_batch)

    for batch_x, batch_y in tqdm(train_dataloader):
        pass