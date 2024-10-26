from text_dataset import TextDataset,read_data
from torch.utils.data import DataLoader
import argparse
import os
from utils import build_word_2_index
import torch
from tqdm import tqdm
from models import LlamaModel
import logging

def parse_args():

    parser = argparse.ArgumentParser(description="transformer_model")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--head_nums", type=int, default=12)
    parser.add_argument("--block_nums", type=int, default=24)
    parser.add_argument("--loss_rate", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=2000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(filename="train_loss.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_args()
    path = os.path.join("data", "train.txt")
    all_data, max_lens = read_data(path)
    word_2_index, index_2_word = build_word_2_index(path)
    train_dataset = TextDataset(all_data, max_lens, word_2_index)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.process_batch)



    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LlamaModel(
        embedding_nums=1024,
        token_nums=len(index_2_word),
        max_lens=max_lens,
        head_nums = args.head_nums,
        block_nums=args.block_nums
    ).to(device)

    # 计算参数量
    nums_params = sum([i.numel() for i in model.parameters()])
    # model = torch.load("model_step70.pt", map_location=device)

    opt = torch.optim.AdamW(params=model.parameters(), lr=args.loss_rate)
    step = 0
    count = 0
    epoch_loss = 0
    best_loss = 999
    for i in range(args.epoch):
        for batch_x, batch_y in tqdm(train_dataloader):
            step += 1

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            loss = model(batch_x, batch_y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            count += 1
            epoch_loss += loss
            logging.info(f"step:{step}: loss:{loss:.5f}")

        avg_loss = epoch_loss / count
        epoch_loss = 0
        count = 0
        logging.info(f"epoch:{i+1}: avg_loss:{avg_loss:.5f}, best_loss:{best_loss:.5f}")
        print(f"epoch:{i+1}: avg_loss:{avg_loss:.5f},best_loss:{best_loss:.5f}")
        # if step % 100 == 0 or avg_loss < 0.0005:
        if i % 50 == 0:
            torch.save(model, f"model_{i}.pt")
            best_loss = avg_loss