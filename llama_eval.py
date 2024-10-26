import torch
import os
from utils import build_word_2_index

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = os.path.join("data", "train.txt")
    word_2_index, index_2_word = build_word_2_index(path)

    model = torch.load("best_loss.pt", map_location=device)
    model.eval()

    text_token = []
    while True:
        input_text = input("请输入： ")
        if len(input_text) == 0:
            continue
        input_text_token = [word_2_index.get(i, 1) for i in input_text] + [2]
        text_token.extend(input_text_token)
        text_token_tensor = torch.Tensor(text_token).unsqueeze(dim=0).to(device).long()

        response = model.generate(text_token_tensor, temperature=0.7, topK=1)
        for i in response:
            text_token.append(i)
            print(index_2_word[i], end="")
        print("\n")
        text_token.append(2)

        if len(text_token) >= 100:
            print("已达最大长度，请重新输入")
            text_token = []