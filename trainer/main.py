import json
import os
from time import time

import torch

from batchloader import BatchLoader
from model import PerspectiveNetwork
from train import train


def main():
    if not os.path.exists("network"):
        os.makedirs("network")

    def load_config(config_path="config.json"):
        with open(config_path, 'r') as config_file:
            return json.load(config_file)

    config = load_config()

    data_parser_path = config.get("data_parser")
    data_root = config.get("data_root")
    learning_rate = config.get("learning_rate")
    epochs = config.get("epochs")
    batch_size = config.get("batch_size", 16384)
    wdl = config.get("wdl", 0.5)
    lr_drop_steps = config.get("lr_drop_steps", 10)
    scale = config.get("scale")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PerspectiveNetwork(config["hidden_layer_size"]).to(device)

    paths = [os.path.join(data_root.encode("utf-8"), file.encode("utf-8")) for file in os.listdir(data_root)]

    dataloader = BatchLoader(data_parser_path, paths, batch_size, scale, wdl)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    start_time = time()

    train(model, optimizer, dataloader, epochs, lr_drop_steps, device)

    end_time = time()
    elapsed_time = end_time - start_time

    print(f"Training 10 epochs took: {elapsed_time} seconds")


if __name__ == "__main__":
    main()
