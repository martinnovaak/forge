import json
from time import time

import torch
from batchloader import BatchLoader


def print_epoch_stats(epoch, running_loss, iterations, fens, start_time, current_time):
    epoch_time = current_time - start_time
    message = ("epoch {:<2} | time: {:.2f} s | epoch loss: {:.4f} | speed: {:.2f} pos/s"
               .format(epoch, epoch_time, running_loss.item() / iterations, fens / epoch_time))
    print(message)


def save_model_and_params(model, epoch):
    model_path = f"network/nnue_{epoch}"
    json_path = f"network/nnue_{epoch}.json"

    torch.save(model.state_dict(), model_path)

    param_map = {
        name: param.detach().cpu().numpy().tolist()
        for name, param in model.named_parameters()
    }

    with open(json_path, "w") as json_file:
        json.dump(param_map, json_file)


def train(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: BatchLoader,
        epochs: int,
        device: torch.device,
) -> None:
    running_loss = torch.zeros(1, device=device)
    epoch_start_time = time()
    iterations = 0
    fens = 0
    epoch = 0

    while epoch < epochs:
        new_epoch, batch = dataloader.next_batch(device)
        if new_epoch:
            epoch += 1

            current_time = time()
            print_epoch_stats(epoch, running_loss, iterations, fens, epoch_start_time, current_time)

            running_loss = torch.zeros(1, device=device)
            epoch_start_time = current_time
            iterations = 0
            fens = 0
            save_model_and_params(model, epoch)

        optimizer.zero_grad()
        prediction = model(batch)

        loss = torch.mean((prediction - batch.target) ** 2)
        loss.backward()
        optimizer.step()
        model.clamp_weights()

        running_loss += loss
        iterations += 1
        fens += batch.size
