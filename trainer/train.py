from time import time
import torch

from batchloader import BatchLoader
from quantize import quantize


def print_epoch_stats(epoch, running_loss, iterations, fens, start_time, current_time):
    epoch_time = current_time - start_time
    message = ("epoch {:<2} | time: {:.2f} s | epoch loss: {:.4f} | speed: {:.2f} pos/s"
               .format(epoch, epoch_time, running_loss.item() / iterations, fens / epoch_time))
    print(message)

def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, dataloader: BatchLoader, epochs: int, device: torch.device):
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

            quantize(model, f"network/nnue_{epoch}_scaled.bin")

        optimizer.zero_grad()
        prediction = model(batch)

        loss = torch.mean((prediction - batch.target) ** 2)
        loss.backward()
        optimizer.step()
        model.clamp_weights()

        running_loss += loss
        iterations += 1
        fens += batch.size
