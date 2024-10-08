from time import time
import torch

from batchloader import BatchLoader
from quantize import quantize


def print_epoch_stats(epoch, running_loss, iterations, fens, start_time, current_time):
    epoch_time = current_time - start_time
    message = ("\nepoch {:<2} | time: {:.2f} s | epoch loss: {:.7f} | speed: {:.2f} pos/s"
               .format(epoch, epoch_time, running_loss.item() / iterations, fens / epoch_time))
    print(message)

def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, dataloader: BatchLoader, epochs: int, lr_drop_steps: int, device: torch.device, resume_training: bool = False):
    if resume_training:
        model, optimizer, start_epoch, best_loss = load_checkpoint(model, optimizer, "checkpoint.pth")
    else:
        start_epoch = 0

    running_loss = torch.zeros(1, device=device)
    epoch_start_time = time()
    iterations = 0
    fens = 0
    epoch = start_epoch

    while epoch < epochs:
        new_epoch, batch = dataloader.next_batch(device)
        if new_epoch:
            epoch += 1

            current_time = time()
            print_epoch_stats(epoch, running_loss, iterations, fens, epoch_start_time, current_time)

            if epoch % lr_drop_steps == 0:
                optimizer.param_groups[0]["lr"] *= 0.1
                print("LR dropped")

            running_loss = torch.zeros(1, device=device)
            epoch_start_time = current_time
            iterations = 0
            fens = 0

            quantize(model, f"network/nnue_{epoch}_scaled.bin")

            save_checkpoint(model, optimizer, epoch, running_loss, "checkpoint.pth")

            model.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", device)

        optimizer.zero_grad()
        prediction = model(batch)

        loss = torch.mean(torch.abs(prediction - batch.target) ** 2.5)
        loss.backward()
        optimizer.step()
        model.clamp_weights()

        running_loss += loss
        iterations += 1
        fens += batch.size

        if fens % 163_840 == 0:
            epoch_time = time() - epoch_start_time
            formatted_fens = "{0:_}".format(fens)
            formatted_speed = "{0:_}".format(int(fens / epoch_time))
            print("\rTotal fens parsed in this epoch: {}, Time: {:.2f} s, Speed: {} pos/s"
                  .format(formatted_fens, epoch_time, formatted_speed), end='', flush=True)

        if fens % 99_942_400 == 0:
            print_epoch_stats(epoch, running_loss, iterations, fens, epoch_start_time, time())
            model.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", device)