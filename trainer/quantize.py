import torch
import struct
import numpy as np

from model import PerspectiveNetwork

QA = 403
QB = 64
QAB = QA * QB

def quantize(model, bin_path):
    # Extract weights from the model
    feature_weights = model.feature_transformer.weight.detach().cpu().numpy()
    feature_bias = model.feature_transformer.bias.detach().cpu().numpy()
    output_weights = model.output_layer.weight.detach().cpu().numpy()
    output_bias = model.output_layer.bias.detach().cpu().numpy()

    # Quantized weights
    feature_weight_quantized = (feature_weights * QA).T.astype(np.int16)
    feature_bias_quantized = (feature_bias * QA).astype(np.int16)
    output_weight_quantized = (output_weights * QB).astype(np.int16)
    output_bias_quantized = (output_bias * QAB).astype(np.int16)

    # Flatten to 1D lists
    feature_weight_values = feature_weight_quantized.flatten().tolist()
    feature_bias_values = feature_bias_quantized.flatten().tolist()
    output_weight_values = output_weight_quantized.flatten().tolist()
    output_bias_values = output_bias_quantized.flatten().tolist()

    # Save to binary file
    with open(bin_path, "wb") as bin_file:
        bin_file.write(struct.pack('<' + 'h' * len(feature_weight_values), *feature_weight_values))
        bin_file.write(struct.pack('<' + 'h' * len(feature_bias_values), *feature_bias_values))
        bin_file.write(struct.pack('<' + 'h' * len(output_weight_values), *output_weight_values))
        bin_file.write(struct.pack('<' + 'h' * len(output_bias_values), *output_bias_values))


def load_quantized_net(bin_path, hl_size, qa, qb):
    with open(bin_path, "rb") as bin_file:
        # Read feature weights
        feature_weights = struct.unpack(f'<{768 * hl_size}h', bin_file.read(768 * hl_size * 2))
        feature_bias = struct.unpack(f'<{hl_size}h', bin_file.read(hl_size * 2))
        output_weights = struct.unpack(f'<{2 * hl_size}h', bin_file.read(2 * hl_size * 2))
        output_bias = struct.unpack('<1h', bin_file.read(1 * 2))

    model = PerspectiveNetwork(hl_size)
    model.feature_transformer.weight.data = torch.tensor(np.array(feature_weights).reshape(768, hl_size).T / qa, dtype=torch.float32)
    model.feature_transformer.bias.data = torch.tensor(np.array(feature_bias) / qa, dtype=torch.float32)
    model.output_layer.weight.data = torch.tensor(np.array(output_weights).reshape(1, 2 * hl_size) / qb, dtype=torch.float32)
    model.output_layer.bias.data = torch.tensor(np.array(output_bias) / (qa * qb), dtype=torch.float32)

    return model
