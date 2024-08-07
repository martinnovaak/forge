import torch
import struct
import numpy as np

QA = 403
QB = 64
QAB = QA * QB

def quantize(model, bin_path):
    def quant(arr, scale):
        return (arr * scale).astype(np.int16).flatten()

    f_w = model.feature_transformer.weight.detach().cpu().numpy()
    f_b = model.feature_transformer.bias.detach().cpu().numpy()
    o_w = model.output_layer.weight.detach().cpu().numpy()
    o_b = model.output_layer.bias.detach().cpu().numpy()

    quantized_data = np.concatenate([
        quant(f_w.T, QA),
        quant(f_b, QA),
        quant(o_w, QB),
        quant(o_b, QAB)
    ])

    quantized_data.tofile(bin_path)


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