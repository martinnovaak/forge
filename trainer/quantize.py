import torch
import struct
import numpy as np

def quantize(model, bin_path):
    feature_weight_values = (model.feature_transformer.weight.detach().cpu().numpy() * 255).T.astype(np.int16).flatten().tolist()
    feature_bias_values = (model.feature_transformer.bias.detach().cpu().numpy() * 255).astype(np.int16).flatten().tolist()
    output_weight_values = (model.output_layer.weight.detach().cpu().numpy() * 64).astype(np.int16).flatten().tolist()
    output_bias_values = (model.output_layer.bias.detach().cpu().numpy() * 16320).astype(np.int16).flatten().tolist()

    with open(bin_path, "wb") as bin_file:
        bin_file.write(struct.pack('<' + 'h' * len(feature_weight_values), *feature_weight_values))
        bin_file.write(struct.pack('<' + 'h' * len(feature_bias_values), *feature_bias_values))
        bin_file.write(struct.pack('<' + 'h' * len(output_weight_values), *output_weight_values))
        bin_file.write(struct.pack('<' + 'h' * len(output_bias_values), *output_bias_values))