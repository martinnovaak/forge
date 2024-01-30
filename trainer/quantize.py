import torch
import struct
import numpy as np

QA = 255
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
