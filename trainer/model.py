import torch

from batchloader import Batch


# (768 -> N) x 2 -> 1
class PerspectiveNetwork(torch.nn.Module):
    def __init__(self, feature_output_size: int):
        super().__init__()
        self.feature_transformer = torch.nn.Linear(768, feature_output_size)
        self.output_layer = torch.nn.Linear(feature_output_size * 2, 1)

    def forward(self, batch: Batch):
        board_stm = batch.stm_sparse.to_dense()
        board_nstm = batch.nstm_sparse.to_dense()

        stm_perspective = self.feature_transformer(board_stm)
        nstm_perspective = self.feature_transformer(board_nstm)

        hidden_features = torch.clamp(torch.cat((stm_perspective, nstm_perspective), dim=1), 0, 1)

        return torch.sigmoid(self.output_layer(hidden_features))

    def clamp_weights(self):
        self.feature_transformer.weight.data.clamp_(-2.0, 2.0)
        self.output_layer.weight.data.clamp_(-2.0, 2.0)
