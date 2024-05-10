import torch

from batchloader import Batch

class SCReLU(torch.nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return torch.pow(torch.clamp(input, 0, 1), 2)


# (768 -> N) x 2 -> 1
class PerspectiveNetwork(torch.nn.Module):
    def __init__(self, feature_output_size: int):
        super().__init__()
        self.feature_transformer = torch.nn.Linear(768, feature_output_size)
        self.output_layer = torch.nn.Linear(feature_output_size * 2, 1)
        self.screlu = SCReLU()

    def forward(self, batch: Batch):
        board_stm = batch.stm_sparse.to_dense()
        board_nstm = batch.nstm_sparse.to_dense()

        stm_perspective = self.feature_transformer(board_stm)
        nstm_perspective = self.feature_transformer(board_nstm)

        hidden_features = torch.cat((stm_perspective, nstm_perspective), dim=1)
        hidden_features = self.screlu(hidden_features)

        return torch.sigmoid(self.output_layer(hidden_features))

    def eval(self, fen, device):
        fen = fen.split(" ")[0]
        stm_features_dense_tensor = torch.zeros(768, device=device)
        nstm_features_dense_tensor = torch.zeros(768, device=device)

        for rank_idx, rank in enumerate(fen.split('/')):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    sq = 8 * (7 - rank_idx) + file_idx
                    piece_type = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}[char.lower()]

                    is_black_piece = char.islower()
                    piece_color = 1 if is_black_piece else 0

                    stm_features_dense_tensor[piece_color * 384 + piece_type * 64 + sq] = 1
                    nstm_features_dense_tensor[(1 - piece_color) * 384 + piece_type * 64 + (sq ^ 56)] = 1

                    file_idx += 1

        board_stm = stm_features_dense_tensor.to_dense()
        board_nstm = nstm_features_dense_tensor.to_dense()

        stm_perspective = self.feature_transformer(board_stm)
        nstm_perspective = self.feature_transformer(board_nstm)

        hidden_features = torch.cat((stm_perspective, nstm_perspective))
        hidden_features = self.screlu(hidden_features)

        print(int((torch.special.logit(torch.sigmoid(self.output_layer(hidden_features))) * 400).item()))

    def clamp_weights(self):
        self.feature_transformer.weight.data.clamp_(-1.27, 1.27)
        self.output_layer.weight.data.clamp_(-1.27, 1.27)
