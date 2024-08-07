from __future__ import annotations
from dataclasses import dataclass
import ctypes
import numpy as np
import torch

result_types = {
    "batch_new": ctypes.c_void_p,
    "batch_drop": None,
    "batch_get_len": ctypes.c_uint32,
    "get_row_features": ctypes.POINTER(ctypes.c_int16),
    "get_stm_features": ctypes.POINTER(ctypes.c_int16),
    "get_nstm_features": ctypes.POINTER(ctypes.c_int16),
    "batch_get_total_features": ctypes.c_uint32,
    "get_targets": ctypes.POINTER(ctypes.c_float),
    "file_reader_new": ctypes.c_void_p,
    "close_file": None,
    "try_to_load_batch": ctypes.c_bool,
}


@dataclass
class SparseBatch:
    stm_sparse: torch.Tensor
    nstm_sparse: torch.Tensor
    target: torch.Tensor
    size: int


class BatchLoader:
    def __init__(self, lib_path: str, files: list[bytes], batch_size: int, scale: float, wdl: float) -> None:
        self.parse_lib = None
        if not files: raise ValueError("The files list cannot be empty.")
        try: self.parse_lib = ctypes.CDLL(lib_path)
        except OSError as e: raise Exception(f"Failed to load the library: {e}")
        self.load_parse_lib()

        self.files, self.file_index = files, 0
        self.batch = ctypes.c_void_p(self.parse_lib.batch_new(ctypes.c_uint32(batch_size), ctypes.c_float(scale), ctypes.c_float(wdl)))
        if self.batch.value is None: raise Exception("Failed to create batch")

        self.current_reader = ctypes.c_void_p(self.parse_lib.file_reader_new(ctypes.create_string_buffer(files[0])))
        if self.current_reader.value is None: raise Exception("Failed to create file reader")

    def next_batch(self, device: torch.device) -> tuple[bool, SparseBatch]:
        new_epoch = False
        while not self.parse_lib.try_to_load_batch(self.current_reader, self.batch):
            self.parse_lib.close_file(self.current_reader)
            self.file_index = (self.file_index + 1) % len(self.files)
            file_path_buffer = ctypes.create_string_buffer(self.files[self.file_index])
            self.current_reader = ctypes.c_void_p(self.parse_lib.file_reader_new(file_path_buffer))
            new_epoch = self.file_index == 0
        return new_epoch, self.to_pytorch_batch(device)

    def to_pytorch_batch(self, device: torch.device) -> SparseBatch:
        def to_pytorch(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array).to(device, non_blocking=True)

        total_features = self.parse_lib.batch_get_total_features(self.batch)

        rows_buffer = self.parse_lib.get_row_features(self.batch)
        rows = to_pytorch(np.ctypeslib.as_array(rows_buffer, shape=(total_features,)))

        stm_cols_buffer = self.parse_lib.get_stm_features(self.batch)
        stm_cols = to_pytorch(np.ctypeslib.as_array(stm_cols_buffer, shape=(total_features,)))

        nstm_cols_buffer = self.parse_lib.get_nstm_features(self.batch)
        nstm_cols = to_pytorch(np.ctypeslib.as_array(nstm_cols_buffer, shape=(total_features,)))

        values = torch.ones(total_features, device=device, dtype=torch.float32)

        batch_len = self.parse_lib.batch_get_len(self.batch)
        stm_sparse = torch.sparse_coo_tensor(torch.stack([rows, stm_cols], dim=0), values, (batch_len, 768))
        nstm_sparse = torch.sparse_coo_tensor(torch.stack([rows, nstm_cols], dim=0), values, (batch_len, 768))

        target = to_pytorch(np.ctypeslib.as_array(self.parse_lib.get_targets(self.batch), shape=(batch_len, 1)))

        return SparseBatch(stm_sparse, nstm_sparse, target, batch_len)

    def load_parse_lib(self):
        for func_name, restype in result_types.items():
            func = getattr(self.parse_lib, func_name, None)
            if func:
                setattr(self.parse_lib, func_name, func)
                func.restype = restype

    def __del__(self) -> None:
        self.parse_lib.close_file(self.current_reader)
        self.parse_lib.batch_drop(self.batch)
