from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from cffi import FFI

@dataclass
class SparseBatch:
    stm_sparse: torch.Tensor
    nstm_sparse: torch.Tensor
    target: torch.Tensor
    size: int

class LibWrapper:
    def __init__(self, lib_path: str) -> None:
        self.ffi = FFI()
        self.lib = self.ffi.dlopen(lib_path)
        self._define_functions()

    def _define_functions(self) -> None:
        self.ffi.cdef("""
            void* batch_new(uint32_t batch_size, float scale, float wdl);
            void batch_drop(void* batch);
            uint32_t batch_get_len(void* batch);
            int16_t* get_row_features(void* batch);
            int16_t* get_stm_features(void* batch);
            int16_t* get_nstm_features(void* batch);
            uint32_t batch_get_total_features(void* batch);
            float* get_targets(void* batch);
            void* file_reader_new(const char* file_path);
            void close_file(void* reader);
            bool try_to_load_batch(void* reader, void* batch);
        """)

    def batch_new(self, batch_size: int, scale: float, wdl: float) -> ffi.CData:
        return self.lib.batch_new(batch_size, scale, wdl)

    def batch_drop(self, batch: ffi.CData) -> None:
        self.lib.batch_drop(batch)

    def batch_get_len(self, batch: ffi.CData) -> int:
        return self.lib.batch_get_len(batch)

    def get_row_features(self, batch: ffi.CData) -> ffi.CData:
        return self.lib.get_row_features(batch)

    def get_stm_features(self, batch: ffi.CData) -> ffi.CData:
        return self.lib.get_stm_features(batch)

    def get_nstm_features(self, batch: ffi.CData) -> ffi.CData:
        return self.lib.get_nstm_features(batch)

    def batch_get_total_features(self, batch: ffi.CData) -> int:
        return self.lib.batch_get_total_features(batch)

    def get_targets(self, batch: ffi.CData) -> ffi.CData:
        return self.lib.get_targets(batch)

    def file_reader_new(self, file_path: bytes) -> ffi.CData:
        file_path_buffer = self.ffi.new("char[]", file_path)
        return self.lib.file_reader_new(file_path_buffer)

    def close_file(self, reader: ffi.CData) -> None:
        self.lib.close_file(reader)

    def try_to_load_batch(self, reader: ffi.CData, batch: ffi.CData) -> bool:
        return self.lib.try_to_load_batch(reader, batch)


class BatchLoader:
    def __init__(self, lib_path: str, files: list[bytes], batch_size: int, scale: float, wdl: float) -> None:
        if not files:
            raise ValueError("The files list cannot be empty.")

        self.lib_wrapper = LibWrapper(lib_path)
        self.files = files
        self.file_index = 0
        self.batch = self.lib_wrapper.batch_new(batch_size, scale, wdl)
        self.current_reader = self.lib_wrapper.file_reader_new(files[0])

    def next_batch(self, device: torch.device) -> tuple[bool, SparseBatch]:
        new_epoch = False
        while not self.lib_wrapper.try_to_load_batch(self.current_reader, self.batch):
            self._load_next_file()
            new_epoch = self.file_index == 0
        return new_epoch, self.to_pytorch_batch(device)

    def to_pytorch_batch(self, device: torch.device) -> SparseBatch:
        # Retrieve batch information
        total_features = self.lib_wrapper.batch_get_total_features(self.batch)
        batch_len = self.lib_wrapper.batch_get_len(self.batch)

        # Get feature buffers
        rows_buffer = self.lib_wrapper.get_row_features(self.batch)
        stm_cols_buffer = self.lib_wrapper.get_stm_features(self.batch)
        nstm_cols_buffer = self.lib_wrapper.get_nstm_features(self.batch)

        # Convert buffers to PyTorch tensors
        rows = self._get_buffer_data(rows_buffer, total_features, np.int16, device)
        stm_cols = self._get_buffer_data(stm_cols_buffer, total_features, np.int16, device)
        nstm_cols = self._get_buffer_data(nstm_cols_buffer, total_features, np.int16, device)

        # Create sparse tensors for STM and NSTM features
        values = torch.ones(total_features, device=device, dtype=torch.float32)
        stm_indices = torch.stack([rows, stm_cols], dim=0)
        nstm_indices = torch.stack([rows, nstm_cols], dim=0)

        stm_sparse = torch.sparse_coo_tensor(stm_indices, values, (batch_len, 768))
        nstm_sparse = torch.sparse_coo_tensor(nstm_indices, values, (batch_len, 768))

        # Get and process target buffer
        targets_buffer = self.lib_wrapper.get_targets(self.batch)
        target = self._get_buffer_data(targets_buffer, batch_len, np.float32, device, reshape=True)

        return SparseBatch(stm_sparse, nstm_sparse, target, batch_len)

    def _get_buffer_data(self, buffer: ffi.CData, length: int, dtype: np.dtype, device: torch.device, reshape: bool = False) -> torch.Tensor:
        element_size = np.dtype(dtype).itemsize
        expected_size = length * element_size
        data_buffer = self.lib_wrapper.ffi.buffer(buffer, expected_size)
        data = np.frombuffer(data_buffer, dtype=dtype)
        if reshape:
            data = data.reshape((length, 1))
        return torch.from_numpy(data).to(device, non_blocking=True)

    def _load_next_file(self) -> None:
        self.lib_wrapper.close_file(self.current_reader)
        self.file_index = (self.file_index + 1) % len(self.files)
        self.current_reader = self.lib_wrapper.file_reader_new(self.files[self.file_index])

    def __del__(self) -> None:
        if hasattr(self, 'current_reader') and self.current_reader:
            self.lib_wrapper.close_file(self.current_reader)
        if hasattr(self, 'batch') and self.batch:
            self.lib_wrapper.batch_drop(self.batch)
