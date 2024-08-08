from cffi import FFI
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class SparseBatch:
    stm_sparse: torch.Tensor
    nstm_sparse: torch.Tensor
    target: torch.Tensor
    size: int

class LibWrapper:
    _cdef = """
        typedef struct {
            int16_t row_feature_buffer[16384 * 32];
            int16_t stm_feature_buffer[16384 * 32];
            int16_t nstm_feature_buffer[16384 * 32];
            float target[16384];
            size_t capacity;
            size_t total_features;
            size_t entries;
            float scale;
            float wdl;
        } Batch;

        typedef struct {} FileReader;

        Batch* batch_new(uint32_t batch_size, float scale, float wdl);
        void batch_drop(Batch* batch);
        FileReader* file_reader_new(const char* path);
        void close_file(FileReader* reader);
        bool try_to_load_batch(FileReader* reader, Batch* batch);
    """

    def __init__(self, lib_path: str) -> None:
        self.ffi = FFI()
        self.ffi.cdef(self._cdef)
        self.lib = self.ffi.dlopen(lib_path)

    def batch_new(self, batch_size: int, scale: float, wdl: float):
        return self.lib.batch_new(batch_size, scale, wdl)

    def batch_drop(self, batch):
        self.lib.batch_drop(batch)

    def file_reader_new(self, file_path: bytes):
        return self.lib.file_reader_new(file_path)

    def close_file(self, reader):
        self.lib.close_file(reader)

    def try_to_load_batch(self, reader, batch):
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
        total_features = self.batch.total_features
        batch_len = self.batch.entries

        rows = self._get_buffer_data(self.batch.row_feature_buffer, total_features, np.int16, device)
        stm_cols = self._get_buffer_data(self.batch.stm_feature_buffer, total_features, np.int16, device)
        nstm_cols = self._get_buffer_data(self.batch.nstm_feature_buffer, total_features, np.int16, device)

        values = torch.ones(total_features, device=device, dtype=torch.float32)
        stm_indices = torch.stack([rows, stm_cols], dim=0)
        nstm_indices = torch.stack([rows, nstm_cols], dim=0)

        stm_sparse = torch.sparse_coo_tensor(stm_indices, values, (batch_len, 768))
        nstm_sparse = torch.sparse_coo_tensor(nstm_indices, values, (batch_len, 768))

        targets = self._get_buffer_data(self.batch.target, batch_len, np.float32, device, reshape=True)

        return SparseBatch(stm_sparse, nstm_sparse, targets, batch_len)

    def _get_buffer_data(self, buffer, length: int, dtype: np.dtype, device: torch.device, reshape: bool = False) -> torch.Tensor:
        # Get a buffer view of the _CDataBase object
        data_buffer = self.lib_wrapper.ffi.buffer(buffer, length * np.dtype(dtype).itemsize)
        # Convert this buffer view into a numpy array
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
