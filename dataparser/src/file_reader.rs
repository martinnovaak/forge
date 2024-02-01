use std::{fs::File, io::Read, path::Path};

use crate::batch::Batch;
use crate::position::Position;

pub struct FileReader {
    file: File,
}

impl FileReader {
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self { file })
    }

    pub fn get_chunk(&mut self, chunk_size: usize) -> std::io::Result<Vec<Position>> {
        let mut buffer = Vec::with_capacity(chunk_size);
        buffer.resize(chunk_size, Position::default());

        let buffer_ptr = buffer.as_mut_ptr() as *mut u8;

        let buffer_size_bytes = std::mem::size_of::<Position>() * chunk_size;

        match self.file.read_exact(unsafe {
            std::slice::from_raw_parts_mut(buffer_ptr, buffer_size_bytes)
        }) {
            Ok(()) => Ok(buffer),
            Err(_) => Ok(Vec::new())
        }
    }

    pub fn load_next_batch(&mut self, batch: &mut Batch) -> bool {
        batch.clear();

        let chunk_size = batch.get_capacity();
        let positions_result = self.get_chunk(chunk_size);

        match positions_result {
            Ok(positions) => {
                for annotated in positions.iter() {
                    annotated.process_features(|stm_feature, nstm_feature| {
                        batch.add_feature_sparse(stm_feature, nstm_feature);
                    });
                    batch.add_target(annotated.get_score(), annotated.get_result());
                }
                batch.get_capacity() == batch.get_len()
            }
            Err(_) => false,
        }
    }
}
