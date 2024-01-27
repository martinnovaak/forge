use std::ffi::CStr;
use std::os::raw::c_char;

mod batch;
mod file_reader;
mod position;

use batch::Batch;
use file_reader::FileReader;

#[no_mangle]
pub unsafe extern "C" fn batch_new(batch_size: u32, scale: f32, wdl: f32) -> *mut Batch {
    let batch = Batch::new(batch_size as usize, scale, wdl);
    Box::into_raw(Box::new(batch))
}

#[no_mangle]
pub unsafe extern "C" fn batch_drop(batch: *mut Batch) {
    drop(Box::from_raw(batch));
}

#[no_mangle]
pub unsafe extern "C" fn batch_get_len(batch: &mut Batch) -> u32 {
    batch.get_len() as u32
}

#[no_mangle]
pub unsafe extern "C" fn get_row_features(batch: &mut Batch) -> *const i16 {
    batch.get_row_features()
}

#[no_mangle]
pub unsafe extern "C" fn get_stm_features(batch: &mut Batch) -> *const i16 {
    batch.get_stm_features()
}

#[no_mangle]
pub unsafe extern "C" fn get_nstm_features(batch: &mut Batch) -> *const i16 {
    batch.get_nstm_features()
}

#[no_mangle]
pub unsafe extern "C" fn batch_get_total_features(batch: &mut Batch) -> u32 {
    batch.get_total_features() as u32
}

#[no_mangle]
pub unsafe extern "C" fn get_targets(batch: &mut Batch) -> *const f32 {
    batch.get_targets()
}

#[no_mangle]
pub unsafe extern "C" fn file_reader_new(path: *const c_char) -> *mut FileReader {
    let path_str = CStr::from_ptr(path).to_str().unwrap();
    let reader = FileReader::new(path_str).unwrap();

    Box::into_raw(Box::new(reader))
}

#[no_mangle]
pub unsafe extern "C" fn close_file(reader: *mut FileReader) {
    drop(Box::from_raw(reader));
}

#[no_mangle]
pub unsafe extern "C" fn try_to_load_batch(reader: &mut FileReader, batch: &mut Batch) -> bool {
    reader.load_next_batch(batch)
}
