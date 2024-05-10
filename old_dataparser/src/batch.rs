pub struct Batch {
    row_feature_buffer: Box<[i16]>,
    stm_feature_buffer: Box<[i16]>,
    nstm_feature_buffer: Box<[i16]>,
    target: Box<[f32]>,
    capacity: usize,
    total_features: usize,
    entries: usize,
    scale: f32,
    wdl: f32
}

impl Batch {
    pub fn new(capacity: usize, scale: f32, wdl: f32) -> Self {
        Self {
            row_feature_buffer: vec![0; capacity * 32].into_boxed_slice(),
            stm_feature_buffer: vec![0; capacity * 32].into_boxed_slice(),
            nstm_feature_buffer: vec![0; capacity * 32].into_boxed_slice(),
            target: vec![0_f32; capacity].into_boxed_slice(),
            capacity,
            total_features: 0,
            entries: 0,
            scale,
            wdl
        }
    }

    pub fn clear(&mut self) {
        self.entries = 0;
        self.total_features = 0;
    }

    pub fn add_target(&mut self, cp: f32, wdl: f32) {
        fn sigmoid(x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }

        self.target[self.entries] = sigmoid(cp / self.scale) * (1.0 - self.wdl) + wdl * self.wdl;
        self.entries += 1;
    }

    pub fn add_feature_sparse(&mut self, stm_feature: i16, nstm_feature: i16) {
        let index = self.total_features;
        self.row_feature_buffer[index] = self.entries as i16;
        self.stm_feature_buffer[index] = stm_feature;
        self.nstm_feature_buffer[index] = nstm_feature;
        self.total_features += 1;
    }

    pub fn get_capacity(&self) -> usize {
        self.capacity
    }

    pub fn get_len(&self) -> usize {
        self.entries
    }

    pub fn get_total_features(&self) -> usize {
        self.total_features
    }

    pub fn get_row_features(&self) -> *const i16 {
        self.row_feature_buffer.as_ptr()
    }

    pub fn get_stm_features(&self) -> *const i16 {
        self.stm_feature_buffer.as_ptr()
    }

    pub fn get_nstm_features(&self) -> *const i16 {
        self.nstm_feature_buffer.as_ptr()
    }

    pub fn get_targets(&self) -> *const f32 {
        self.target.as_ptr()
    }
}
