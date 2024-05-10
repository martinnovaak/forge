// BulletFormat
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Position {
    occupancy: u64,
    pieces: u128,
    score: i16,
    result: u8,
    stm_king: u8,
    nstm_king: u8,
}

impl Position {
    pub fn get_score(&self) -> f32 {
        self.score as f32
    }

    pub fn get_result(&self) -> f32 {
        self.result as f32 / 2.0
    }

    pub fn process_features<F>(&self, mut features: F)
        where
            F: FnMut(i16, i16),
    {
        let mut occupancy = self.occupancy;
        let mut pieces = self.pieces;
        const PIECE_VALUES: [i16; 14] = [ 0, 64, 128, 192, 256, 320, 0, 0, 384, 448, 512, 576, 640, 704, ];

        while occupancy != 0 {
            let square = occupancy.trailing_zeros() as i16;
            let colored_piece = (pieces & 0b1111) as i16;

            occupancy &= occupancy - 1;
            pieces >>= 4;

            let stm_feature = PIECE_VALUES[colored_piece as usize] + square;
            let nstm_feature = PIECE_VALUES[(colored_piece ^ 8) as usize] + (square ^ 56);

            features(stm_feature, nstm_feature);
        }
    }
}
