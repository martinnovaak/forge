// BulletFormat
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Position {
    occupancy: u64,
    pieces: [u8; 16],
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
        let mut index = 0;
        const PIECE_MIRROR: [i16; 12] = [384, 448, 512, 576, 640, 704, 0, 64, 128, 192, 256, 320];

        while occupancy != 0 {
            let square = occupancy.trailing_zeros() as i16;
            let shifts = [1, 16];
            let piece = ((self.pieces[index / 2] / shifts[index % 2]) & 0b1111) as i16;

            occupancy &= occupancy - 1;
            index += 1;

            let stm_feature = piece * 64 + square;
            let nstm_feature = PIECE_MIRROR[piece as usize] + square ^ 56;

            features(stm_feature, nstm_feature);
        }
    }
}
