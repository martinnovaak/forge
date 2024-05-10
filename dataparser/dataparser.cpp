#include <iostream>
#include <array>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <fstream>
#include <cmath> // std::exp

constexpr std::array<std::int16_t, 14> PIECE_VALUES = {0, 64, 128, 192, 256, 320, 0, 0, 384, 448, 512, 576, 640, 704};

constexpr size_t CAPACITY = 16384;

struct Batch {
    std::array<int16_t, CAPACITY * 32> row_feature_buffer;
    std::array<int16_t, CAPACITY * 32> stm_feature_buffer;
    std::array<int16_t, CAPACITY * 32> nstm_feature_buffer;
    std::array<float, CAPACITY> target;
    size_t capacity;
    size_t total_features;
    size_t entries;
    float scale;
    float wdl;

    Batch(size_t capacity, float scale, float wdl)
    : capacity(capacity), total_features(0), entries(0), scale(scale), wdl(wdl) {}

    void clear() {
        entries = 0;
        total_features = 0;
    }

    void add_target(float cp, float wdl) {
        auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };

        target[entries] = sigmoid(cp / scale) * (1.0f - this->wdl) + wdl * this->wdl;
        entries++;
    }

    void add_feature_sparse(int16_t stm_feature, int16_t nstm_feature) {
        size_t index = total_features;
        row_feature_buffer[index] = static_cast<int16_t>(entries);
        stm_feature_buffer[index] = stm_feature;
        nstm_feature_buffer[index] = nstm_feature;
        total_features++;
    }

    [[nodiscard]] size_t get_capacity() const {
        return capacity;
    }

    [[nodiscard]] size_t get_len() const {
        return entries;
    }

    [[nodiscard]] size_t get_total_features() const {
        return total_features;
    }

    [[nodiscard]] const int16_t* get_row_features() const {
        return row_feature_buffer.data();
    }

    [[nodiscard]] const int16_t* get_stm_features() const {
        return stm_feature_buffer.data();
    }

    [[nodiscard]] const int16_t* get_nstm_features() const {
        return nstm_feature_buffer.data();
    }

    [[nodiscard]] const float* get_targets() const {
        return target.data();
    }
};

struct Position {
    std::uint64_t occupancy;
    __uint128_t pieces;
    int16_t score;
    uint8_t result;
    uint8_t stm_king;
    uint8_t nstm_king;
    std::array<uint8_t, 3> extra;

    [[nodiscard]] float get_score() const {
        return static_cast<float>(score);
    }

    [[nodiscard]] float get_result() const {
        return static_cast<float>(result) / 2.0f;
    }

    void process_features(Batch & batch) const {
        uint64_t occupancy = this->occupancy;
        __uint128_t pieces = this->pieces;

        while (occupancy != 0) {
            auto square = static_cast<std::int16_t>(__builtin_ctzll(occupancy));
            auto colored_piece = static_cast<std::int16_t>(pieces & 0b1111);

            occupancy &= occupancy - 1;
            pieces >>= 4;

            std::int16_t stm_feature = PIECE_VALUES[colored_piece] + square;
            std::int16_t nstm_feature = PIECE_VALUES[colored_piece ^ 8] + (square ^ 56);

            batch.add_feature_sparse(stm_feature, nstm_feature);
        }
    }
} __attribute__((packed));

class FileReader {
private:
    std::ifstream file;

public:
    explicit FileReader(const std::string& path) : file(path, std::ios::binary) {}

    std::vector<Position> get_chunk(size_t chunk_size) {
        std::vector<Position> buffer(chunk_size); // Initialize vector with zeros

        // Read bytes from the file into the buffer
        file.read(reinterpret_cast<char*>(buffer.data()), sizeof(Position) * chunk_size);

        // Check if the file reading operation failed or if less than the requested number of bytes were read
        if (!file || file.gcount() < static_cast<std::streamsize>(sizeof(Position) * chunk_size)) {
            buffer.clear(); // Clear the buffer and return an empty vector
        }

        return buffer;
    }

    bool load_next_batch(Batch& batch) {
        batch.clear();

        size_t chunk_size = batch.get_capacity();
        auto positions = get_chunk(chunk_size);

        if (positions.size() == chunk_size) {
            for (const auto& annotated : positions) {
                annotated.process_features(batch);
                batch.add_target(annotated.get_score(), annotated.get_result());
            }
            return true;
        }
        return false;
    }
};

extern "C"  {
    __declspec(dllexport) Batch* batch_new(uint32_t batch_size, float scale, float wdl) {
        return new Batch(batch_size, scale, wdl);
    }

    __declspec(dllexport) void batch_drop(Batch* batch) {
        delete batch;
    }

    __declspec(dllexport) uint32_t batch_get_len(Batch* batch) {
        return static_cast<uint32_t>(batch->get_len());
    }

    __declspec(dllexport) const int16_t* get_row_features(Batch* batch) {
        return batch->get_row_features();
    }

    __declspec(dllexport) const int16_t* get_stm_features(Batch* batch) {
        return batch->get_stm_features();
    }

    __declspec(dllexport) const int16_t* get_nstm_features(Batch* batch) {
        return batch->get_nstm_features();
    }

    __declspec(dllexport) uint32_t batch_get_total_features(Batch* batch) {
        return static_cast<uint32_t>(batch->get_total_features());
    }

    __declspec(dllexport) const float* get_targets(Batch* batch) {
        return batch->get_targets();
    }

    __declspec(dllexport) FileReader* file_reader_new(const char* path) {
        return new FileReader(path);
    }

    __declspec(dllexport) void close_file(FileReader* reader) {
        delete reader;
    }

    __declspec(dllexport) bool try_to_load_batch(FileReader* reader, Batch* batch) {
        try {
            return reader->load_next_batch(*batch);
        } catch (const std::exception& e) {
            std::cerr << "Exception occurred during batch loading: " << e.what() << std::endl;
            return false;
        }
    }
};
