#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm> // for std::shuffle
#include <random>    // for std::default_random_engine
#include <sstream>   // for std::ostringstream
#include <numeric>   // for std::iota
#include <chrono>
#include <execution>

struct Position 
{
    uint64_t occupancy;
    uint64_t pieces_lo;
    uint64_t pieces_hi;
    int16_t score;
    uint8_t result;
    uint8_t stm_king;
    uint8_t nstm_king;
};

constexpr std::size_t CHUNK_SIZE = 16384 * 32;

void saveChunkToFile(const std::vector<Position>& chunk, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(chunk.data()), chunk.size() * sizeof(Position));
    file.close();
}

std::string getTempFilename(std::size_t index) {
    return "temp_chunk_" + std::to_string(index) + ".bin";
}

void deleteFile(const std::string& filename) 
{
    if (std::remove(filename.c_str()) != 0) 
    {
        std::cerr << "Error deleting file: " << filename << std::endl;
    }
}

void interleaveChunks(const std::vector<Position>& chunk1, const std::vector<Position>& chunk2,
    std::vector<Position>& result, std::default_random_engine& generator) 
{
    std::bernoulli_distribution distribution(0.5);

    // Interleave the contents of chunk1 and chunk2 based on Bernoulli distribution
    std::size_t i = 0, j = 0;
    while (i < chunk1.size() && j < chunk2.size()) 
    {
        if (distribution(generator)) 
        {
            result.push_back(chunk1[i++]);
        }
        else 
        {
            result.push_back(chunk2[j++]);
        }
    }

    // Add the remaining elements from chunk1, if any
    while (i < chunk1.size()) 
    {
        result.push_back(chunk1[i++]);
    }

    // Add the remaining elements from chunk2, if any
    while (j < chunk2.size()) 
    {
        result.push_back(chunk2[j++]);
    }
}

// Function to process the binary file and shuffle chunks
void shuffleChunks(const std::string& inputFilename, std::vector<std::string>& tempFiles, std::size_t& chunkIndex) 
{
    std::ifstream input(inputFilename, std::ios::binary | std::ios::in);
    if (!input.is_open()) 
    {
        std::cerr << "Error opening input file" << std::endl;
        return;
    }

    std::vector<Position> chunk(CHUNK_SIZE);

    while (input.read(reinterpret_cast<char*>(chunk.data()), CHUNK_SIZE * sizeof(Position))) 
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(chunk.begin(), chunk.end(), g);

        std::string tempFilename = getTempFilename(chunkIndex);
        saveChunkToFile(chunk, tempFilename);
        tempFiles.push_back(tempFilename);

        chunk.clear();
        chunk.resize(CHUNK_SIZE);

        chunkIndex++;
    }

    chunk.resize(input.gcount() / sizeof(Position));

    if (!chunk.empty()) 
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(chunk.begin(), chunk.end(), g);

        std::string tempFilename = getTempFilename(chunkIndex);
        saveChunkToFile(chunk, tempFilename);
        tempFiles.push_back(tempFilename);
        chunkIndex++;
    }

    input.close();
}

// Function to interleave chunks in parallel
void interleaveChunksInParallel(std::vector<std::string>& tempFiles, std::size_t& chunkIndex) {
    std::random_device rd;
    std::mt19937 g(rd());

    // Interleave files until only one file is left
    while (tempFiles.size() > 1) {
        std::vector<std::string> newTempFiles;
        std::vector<std::size_t> indices(tempFiles.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), g);

        // If there's an odd number of files, move the last one to newTempFiles
        if (indices.size() % 2 == 1) 
        {
            newTempFiles.push_back(tempFiles[indices.back()]);
            indices.pop_back();
        }

        // Create pairs of indices for interleaving
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        for (std::size_t i = 0; i < indices.size(); i += 2)
        {
            pairs.emplace_back(indices[i], indices[i + 1]);
        }

        // Interleave files in parallel
        std::for_each(std::execution::par, pairs.begin(), pairs.end(), [&](const std::pair<size_t, size_t>& pair) 
        {
            // Get the index of the current pair
            std::size_t index = std::distance(pairs.begin(), std::find(pairs.begin(), pairs.end(), pair));
            auto [index1, index2] = pair;

            std::vector<Position> chunk1, chunk2, interleavedChunk;

            std::ifstream file1(tempFiles[index1], std::ios::binary | std::ios::in);
            std::ifstream file2(tempFiles[index2], std::ios::binary | std::ios::in);

            if (!file1.is_open() || !file2.is_open()) 
            {
                std::cerr << "Error opening input files for interleaving" << std::endl;
                return;
            }

            // Interleave files into new temporary file
            std::string newTempFilename = getTempFilename(chunkIndex + index);
            newTempFiles.push_back(newTempFilename);

            std::ofstream mergeFile(newTempFilename, std::ios::binary | std::ios::out);
            if (!mergeFile.is_open()) 
            {
                std::cerr << "Error opening file: " << newTempFilename << std::endl;
                return;
            }

            // Continue interleaving until both input files are read completely
            while (true) 
            {
                chunk1.resize(CHUNK_SIZE);
                chunk2.resize(CHUNK_SIZE);

                file1.read(reinterpret_cast<char*>(chunk1.data()), CHUNK_SIZE * sizeof(Position));
                file2.read(reinterpret_cast<char*>(chunk2.data()), CHUNK_SIZE * sizeof(Position));

                chunk1.resize(file1.gcount() / sizeof(Position));
                chunk2.resize(file2.gcount() / sizeof(Position));

                if (chunk1.empty() && chunk2.empty()) 
                {
                    break;
                }

                interleaveChunks(chunk1, chunk2, interleavedChunk, g);

                mergeFile.write(reinterpret_cast<const char*>(interleavedChunk.data()), interleavedChunk.size() * sizeof(Position));
                interleavedChunk.clear();
            }

            mergeFile.close();
            file1.close();
            file2.close();

            // Delete the old temporary files
            deleteFile(tempFiles[index1]);
            deleteFile(tempFiles[index2]);
        });

        chunkIndex += pairs.size();
        tempFiles = std::move(newTempFiles);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_filename>" << std::endl;
        return 1;
    }

    std::string inputFilename(argv[1]);

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::string> tempFiles;
    std::size_t chunkIndex = 0;

    shuffleChunks(inputFilename, tempFiles, chunkIndex);
    interleaveChunksInParallel(tempFiles, chunkIndex);

    std::cout << "Final result saved in: " << tempFiles.front() << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
