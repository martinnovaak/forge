#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <cmath>
#include <array>
#include <sstream>

enum Color : std::uint8_t {
    White, Black
};

enum Piece : std::uint8_t {
    WhitePawn, WhiteKnight, WhiteBishop, WhiteRook, WhiteQueen, WhiteKing,
    BlackPawn, BlackKnight, BlackBishop, BlackRook, BlackQueen, BlackKing
};

Piece get_white_piece(char fen_char) 
{
    switch (fen_char) {
    case 'P': return WhitePawn;
    case 'R': return WhiteRook;
    case 'N': return WhiteKnight;
    case 'B': return WhiteBishop;
    case 'Q': return WhiteQueen;
    case 'K': return WhiteKing;
    case 'p': return BlackPawn;
    case 'r': return BlackRook;
    case 'n': return BlackKnight;
    case 'b': return BlackBishop;
    case 'q': return BlackQueen;
    case 'k': return BlackKing;
    default:  throw std::runtime_error("Wrong piece char: " + std::string(1, fen_char));
    }
}

Piece get_black_piece(char fen_char) {
    switch (fen_char) {
    case 'p': return WhitePawn;
    case 'r': return WhiteRook;
    case 'n': return WhiteKnight;
    case 'b': return WhiteBishop;
    case 'q': return WhiteQueen;
    case 'k': return WhiteKing;
    case 'P': return BlackPawn;
    case 'R': return BlackRook;
    case 'N': return BlackKnight;
    case 'B': return BlackBishop;
    case 'Q': return BlackQueen;
    case 'K': return BlackKing;
    default:  throw std::runtime_error("Wrong piece char: " + std::string(1, fen_char));
    }
}

Piece get_piece(Color color, char fen_char)
{
    return (color == White) ? get_white_piece(fen_char) : get_black_piece(fen_char);
}

struct Position {
    std::uint64_t occupancy;             // occupancy bitboard
    std::array<std::uint8_t, 16> pieces; // piece list
    std::int16_t score;
    std::uint8_t result;
    std::uint8_t stm_king;
    std::uint8_t nstm_king;

    Position() : occupancy(0), pieces{}, score(0), result(0), stm_king(0), nstm_king(0) {}
};

std::string swap_fen_ranks(const std::string& fen, Color side) {
    std::array<std::string, 8> ranks;
    std::istringstream iss(fen);

    int start = (side == White) ? 7 : 0;
    int step = (side == White) ? -1 : 1;

    for (int index = start; index >= 0 && index < 8; index += step) {
        std::getline(iss, ranks[index], '/');
    }

    std::string result;
    for (const auto& rank : ranks) {
        result += rank;
    }

    return result;
}

static Position from_legacy(const std::string& raw_fen) {
    Position pos;

    std::string fen = raw_fen;
    fen.erase(std::remove_if(fen.begin(), fen.end(), [](char c) {
        return c == '|' || c == '[' || c == ']' || c == ';';
    }), fen.end());

    std::string board_str, color, _, score_str, result_str;

    std::stringstream ss(fen);
    ss >> board_str >> color >> _ >> _  >> _ >> _ >> score_str >> result_str;

    Color side = (color == "w") ? Color::White : Color::Black;

    board_str = swap_fen_ranks(board_str, side);

    int square = 0, index = 0;
    int shift[2] = { 1, 16 };

    for (char fen_char : board_str) 
    {
        if (std::isdigit(fen_char)) 
        {
            square += fen_char - '0';
        }
        else 
        {
            Piece piece = get_piece(side, fen_char);
            pos.occupancy |= (1ull << square);
            pos.pieces[index / 2] |= piece * shift[index % 2];
            square += 1;
            index += 1;

            if (piece == Piece::WhiteKing) 
            {
                pos.stm_king = square;
            } 
            else if (piece == Piece::BlackKing) 
            {
                pos.nstm_king = square;
            }
        }
    }

    if (!score_str.empty()) 
    {
        pos.score = std::stoi(score_str);
        if (side == Black) {
            pos.score = -pos.score;
        }
    }

    if (result_str == "1.0" || result_str == "1") 
    {
        pos.result = side == White ? 2 : 0;
    }
    else if (result_str == "0.5") 
    {
        pos.result = 1;
    }
    else if (result_str == "0.0" || result_str == "0") 
    {
        pos.result = side == White ? 0 : 2;
    }
    else 
    {
        throw std::runtime_error("Bad game result!");
    }

    return pos;
}

int main(int argc, char* argv[]) {
    if (argc < 3) 
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    std::ifstream input_file(argv[1]);
    if (!input_file) 
    {
        std::cerr << "Error opening input file\n";
        return 1;
    }

    std::ofstream output_file(argv[2], std::ios::binary);
    if (!output_file) 
    {
        std::cerr << "Error opening output file\n";
        return 1;
    }

    std::uint64_t max_position = std::numeric_limits<std::uint64_t>::max();
    if (argc >= 4) 
    {
        max_position = std::stoull(argv[3]);
    }

    std::string line;
    const int chunk_size = 16384;
    int lines_read = 0;
    int total_positions = 0;
    const int print_interval = 1'000'000; 

    std::vector<Position> data;
    std::vector<std::uint64_t> results(3, 0); // Wins, Draws, Losses

    auto start_time = std::chrono::high_resolution_clock::now();
    auto interval_start_time = std::chrono::high_resolution_clock::now();

    while (std::getline(input_file, line)) 
    {
        if (line.empty()) 
        {
            break;
        }

        Position pos = from_legacy(line);
        results[pos.result] += 1;
        data.emplace_back(pos);

        lines_read++;

        total_positions++;

        if (lines_read == chunk_size)
        {
            output_file.write(reinterpret_cast<const char*>(data.data()), chunk_size * sizeof(Position));

            data.clear();
            lines_read = 0;
        }

        if (total_positions % print_interval == 0)
        {
            auto interval_end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(interval_end_time - interval_start_time).count();
            std::cout << "Processed " << total_positions << " positions in " << duration << " milliseconds\n";
            interval_start_time = std::chrono::high_resolution_clock::now();
        }

        if (total_positions >= max_position)
        {
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end_time - start_time;

    std::cout << "Loaded [" << argv[1] << "]\n";
    std::cout << "Summary: " << total_positions << " Positions in " << total_time.count() << " seconds\n";
    std::cout << "Wins: " << static_cast<int>(results[2]) << ", Draws: " << static_cast<int>(results[1]) << ", Losses: " << static_cast<int>(results[0]) << '\n';
    std::cout << "Written to [" << argv[2] << "]\n";

    return 0;
}
