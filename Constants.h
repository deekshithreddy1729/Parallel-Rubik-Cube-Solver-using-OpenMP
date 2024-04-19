#include <unordered_map>

#ifndef CYCLES_H
#define CYCLES_H

const unsigned CUBE_ORDER        = 3;
const unsigned SIDE_COUNT        = 6;
const unsigned SIDE_PIECE_COUNT  = 8;
const unsigned PIECE_COUNT       = 48;
const unsigned CYCLE_LENGTH      = 4;
const unsigned CYCLE_COUNT       = 5;
const unsigned START_MOVE_COUNT  = 18;
const unsigned UNIQUE_EDGE_COUNT = 12;
constexpr char MOVE_NAMES[7]     = "ULFRBD";
constexpr char OPP_MOVE_NAMES[7] = "DRBLFU";
constexpr char COLOR_NAMES[7]    = "WOGRBY";

const int MOVE_CYCLES[6][5][4]
{ 
  { {  0,  2,  7,  5 }, {  1,  4,  6,  3 }, {  8, 32, 24, 16 }, {  9, 33, 25, 17 }, { 10, 34, 26, 18 } },
  { {  8, 10, 15, 13 }, {  9, 12, 14, 11 }, {  0, 16, 40, 39 }, {  3, 19, 43, 36 }, {  5, 21, 45, 34 } },
  { { 16, 18, 23, 21 }, { 17, 20, 22, 19 }, {  5, 24, 42, 15 }, {  6, 27, 41, 12 }, {  7, 29, 40, 10 } },
  { { 24, 26, 31, 29 }, { 25, 28, 30, 27 }, { 18,  2, 37, 42 }, { 20,  4, 35, 44 }, { 23,  7, 32, 47 } },
  { { 32, 34, 39, 37 }, { 33, 36, 38, 35 }, {  2,  8, 45, 31 }, {  1, 11, 46, 28 }, {  0, 13, 47, 26 } },
  { { 40, 42, 47, 45 }, { 41, 44, 46, 43 }, { 21, 29, 37, 13 }, { 22, 30, 38, 14 }, { 23, 31, 39, 15 } }
};

const int EDGES[12] = { 1, 3, 4, 6, 10, 11, 13, 20, 22, 28, 31, 38 };

const std::unordered_map<char, unsigned> m_moveMap { {'U', 0}, {'L', 1}, {'F', 2}, {'R', 3}, {'B', 4}, {'D', 5} };
#endif
