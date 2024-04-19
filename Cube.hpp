/*
 * Sean Malloy
 * 04/26/2020
 * Cube.hpp
 * 3x3x3 cube data structure represented as a 1D array of integers.
 */

/* SOURCES
 * [1] - https://dl.acm.org/doi/abs/10.1145/29309.29316
 * [2] - https://www.cs.princeton.edu/courses/archive/fall06/cos402/papers/korfrubik.pdf
 *
 * [1] was used as inspiration for how I represented the cube in the end,
 * however I grouped numbers by face, unlike the paper which grouped three
 * faces together when numbering.
 *
 * [2] was used as inspiration for the heuristic, which I did not follow
 * exactly. Instead I counted number of incorrect pieces rather than manhattan
 * distances, but still divided by 4 as to not overestimate.
 */

/************************************************/
// System includes
#include <iostream>
#include <cstdio>
#include <sstream>
#include <string>
#include <unordered_map>
#include <cmath>

/************************************************/
// Local includes
#include "Constants.h"

/************************************************/
// Typedefs/Macros

typedef int piece_t;

#ifndef CUBE_HPP
#define CUBE_HPP

/************************************************/

class Cube
{
public:
  // default ctor
  Cube()
  {
    for (unsigned i = 0; i < PIECE_COUNT; ++i)
      m_cube[i] = i;
  }
  
  // copy ctor
  Cube(const Cube& other)
  {
    for (unsigned i = 0; i < PIECE_COUNT; ++i)
      m_cube[i] = other.m_cube[i];
  }

  // rotate single side given a move
  void
  move(const std::string& moveStr)
  {
    if (m_moveMap.find(moveStr[0]) == m_moveMap.end())
      fprintf(stderr, "Invalid move (%s)\n", moveStr.c_str());
 
    unsigned index = m_moveMap.at(moveStr[0]);
    if (moveStr.size() == 2)
    {
      if (moveStr[1] == '\'')
        rotateSide(index, true);
      else if (moveStr[1] == '2')
      {
        rotateSide(index, false);
        rotateSide(index, false);
      }
      else
        fprintf(stderr, "Invalid move (%s)\n", moveStr.c_str());
    }
    else if (moveStr.size() == 1 && m_moveMap.find(moveStr[0]) != m_moveMap.end())
      rotateSide(index, false);
  }

  // scramble the cube given a space seperated scramble string
  void
  scramble(const std::string& moveStr)
  {
    std::stringstream tokenize(moveStr);
    std::string token;

    while (std::getline(tokenize, token, ' '))
      move(token);
  }

  bool
  isSolved()
  {
    for (unsigned i = 0; i < PIECE_COUNT; ++i)
      if ((piece_t) i != m_cube[i])
        return false;

    return true;
  }

  bool
  isSolved() const
  {
    for (unsigned i = 0; i < PIECE_COUNT; ++i)
      if ((piece_t) i != m_cube[i])
        return false;

    return true;
  }
  
  // Adapted from
  // https://www.cs.princeton.edu/courses/archive/fall06/cos402/papers/korfrubik.pdf 
  int
  distanceToSolved() const
  {
    int totalDist = 0;
    for (unsigned i = 0; i < UNIQUE_EDGE_COUNT; ++i)
      totalDist += EDGES[i] - m_cube[EDGES[i]];

    return totalDist / 4;
  }
  
  bool
  operator<(const Cube& other) const
  {
    return distanceToSolved() < other.distanceToSolved();
  }

private:
  // Rotate single side 90 degrees
  // If prime is true, rotate counter-clockwise 90 degrees, otherwise clockwise
  void
  rotateSide(int sideNum, bool prime)
  {
    for (unsigned i = 0; i < CYCLE_COUNT; ++i)
    {
      auto cycle = MOVE_CYCLES[sideNum][i];
      piece_t buffer[CYCLE_LENGTH];

      if (prime)
      {
        piece_t revCycle[CYCLE_LENGTH];
        for (unsigned f = 0, b = CYCLE_LENGTH - 1; f < CYCLE_LENGTH; ++f, --b)
          revCycle[f] = cycle[b];

        cycle = revCycle;
      }

      for (unsigned j = 1; j < CYCLE_LENGTH; ++j)
        buffer[j] = m_cube[cycle[j - 1]];
      buffer[0] = m_cube[cycle[CYCLE_LENGTH - 1]];

      for (unsigned j = 0; j < CYCLE_LENGTH; ++j)
        m_cube[cycle[j]] = buffer[j];
    }
  }
  
  // member variables
  piece_t m_cube[PIECE_COUNT];
};

#endif
