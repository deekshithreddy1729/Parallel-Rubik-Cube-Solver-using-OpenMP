// System includes
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <future>

/************************************************/
// Local includes
#include "Cube.hpp"
#include "Constants.h"
#include "Timer.hpp"

/************************************************/
// Typedefs/structs
typedef std::vector<std::string> moveset_t;

struct CubeState
{
  CubeState()
    : cube(),
      solution()
  { }

  CubeState(const Cube& otherCube)
    : cube(otherCube),
      solution()
  { }

  CubeState(const CubeState& state)
    : cube(state.cube),
      solution(state.solution)
  { }

  bool
  operator<(const CubeState& state) const
  {
    return cube < state.cube;
  }

  void
  printSolution()
  {
    for (const auto& m : solution)
      std::cout << m << ' ';
  }
  
  Cube cube;
  moveset_t solution;
};

typedef std::queue<CubeState> frontierBFS_t;
typedef std::priority_queue<CubeState> frontierAStar_t;

moveset_t
getStartMoves(const std::string& faceNames);

moveset_t
serialBFS(Cube& cube);

moveset_t
serialBFSHelper(frontierBFS_t& frontier, const moveset_t& moves);

moveset_t
parallelBFS(Cube& cube, unsigned p);

CubeState
parallelBFSHelper(frontierBFS_t frontier, const moveset_t& moves, bool& finished, std::mutex& lock);

// Serial A* search, adapted from BFS.
moveset_t
serialAStar(Cube& cube);

moveset_t
serialAStarHelper(frontierAStar_t& frontier, const moveset_t& moves);

// Parallel A* search, adapted from parallel BFS.
moveset_t
parallelAStar(Cube& cube, unsigned p);


CubeState
parallelAStarHelper(frontierAStar_t frontier, const moveset_t& moves, bool& finished, std::mutex& lock);

// Returns true if same move is not being done more than once in a row, or when
// opposite face is moved before it.
bool
uniqueMoves(const char face, const moveset_t& solution);

// Returns letter representing opposite face of 'face'
char
oppositeFace(const char face);

// Partition calculation used for chunking starting move vector.
unsigned
partitionStart(const unsigned p, const unsigned tid);
/************************************************/

int
main()
{
  std::string scramble_strings[5] = {
    "F2 R' D2 F2 R2 D2",
    "D' F2 L' F2 U B2",
    "F2 U F2 D2 U L2",
    "R2 U R2 B2 U' R2",
    "L2 D2 R2 B' D2 B2"
  };

  int randNum = rand()%(5 + 1);
  
  std::string scramble = scramble_strings[randNum];

  std::cout << "Serial/Parallel (s/p) => ";
  std::string version;
  std::cin >> version;

  std::cout << "Algorithm (bfs/astar) => ";
  std::string algorithm;
  std::cin >> algorithm;

  Cube cube;
  cube.scramble(scramble);
  
  Timer t;
  moveset_t solution;
  unsigned p;
  if (version == "s")
  {
    if (algorithm == "bfs")
    {
      t.start();
      solution = serialBFS(cube);
      t.stop();
    }
    else
    {
      t.start();
      solution = serialAStar(cube);
      t.stop();
    }
  }
  else
  {
    std::cout << "p => ";
    std::cin >> p;

    if (algorithm == "bfs")
    {
      t.start();
      solution = parallelBFS(cube, p);
      t.stop();
    }
    else
    {
      t.start();
      solution = parallelAStar(cube, p);
      t.stop();
    }
  }

  std::cout << "\nScramble used => ";
  std::cout << scramble << "\n";

  std::cout << "\nSolution: ";
  for (const auto& m : solution)
    std::cout << m << ' ';
  std::cout << '\n';

  printf("Time: %.3f ms\n", t.elapsed());
  
  return 0;
}

moveset_t
getStartMoves(const std::string& faceNames)
{
  moveset_t moves;
  std::string variants = "2\'";

  for (const char faceChar : faceNames)
  {
    std::string face;
    face += faceChar;
    moves.push_back(face);

    for (const char var : variants)
      moves.push_back(face + var);
  }

  return moves;
}


moveset_t
serialBFS(Cube& cube)
{
  if (cube.isSolved())
    return moveset_t();

  moveset_t initMoves = getStartMoves(MOVE_NAMES);
  frontierBFS_t frontier;

  for (const auto& move : initMoves)
  {
    CubeState state(cube);
    state.cube.move(move);
    state.solution.push_back(move);

    if (state.cube.isSolved())
      return state.solution;

    frontier.push(state);
  }

  return serialBFSHelper(frontier, initMoves);
}

moveset_t
serialBFSHelper(frontierBFS_t& frontier, const moveset_t& moves)
{
  while (true)
  {
    for (const auto& move : moves)
    {
      if (uniqueMoves(move[0], frontier.front().solution))
      {
        CubeState copyState(frontier.front());
        copyState.cube.move(move);
        copyState.solution.push_back(move);
        
        if (copyState.cube.isSolved())
          return copyState.solution;

        frontier.push(copyState);
      }
    }
    
    frontier.pop();
  }
}

moveset_t
parallelBFS(Cube& cube, unsigned p)
{
  if (cube.isSolved())
    return moveset_t();

  bool finished = false;
  moveset_t initMoves = getStartMoves(MOVE_NAMES);

  std::vector<std::future<CubeState>> threads;
  std::mutex lock;
  #pragma omp parallel for num_threads(4) schedule(static,4)
  for (unsigned tid = 0; tid < p; ++tid)
  {
    frontierBFS_t frontier;
    for (unsigned m = partitionStart(p, tid); m < partitionStart(p, tid + 1); ++m)
    {
      CubeState state(cube);
      state.cube.move(initMoves[m]);
      state.solution.push_back(initMoves[m]);

      frontier.push(state);
    }

    threads.push_back(std::async(std::launch::async, parallelBFSHelper, 
          frontier, std::cref(initMoves), std::ref(finished), std::ref(lock)));
  }

  bool foundSolved = false;
  CubeState solved;
  // #pragma omp parallel for num_threads(4)
  for (auto& t : threads)
  {
    CubeState state = t.get();
    if (!foundSolved && state.cube.isSolved())
    {
      foundSolved = true;
      solved = state;
    }
  }

  return solved.solution;
}

CubeState
parallelBFSHelper(frontierBFS_t frontier, const moveset_t& moves, bool& finished, std::mutex& lock)
{
  while (!finished)
  {
    // #pragma omp parallel for num_threads(4)
    for (const auto& move : moves)
    {
      if (uniqueMoves(move[0], frontier.front().solution))
      {
        CubeState copyState(frontier.front());
        copyState.cube.move(move);
        copyState.solution.push_back(move);        
        
        if (!finished && copyState.cube.isSolved())
        { 
          lock.lock();
          finished = true;
          lock.unlock();
          return copyState;
        }

        frontier.push(copyState);
      }
    }

    frontier.pop();
  }

  return frontier.front();
}

/************************************************/

// Serial A* adapted from BFS.
moveset_t
serialAStar(Cube& cube)
{
  if (cube.isSolved())
    return moveset_t();

  moveset_t initMoves = getStartMoves(MOVE_NAMES);
  frontierAStar_t frontier;

  for (const auto& move : initMoves)
  {
    CubeState state(cube);
    state.cube.move(move);
    state.solution.push_back(move);
    
    if (state.cube.isSolved())
      return state.solution;

    frontier.push(state);
  }

  return serialAStarHelper(frontier, initMoves);
}


moveset_t
serialAStarHelper(frontierAStar_t& frontier, const moveset_t& moves)
{
  frontierAStar_t temp;
  while (!frontier.top().cube.isSolved())
  {
    if (frontier.size() == 0)
    {
      frontier = temp;
      temp = frontierAStar_t();
    }

    for (const auto& move : moves)
    {
      if (uniqueMoves(move[0], frontier.top().solution))
      {
        CubeState copyState(frontier.top());
        copyState.cube.move(move);
        copyState.solution.push_back(move);
        
        if (copyState.cube.isSolved())
          return copyState.solution;

        temp.push(copyState);
      }
    }
    
    frontier.pop();
  }

  return frontier.top().solution;
}

/************************************************/

// Parallel A* adapted from parallel BFS.
moveset_t
parallelAStar(Cube& cube, unsigned p)
{
  if (cube.isSolved())
    return moveset_t();

  bool finished = false;
  moveset_t initMoves = getStartMoves(MOVE_NAMES);

  std::vector<std::future<CubeState>> threads;
  std::mutex lock;
  #pragma omp parallel for num_threads(4) schedule(static,4)
  for (unsigned tid = 0; tid < p; ++tid)
  {
    frontierAStar_t frontier;
    for (unsigned m = partitionStart(p, tid); m < partitionStart(p, tid + 1); ++m)
    {
      CubeState state(cube);
      state.cube.move(initMoves[m]);
      state.solution.push_back(initMoves[m]);

      frontier.push(state);
    }

    threads.push_back(std::async(std::launch::async, parallelAStarHelper,
          frontier, std::cref(initMoves), std::ref(finished), std::ref(lock)));
  }

  bool foundSolved = false;
  CubeState solved;
  // #pragma omp parallel for num_threads(4)
  for (auto& t : threads)
  {
    CubeState state = t.get();
    if (!foundSolved && state.cube.isSolved())
    {
      foundSolved = true;
      solved = state;
    }
  }

  return solved.solution;
}


CubeState
parallelAStarHelper(frontierAStar_t frontier, const moveset_t& moves, bool& finished, std::mutex& lock)
{
  frontierAStar_t temp;
  while (!finished)
  {
    if (frontier.size() == 0)
    {
      frontier = temp;
      temp = frontierAStar_t();
    }

    // #pragma omp parallel for num_threads(4)
    for (const auto& move : moves)
    {
      if (uniqueMoves(move[0], frontier.top().solution))
      {
        CubeState copyState(frontier.top());
        copyState.cube.move(move);
        copyState.solution.push_back(move);
        
        if (copyState.cube.isSolved())
        {
          lock.lock();
          finished = true;
          lock.unlock();

          return copyState;
        }

        temp.push(copyState);
      }
    }
    
    frontier.pop();
  }

  return frontier.top();
}

bool
uniqueMoves(const char face, const moveset_t& solution)
{
  return (face != solution.back()[0] && !(solution.size() >= 3 && face == solution[solution.size() - 3][0] && 
      solution[solution.size() - 2][0] == oppositeFace(face)));
}


char
oppositeFace(const char face)
{
  switch (face)
  {
    case 'U':
      return 'D';
    case 'D':
      return 'U';
    case 'R':
      return 'L';
    case 'L':
      return 'R';
    case 'F':
      return 'B';
    default:
      return 'F';
  }
}

unsigned
partitionStart(const unsigned p, const unsigned tid)
{
  return START_MOVE_COUNT * tid / p;
}