#############################################################
# Sean Malloy                                               #
# CSCI 476 - Cube Solver Makefile                           #
# Spring 2020                                               #
#############################################################

#############################################################
# Variables                                                 #
#############################################################

# C++ compiler
CXX      := g++

# C++ compiler flags
# CXXFLAGS := -std=c++17 -g -Wall -Werror -pthread -fopenmp
CXXFLAGS := -std=c++17 -O3 -Wall -Werror -pthread -fopenmp

#############################################################
# Rules                                                     #
#############################################################

driver : driver.cpp 
	$(CXX) $(CXXFLAGS) $^ -o $@

test : test.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

#############################################################

.PHONY: driver test

clean :
	@$(RM) driver
	@$(RM) *.o
	@$(RM) *~ 

#############################################################

