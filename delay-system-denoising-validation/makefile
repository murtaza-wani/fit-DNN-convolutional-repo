# Path to the Armadillo library
ARMADILLO_PATH = /opt/homebrew/opt/armadillo/lib/

# Compiler options
CXXFLAGS = -std=c++11 -Ofast

# Libraries
LIBS = -llapack -lblas -larmadillo

# Additional library search path
LDFLAGS = -L$(ARMADILLO_PATH)

# Source files
SRC = main.cpp helper_functions.cpp solve_network.cpp solve_dde.cpp backprop.cpp

# Output binary
prog: $(SRC)
	g++ $(CXXFLAGS) -O3 -o prog $(SRC) $(LDFLAGS) $(LIBS)