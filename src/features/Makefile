CXX ?= g++
CXXFLAGS ?= -O3 -march=native -fopenmp -fPIC -std=c++17 -Wall -Wextra

TARGET_LIB = libfast_features.so
TARGET_CLI = fast_features_cli

all: $(TARGET_LIB) $(TARGET_CLI)

$(TARGET_LIB): fast_features.cpp
	$(CXX) $(CXXFLAGS) -shared fast_features.cpp -o $(TARGET_LIB)

$(TARGET_CLI): fast_features_cli.cpp fast_features.cpp
	$(CXX) $(CXXFLAGS) fast_features_cli.cpp fast_features.cpp -o $(TARGET_CLI)

clean:
	rm -f $(TARGET_LIB) $(TARGET_CLI)

.PHONY: all clean
