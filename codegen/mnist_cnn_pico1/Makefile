# Makefile for mnist_cnn_pico1 inference
# Auto-generated - compiles the inference executable

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Source files
SOURCES = mnist_cnn_pico1_weights.cpp \
          mnist_cnn_pico1.cpp \
          mnist_cnn_pico1_inference.cpp

# Header files
HEADERS = mnist_cnn_pico1.h

OBJECTS = $(SOURCES:.cpp=.o)
TARGET = mnist_cnn_pico1_inference

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: clean

