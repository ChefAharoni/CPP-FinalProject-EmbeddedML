# Makefile for code generator

CXXFLAGS = -O3 -std=c++20 \
  -I./include 

# Build
SOURCES = codegen.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = codegen

$(TARGET): $(OBJECTS)
	$(CXX) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: clean

