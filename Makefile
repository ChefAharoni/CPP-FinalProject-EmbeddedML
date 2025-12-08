# Makefile for code generator

CXXFLAGS = -O3 -std=c++20 \
  -I./codegen/include 

# Build
SOURCES = src/codegen.cpp \
          src/model_utils.cpp \
          src/model_validator.cpp \
          src/code_generator.cpp \
          src/filesystem_utils.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = miko_codegen

$(TARGET): $(OBJECTS)
	$(CXX) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: clean

