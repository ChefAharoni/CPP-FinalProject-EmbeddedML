# Makefile for code generator
# This builds the code generator (requires TFLite)
# The generated code does NOT require TFLite

# TensorFlow Lite paths (same as main Makefile)
TFLITE_INCLUDE ?= $(HOME)/tflite-build/tensorflow
TFLITE_LIB_DIR ?= $(HOME)/tflite-build/tensorflow/tensorflow/lite/cmake-build
TFLITE_LIB_SUBDIR ?= $(TFLITE_LIB_DIR)
TFLITE_LIB := $(TFLITE_LIB_SUBDIR)/libtensorflow-lite.a

RUY_LIB_DIR ?= $(TFLITE_LIB_DIR)/_deps/ruy-build/ruy
RUY_LIBS := $(wildcard $(RUY_LIB_DIR)/libruy*.a)

CPUINFO_LIB_DIR ?= $(TFLITE_LIB_DIR)/_deps/cpuinfo-build
CPUINFO_LIB := $(CPUINFO_LIB_DIR)/libcpuinfo.a

PTHREADPOOL_LIB_DIR ?= $(TFLITE_LIB_DIR)/pthreadpool
PTHREADPOOL_LIB := $(PTHREADPOOL_LIB_DIR)/libpthreadpool.a

FLATBUFFERS_INCLUDE ?= $(TFLITE_LIB_DIR)/flatbuffers/include

ifdef TFLITE_INCLUDE_PATH
	TFLITE_INCLUDE = $(TFLITE_INCLUDE_PATH)
endif
ifdef TFLITE_LIB_PATH
	TFLITE_LIB_DIR = $(TFLITE_LIB_PATH)
endif

# Compiler flags
CXXFLAGS = -O3 -std=c++20 \
  -I$(TFLITE_INCLUDE) \
  -I$(FLATBUFFERS_INCLUDE) \
  -I/opt/homebrew/opt/inja/include \
  -I/opt/homebrew/include \
  -I.

# Linker flags
LDFLAGS = \
  -L$(TFLITE_LIB_SUBDIR) -ltensorflow-lite \
  $(foreach L,$(RUY_LIBS),-Wl,-force_load,$(L)) \
  -Wl,-force_load,$(CPUINFO_LIB) \
  -Wl,-force_load,$(PTHREADPOOL_LIB) \
  -lpthread -lm -ldl

# Build
SOURCES = codegen.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = codegen

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: clean

