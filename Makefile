# TensorFlow Lite paths
TFLITE_INCLUDE ?= $(HOME)/tflite-build/tensorflow
TFLITE_LIB_DIR ?= $(HOME)/tflite-build/tensorflow/tensorflow/lite/cmake-build

# If libs are inside cmake-build/lib, prefer that
TFLITE_LIB_SUBDIR ?= $(TFLITE_LIB_DIR)          # adjust if your .a is in a /lib subfolder
TFLITE_LIB        := $(TFLITE_LIB_SUBDIR)/libtensorflow-lite.a

RUY_LIB_DIR ?= $(TFLITE_LIB_DIR)/_deps/ruy-build/ruy
RUY_LIBS    := $(wildcard $(RUY_LIB_DIR)/libruy*.a)

CPUINFO_LIB_DIR ?= $(TFLITE_LIB_DIR)/_deps/cpuinfo-build
CPUINFO_LIB := $(CPUINFO_LIB_DIR)/libcpuinfo.a

PTHREADPOOL_LIB_DIR ?= $(TFLITE_LIB_DIR)/pthreadpool
PTHREADPOOL_LIB := $(PTHREADPOOL_LIB_DIR)/libpthreadpool.a


# FlatBuffers include: you FOUND it here
FLATBUFFERS_INCLUDE ?= $(TFLITE_LIB_DIR)/flatbuffers/include

FFT2D_LIB_DIR ?= $(TFLITE_LIB_DIR)/_deps/fft2d-build

PTHREADPOOL_LIB_DIR ?= $(TFLITE_LIB_DIR)/pthreadpool

ifdef TFLITE_INCLUDE_PATH
	TFLITE_INCLUDE = $(TFLITE_INCLUDE_PATH)
endif
ifdef TFLITE_LIB_PATH
	TFLITE_LIB_DIR = $(TFLITE_LIB_PATH)
endif

# Compiler flags
CXXFLAGS = -std=c++20 \
  -I$(TFLITE_INCLUDE) \
  -I$(FLATBUFFERS_INCLUDE)

# Linker flags
# Main TFLite lib first, then fft2d libs (static link order matters).
LDFLAGS = \
  -L$(TFLITE_LIB_SUBDIR) -ltensorflow-lite \
  $(foreach L,$(RUY_LIBS),-Wl,-force_load,$(L)) \
  -Wl,-force_load,$(CPUINFO_LIB) \
  -Wl,-force_load,$(PTHREADPOOL_LIB) \
  -lpthread -lm -ldl

# Build
SRCDIR = src
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = pico_ml

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: clean