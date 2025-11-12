# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++23 \
           -Wall \
           -Wextra \
           -Werror \
           -Wpedantic \
           -Wshadow \
           -Wconversion \
           -Wsign-conversion \
           -Wnull-dereference \
           -Wold-style-cast \
           -Wcast-align \
           -Wunused \
           -Woverloaded-virtual \
           -Wformat=2 \
           -O2


# Source directory and files
SRCDIR = src

# Identify main/tester and library sources explicitly so we only compile
# the desired entrypoint depending on the target used.
MAIN_SRC   = $(SRCDIR)/main.cpp
TEST_SRC   = $(SRCDIR)/tester.cpp

# All other .cpp files in src are treated as library code
LIB_SOURCES = $(filter-out $(MAIN_SRC) $(TEST_SRC), $(wildcard $(SRCDIR)/*.cpp))

# Application and tester source lists
APP_SOURCES    = $(LIB_SOURCES) $(MAIN_SRC)
TESTER_SOURCES = $(LIB_SOURCES) $(TEST_SRC)

# Object files (generated from source files)
OBJECTS = $(APP_SOURCES:.cpp=.o)
TESTER_OBJECTS = $(TESTER_SOURCES:.cpp=.o)

# Header files (for dependency tracking)
HEADERS = $(wildcard $(SRCDIR)/*.h)

# Default target executable (built from main.cpp)
TARGET = pico_ml

# Tester executable (built from tester.cpp only when `make tester` is invoked)
TESTER_TARGET = pico_ml_tester

# Default target: build the executable
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	@echo "Linking $(TARGET) (uses main.cpp)..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)
	@echo "Build successful! Run with: ./$(TARGET)"

# Tester target: build a separate binary using tester.cpp as the entrypoint.
$(TESTER_TARGET): $(TESTER_OBJECTS)
	@echo "Linking $(TESTER_TARGET) (uses tester.cpp)..."
	$(CXX) $(CXXFLAGS) -o $(TESTER_TARGET) $(TESTER_OBJECTS)
	@echo "Tester build successful! Run with: ./$(TESTER_TARGET)"

# Compile source files to object files
# This pattern works with files under $(SRCDIR) (e.g. src/Imatrix.cpp -> src/Imatrix.o)
%.o: %.cpp $(HEADERS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run the default program (built from main.cpp)
run: $(TARGET)
	@echo "Running $(TARGET)..."
	@./$(TARGET)

# Run the tester binary (built from tester.cpp)
run-tester: $(TESTER_TARGET)
	@echo "Running $(TESTER_TARGET)..."
	@./$(TESTER_TARGET)

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	 rm -f $(OBJECTS) $(TESTER_OBJECTS) $(TARGET) $(TESTER_TARGET)
	@echo "Clean complete."

# Rebuild from scratch
rebuild: clean all

# Debug build with debug symbols and no optimization
debug: CXXFLAGS += -g -O0 -DDEBUG
debug: clean $(TARGET)
	@echo "Debug build complete. Run with gdb: gdb ./$(TARGET)"

# Check for memory leaks using valgrind (if available)
memcheck: $(TARGET)
	@echo "Running memory leak check..."
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(TARGET)


# Show help
help:
	@echo "Embedded ML Makefile"
	@echo "================"
	@echo "Available targets:"
	@echo "  all       - Build the project (default)"
	@echo "  run       - Build and run the program"
	@echo "  clean     - Remove all build artifacts"
	@echo "  rebuild   - Clean and rebuild from scratch"
	@echo "  debug     - Build with debug symbols"
	@echo "  memcheck  - Run valgrind memory leak check"
	@echo "  help      - Show this help message"
	@echo ""

# Phony targets (not actual files)
.PHONY: all run run-tester tester clean rebuild debug memcheck help $(TESTER_TARGET)