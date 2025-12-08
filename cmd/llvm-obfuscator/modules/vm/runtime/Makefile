# VM Interpreter Makefile
# Build and test the standalone VM interpreter

CC = gcc
CFLAGS = -Wall -Wextra -Werror -std=c99 -pedantic
CFLAGS_DEBUG = $(CFLAGS) -g -O0 -DDEBUG
CFLAGS_RELEASE = $(CFLAGS) -O3 -DNDEBUG

# Source files
SRCS = vm_interpreter.c
HDRS = vm_interpreter.h opcodes.h
TEST_SRC = test_interpreter.c

# Output
TEST_BIN = test_vm
OBJ = vm_interpreter.o

.PHONY: all clean test debug release size

# Default target: build and run tests
all: test

# Build and run tests
test: $(TEST_BIN)
	./$(TEST_BIN)

# Build test binary (debug mode)
$(TEST_BIN): $(TEST_SRC) $(SRCS) $(HDRS)
	$(CC) $(CFLAGS_DEBUG) -o $@ $(TEST_SRC) $(SRCS)

# Build in debug mode
debug: $(TEST_SRC) $(SRCS) $(HDRS)
	$(CC) $(CFLAGS_DEBUG) -o $(TEST_BIN) $(TEST_SRC) $(SRCS)

# Build in release mode (optimized)
release: $(TEST_SRC) $(SRCS) $(HDRS)
	$(CC) $(CFLAGS_RELEASE) -o $(TEST_BIN) $(TEST_SRC) $(SRCS)

# Build object file only (for size check)
$(OBJ): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS_RELEASE) -c -o $@ vm_interpreter.c

# Check object file size
size: $(OBJ)
	@echo "Object file size:"
	@ls -lh $(OBJ)
	@echo ""
	@echo "Section breakdown:"
	@size $(OBJ) || true

# Clean build artifacts
clean:
	rm -f $(TEST_BIN) $(OBJ)
	rm -rf *.dSYM

# Compile check only (no linking)
check: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) -fsyntax-only $(SRCS)
	@echo "Syntax check passed!"

# Run with valgrind (if available)
valgrind: $(TEST_BIN)
	valgrind --leak-check=full --error-exitcode=1 ./$(TEST_BIN)

# Help
help:
	@echo "VM Interpreter Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build and run tests (default)"
	@echo "  test     - Build and run tests"
	@echo "  debug    - Build with debug symbols"
	@echo "  release  - Build with optimizations"
	@echo "  size     - Show object file size"
	@echo "  check    - Syntax check only"
	@echo "  clean    - Remove build artifacts"
	@echo "  valgrind - Run tests under valgrind"
	@echo "  help     - Show this help"
