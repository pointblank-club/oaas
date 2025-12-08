/**
 * VM Interpreter Test Suite
 *
 * Standalone test program for the VM interpreter.
 * Compile with: gcc -Wall -Wextra -std=c99 -o test_vm test_interpreter.c vm_interpreter.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vm_interpreter.h"
#include "opcodes.h"

/* Test result tracking */
static int tests_run = 0;
static int tests_passed = 0;

/* Test assertion macro */
#define TEST_ASSERT(condition, msg) do { \
    tests_run++; \
    if (condition) { \
        tests_passed++; \
        printf("  [PASS] %s\n", msg); \
    } else { \
        printf("  [FAIL] %s\n", msg); \
    } \
} while(0)

/* ========================================================================
 * Test Cases
 * ======================================================================== */

/**
 * Test: ADD operation
 * Bytecode: load arg[0], load arg[1], add, return
 * Expected: 5 + 3 = 8
 */
static void test_add(void) {
    printf("\nTest: ADD\n");

    uint8_t bytecode[] = {
        VM_LOAD_ARG, 0x00,  /* push args[0] (5) */
        VM_LOAD_ARG, 0x01,  /* push args[1] (3) */
        VM_ADD,             /* pop two, push sum */
        VM_RET              /* return top */
    };

    int64_t args[] = {5, 3};
    int64_t result = vm_execute(bytecode, sizeof(bytecode), args, 2);

    TEST_ASSERT(result == 8, "5 + 3 = 8");
}

/**
 * Test: SUB operation
 * Bytecode: load arg[0], load arg[1], sub, return
 * Expected: 10 - 4 = 6
 */
static void test_sub(void) {
    printf("\nTest: SUB\n");

    uint8_t bytecode[] = {
        VM_LOAD_ARG, 0x00,  /* push args[0] (10) */
        VM_LOAD_ARG, 0x01,  /* push args[1] (4) */
        VM_SUB,             /* pop two, push difference */
        VM_RET              /* return top */
    };

    int64_t args[] = {10, 4};
    int64_t result = vm_execute(bytecode, sizeof(bytecode), args, 2);

    TEST_ASSERT(result == 6, "10 - 4 = 6");
}

/**
 * Test: XOR operation
 * Bytecode: load arg[0], load arg[1], xor, return
 * Expected: 255 ^ 15 = 240
 */
static void test_xor(void) {
    printf("\nTest: XOR\n");

    uint8_t bytecode[] = {
        VM_LOAD_ARG, 0x00,  /* push args[0] (255) */
        VM_LOAD_ARG, 0x01,  /* push args[1] (15) */
        VM_XOR,             /* pop two, push xor */
        VM_RET              /* return top */
    };

    int64_t args[] = {255, 15};
    int64_t result = vm_execute(bytecode, sizeof(bytecode), args, 2);

    TEST_ASSERT(result == 240, "255 ^ 15 = 240 (0xF0)");
}

/**
 * Test: Complex expression with immediate
 * Bytecode: load arg[0], load arg[1], add, push 10, add, return
 * Expected: (5 + 3) + 10 = 18
 */
static void test_complex(void) {
    printf("\nTest: Complex (a + b + 10)\n");

    uint8_t bytecode[] = {
        VM_LOAD_ARG, 0x00,              /* push args[0] (5) */
        VM_LOAD_ARG, 0x01,              /* push args[1] (3) */
        VM_ADD,                          /* 5 + 3 = 8 */
        VM_PUSH, 0x0A, 0x00, 0x00, 0x00, /* push 10 */
        VM_ADD,                          /* 8 + 10 = 18 */
        VM_RET                           /* return 18 */
    };

    int64_t args[] = {5, 3};
    int64_t result = vm_execute(bytecode, sizeof(bytecode), args, 2);

    TEST_ASSERT(result == 18, "(5 + 3) + 10 = 18");
}

/**
 * Test: Virtual registers
 * Bytecode: store args to registers, load from registers, add
 * Expected: 7 + 8 = 15
 */
static void test_registers(void) {
    printf("\nTest: Virtual Registers\n");

    uint8_t bytecode[] = {
        VM_LOAD_ARG, 0x00,  /* push args[0] (7) */
        VM_STORE, 0x00,     /* store to vreg[0] */
        VM_LOAD_ARG, 0x01,  /* push args[1] (8) */
        VM_STORE, 0x01,     /* store to vreg[1] */
        VM_LOAD, 0x00,      /* load vreg[0] */
        VM_LOAD, 0x01,      /* load vreg[1] */
        VM_ADD,             /* add */
        VM_RET              /* return */
    };

    int64_t args[] = {7, 8};
    int64_t result = vm_execute(bytecode, sizeof(bytecode), args, 2);

    TEST_ASSERT(result == 15, "vreg[0]=7, vreg[1]=8, sum=15");
}

/**
 * Test: PUSH with negative value
 * Expected: Push and return -42
 */
static void test_push_negative(void) {
    printf("\nTest: PUSH Negative\n");

    /* -42 in little-endian two's complement (0xFFFFFFD6) */
    uint8_t bytecode[] = {
        VM_PUSH, 0xD6, 0xFF, 0xFF, 0xFF,  /* push -42 */
        VM_RET
    };

    int64_t result = vm_execute(bytecode, sizeof(bytecode), NULL, 0);

    TEST_ASSERT(result == -42, "push -42 returns -42");
}

/**
 * Test: NOP operation
 * Expected: NOP does nothing, still returns correct result
 */
static void test_nop(void) {
    printf("\nTest: NOP\n");

    uint8_t bytecode[] = {
        VM_NOP,
        VM_PUSH, 0x2A, 0x00, 0x00, 0x00,  /* push 42 */
        VM_NOP,
        VM_NOP,
        VM_RET
    };

    int64_t result = vm_execute(bytecode, sizeof(bytecode), NULL, 0);

    TEST_ASSERT(result == 42, "NOP doesn't affect result");
}

/**
 * Test: POP operation
 * Expected: Pop discards value
 */
static void test_pop(void) {
    printf("\nTest: POP\n");

    uint8_t bytecode[] = {
        VM_PUSH, 0x64, 0x00, 0x00, 0x00,  /* push 100 */
        VM_PUSH, 0x2A, 0x00, 0x00, 0x00,  /* push 42 */
        VM_POP,                            /* discard 42 */
        VM_RET                             /* return 100 */
    };

    int64_t result = vm_execute(bytecode, sizeof(bytecode), NULL, 0);

    TEST_ASSERT(result == 100, "POP discards top, returns 100");
}

/**
 * Test: Invalid opcode handling
 * Expected: Returns error code
 */
static void test_invalid_opcode(void) {
    printf("\nTest: Invalid Opcode\n");

    uint8_t bytecode[] = {
        0xEE,  /* Invalid opcode */
        VM_RET
    };

    int64_t result = vm_execute(bytecode, sizeof(bytecode), NULL, 0);

    TEST_ASSERT(result == VM_ERR_INVALID_OPCODE, "Invalid opcode returns error");
}

/**
 * Test: Stack underflow handling
 * Expected: Returns error code
 */
static void test_stack_underflow(void) {
    printf("\nTest: Stack Underflow\n");

    uint8_t bytecode[] = {
        VM_ADD,  /* Try to add with empty stack */
        VM_RET
    };

    int64_t result = vm_execute(bytecode, sizeof(bytecode), NULL, 0);

    TEST_ASSERT(result == VM_ERR_STACK_UNDERFLOW, "Stack underflow returns error");
}

/**
 * Test: Invalid argument index
 * Expected: Returns error code
 */
static void test_invalid_arg(void) {
    printf("\nTest: Invalid Argument Index\n");

    uint8_t bytecode[] = {
        VM_LOAD_ARG, 0x05,  /* Try to load arg[5] when only 2 args */
        VM_RET
    };

    int64_t args[] = {1, 2};
    int64_t result = vm_execute(bytecode, sizeof(bytecode), args, 2);

    TEST_ASSERT(result == VM_ERR_INVALID_ARG, "Invalid arg index returns error");
}

/**
 * Test: Invalid register index
 * Expected: Returns error code
 */
static void test_invalid_reg(void) {
    printf("\nTest: Invalid Register Index\n");

    uint8_t bytecode[] = {
        VM_LOAD, 0x0A,  /* Try to load vreg[10] (only 0-7 valid) */
        VM_RET
    };

    int64_t result = vm_execute(bytecode, sizeof(bytecode), NULL, 0);

    TEST_ASSERT(result == VM_ERR_INVALID_REG, "Invalid reg index returns error");
}

/**
 * Test: Null bytecode handling
 * Expected: Returns error code
 */
static void test_null_bytecode(void) {
    printf("\nTest: Null Bytecode\n");

    int64_t result = vm_execute(NULL, 0, NULL, 0);

    TEST_ASSERT(result == VM_ERR_NULL_BYTECODE, "Null bytecode returns error");
}

/**
 * Test: Empty bytecode (just RET)
 * Expected: Returns 0 (empty stack)
 */
static void test_empty_ret(void) {
    printf("\nTest: Empty RET\n");

    uint8_t bytecode[] = {
        VM_RET
    };

    int64_t result = vm_execute(bytecode, sizeof(bytecode), NULL, 0);

    TEST_ASSERT(result == 0, "Empty stack returns 0");
}

/**
 * Test: Multiple operations in sequence
 * Bytecode: ((a + b) - c) ^ d
 * Expected: ((10 + 5) - 3) ^ 7 = 12 ^ 7 = 11
 */
static void test_multiple_ops(void) {
    printf("\nTest: Multiple Operations\n");

    uint8_t bytecode[] = {
        VM_LOAD_ARG, 0x00,  /* push 10 */
        VM_LOAD_ARG, 0x01,  /* push 5 */
        VM_ADD,             /* 10 + 5 = 15 */
        VM_LOAD_ARG, 0x02,  /* push 3 */
        VM_SUB,             /* 15 - 3 = 12 */
        VM_LOAD_ARG, 0x03,  /* push 7 */
        VM_XOR,             /* 12 ^ 7 = 11 */
        VM_RET
    };

    int64_t args[] = {10, 5, 3, 7};
    int64_t result = vm_execute(bytecode, sizeof(bytecode), args, 4);

    TEST_ASSERT(result == 11, "((10+5)-3)^7 = 11");
}

/**
 * Test: Context size verification
 * Expected: VMContext < 4KB
 */
static void test_context_size(void) {
    printf("\nTest: Context Size\n");

    size_t size = sizeof(VMContext);
    printf("  VMContext size: %zu bytes\n", size);

    TEST_ASSERT(size < 4096, "VMContext fits in 4KB");
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(void) {
    printf("==========================================\n");
    printf("VM Interpreter Test Suite\n");
    printf("==========================================\n");

    /* Run all tests */
    test_add();
    test_sub();
    test_xor();
    test_complex();
    test_registers();
    test_push_negative();
    test_nop();
    test_pop();
    test_invalid_opcode();
    test_stack_underflow();
    test_invalid_arg();
    test_invalid_reg();
    test_null_bytecode();
    test_empty_ret();
    test_multiple_ops();
    test_context_size();

    /* Print summary */
    printf("\n==========================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("==========================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
