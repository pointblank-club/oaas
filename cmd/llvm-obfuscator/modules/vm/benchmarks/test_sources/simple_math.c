/**
 * Simple Math Functions for VM Benchmark Testing.
 *
 * These functions are simple enough to be fully virtualized
 * by the Level 1 VM (add, sub, xor only, no branches/calls).
 */

#include <stdint.h>

/* Basic arithmetic - perfect for VM */
int32_t add_numbers(int32_t a, int32_t b) {
    return a + b;
}

int32_t sub_numbers(int32_t a, int32_t b) {
    return a - b;
}

int32_t xor_numbers(int32_t a, int32_t b) {
    return a ^ b;
}

/* Chained operations */
int32_t add_sub(int32_t a, int32_t b, int32_t c) {
    int32_t sum = a + b;
    return sum - c;
}

int32_t xor_chain(int32_t a, int32_t b, int32_t c) {
    int32_t x1 = a ^ b;
    return x1 ^ c;
}

/* Mixed operations */
int32_t mixed_ops(int32_t a, int32_t b, int32_t c, int32_t d) {
    int32_t t1 = a + b;
    int32_t t2 = c - d;
    return t1 ^ t2;
}

/* Identity test functions */
int32_t add_zero(int32_t a) {
    return a + 0;
}

int32_t xor_zero(int32_t a) {
    return a ^ 0;
}

/* Self-canceling operations */
int32_t xor_self(int32_t a) {
    return a ^ a;  /* Always 0 */
}

int32_t sub_self(int32_t a) {
    return a - a;  /* Always 0 */
}

/* Multiple return paths - single basic block only */
int32_t simple_expr(int32_t x, int32_t y, int32_t z) {
    int32_t a = x + y;
    int32_t b = a - z;
    int32_t c = b ^ x;
    return c;
}

/*
 * Test harness - not virtualized, uses printf
 */
#ifdef STANDALONE_TEST
#include <stdio.h>

int main(void) {
    printf("add_numbers(5, 3) = %d (expected: 8)\n", add_numbers(5, 3));
    printf("sub_numbers(10, 4) = %d (expected: 6)\n", sub_numbers(10, 4));
    printf("xor_numbers(255, 15) = %d (expected: 240)\n", xor_numbers(255, 15));
    printf("add_sub(10, 5, 3) = %d (expected: 12)\n", add_sub(10, 5, 3));
    printf("xor_chain(1, 2, 4) = %d (expected: 7)\n", xor_chain(1, 2, 4));
    printf("mixed_ops(5, 3, 10, 4) = %d (expected: 14)\n", mixed_ops(5, 3, 10, 4));
    printf("add_zero(42) = %d (expected: 42)\n", add_zero(42));
    printf("xor_zero(42) = %d (expected: 42)\n", xor_zero(42));
    printf("xor_self(42) = %d (expected: 0)\n", xor_self(42));
    printf("sub_self(42) = %d (expected: 0)\n", sub_self(42));
    printf("simple_expr(1, 2, 3) = %d (expected: 0)\n", simple_expr(1, 2, 3));
    return 0;
}
#endif
