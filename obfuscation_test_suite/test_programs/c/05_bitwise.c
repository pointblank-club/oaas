/**
 * Test Program 5: Bitwise Operations
 * Purpose: Test bitwise manipulation and control flow
 * Expected output: Results of bitwise operations
 */
#include <stdio.h>

int main() {
    unsigned int a = 0xDEADBEEF;
    unsigned int b = 0xCAFEBABE;
    unsigned int c, d, e, f;
    int i, count;

    c = a & b;
    d = a | b;
    e = a ^ b;
    f = ~a;

    printf("a: 0x%X\n", a);
    printf("b: 0x%X\n", b);
    printf("AND: 0x%X\n", c);
    printf("OR: 0x%X\n", d);
    printf("XOR: 0x%X\n", e);
    printf("NOT: 0x%X\n", f);

    // Count set bits
    count = 0;
    for (i = 0; i < 32; i++) {
        if (a & (1 << i)) count++;
    }
    printf("Bits in a: %d\n", count);

    return 0;
}
