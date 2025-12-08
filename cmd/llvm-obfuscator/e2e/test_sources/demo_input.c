/**
 * Demo Input for OAAS E2E Test
 * Simple arithmetic functions suitable for VM virtualization.
 */
#include <stdio.h>

int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }

int main() {
    printf("Result: %d\n", add(5, 3));
    return 0;
}
