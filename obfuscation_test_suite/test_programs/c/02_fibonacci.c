/**
 * Test Program 2: Fibonacci Sequence
 * Purpose: Test recursive algorithms
 * Expected output: Fibonacci sequence up to 10 terms
 */
#include <stdio.h>

int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int main() {
    int i;
    for (i = 0; i < 10; i++) {
        printf("%d ", fib(i));
    }
    printf("\n");
    return 0;
}
