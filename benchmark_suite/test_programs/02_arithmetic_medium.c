/**
 * Test Program 2: Arithmetic Operations
 * Purpose: Test basic control flow and arithmetic obfuscation
 * Expected output: Result values that can be validated
 */
#include <stdio.h>

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int main() {
    int fib_result = fibonacci(10);
    int fact_result = factorial(5);
    int gcd_result = gcd(48, 18);

    printf("Fibonacci(10): %d\n", fib_result);
    printf("Factorial(5): %d\n", fact_result);
    printf("GCD(48,18): %d\n", gcd_result);

    return 0;
}
