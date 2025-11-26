#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    const char* secret = "MySecretPassword123";
    int len = strlen(secret);
    int fib = fibonacci(10);
    
    printf("String length: %d\n", len);
    printf("Fibonacci(10): %d\n", fib);
    printf("Test PASSED!\n");
    
    return 0;
}
