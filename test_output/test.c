#include <stdio.h>

const char* secret_message = "Hello from obfuscated binary!";
const int magic_number = 42;

int add_numbers(int a, int b) {
    return a + b;
}

int multiply(int x, int y) {
    return x * y;
}

int main() {
    printf("Message: %s\n", secret_message);
    printf("Magic: %d\n", magic_number);
    printf("Sum: %d\n", add_numbers(10, 20));
    printf("Product: %d\n", multiply(6, 7));
    return 0;
}
