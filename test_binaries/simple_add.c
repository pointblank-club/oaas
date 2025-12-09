#include <stdio.h>
#include <stdlib.h>

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

int main(int argc, char* argv[]) {
    int x = 10;
    int y = 20;

    int sum = add(x, y);
    int product = multiply(x, y);

    printf("Add: %d + %d = %d\n", x, y, sum);
    printf("Multiply: %d * %d = %d\n", x, y, product);

    return 0;
}
