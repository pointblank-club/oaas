/**
 * Test Program 9: Complex Control Flow
 * Purpose: Test nested loops and conditionals
 * Expected output: Pattern printing with complex control flow
 */
#include <stdio.h>

int main() {
    int i, j, n = 5;

    // Nested loop pattern
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= i; j++) {
            if (j % 2 == 0) {
                printf("* ");
            } else {
                printf("# ");
            }
        }
        printf("\n");
    }

    printf("\n");

    // Diamond pattern
    for (i = 0; i < n; i++) {
        for (j = 0; j < n - i; j++) printf(" ");
        for (j = 0; j < 2 * i + 1; j++) printf("*");
        printf("\n");
    }

    for (i = n - 2; i >= 0; i--) {
        for (j = 0; j < n - i; j++) printf(" ");
        for (j = 0; j < 2 * i + 1; j++) printf("*");
        printf("\n");
    }

    return 0;
}
