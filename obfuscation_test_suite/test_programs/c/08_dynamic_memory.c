/**
 * Test Program 8: Dynamic Memory Allocation
 * Purpose: Test dynamic memory and pointer operations
 * Expected output: Array operations with dynamic memory
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    int n = 10;
    int *arr = (int *)malloc(n * sizeof(int));
    int i, sum = 0;

    // Initialize array
    for (i = 0; i < n; i++) {
        arr[i] = (i + 1) * (i + 1);
        sum += arr[i];
    }

    printf("Array: ");
    for (i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    printf("Sum: %d\n", sum);
    printf("Average: %.2f\n", (float)sum / n);

    // String operations with dynamic memory
    char *str = (char *)malloc(256);
    strcpy(str, "Dynamic string allocation");
    printf("String: %s\n", str);
    printf("String length: %lu\n", strlen(str));

    free(arr);
    free(str);

    return 0;
}
