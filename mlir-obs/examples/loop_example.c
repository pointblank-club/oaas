// Loop-heavy example to demonstrate Polygeist's affine/SCF dialect advantages
// Polygeist can analyze and obfuscate these structured loops more effectively

//Simple code

#include <stdio.h>

#define SIZE 100

// Sum array elements (simple loop)
int sum_array(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

// Matrix multiplication (nested loops - Polygeist can use affine dialect)
void matrix_mult(int A[SIZE][SIZE], int B[SIZE][SIZE], int C[SIZE][SIZE], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Conditional loop (demonstrates SCF dialect)
int count_positives(int *arr, int size) {
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] > 0) {
            count++;
        }
    }
    return count;
}

// While loop example
int find_first_zero(int *arr, int size) {
    int i = 0;
    while (i < size && arr[i] != 0) {
        i++;
    }
    return (i < size) ? i : -1;
}

int main(void) {
    int arr[10] = {1, -2, 3, -4, 5, 0, 7, -8, 9, -10};

    printf("Sum: %d\n", sum_array(arr, 10));
    printf("Positives: %d\n", count_positives(arr, 10));
    printf("First zero at: %d\n", find_first_zero(arr, 10));

    return 0;
}
