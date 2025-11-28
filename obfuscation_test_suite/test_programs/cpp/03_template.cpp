/**
 * Test Program 3: Template Functions
 * Purpose: Test C++ template instantiation
 * Expected output: Generic function results
 */
#include <iostream>

template <typename T>
T add(T a, T b) {
    return a + b;
}

template <typename T>
T multiply(T a, T b) {
    return a * b;
}

template <typename T>
T findMax(T arr[], int size) {
    T max = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}

int main() {
    // Integer operations
    std::cout << "Add (5 + 3): " << add(5, 3) << std::endl;
    std::cout << "Multiply (5 * 3): " << multiply(5, 3) << std::endl;

    // Double operations
    std::cout << "Add (5.5 + 2.3): " << add(5.5, 2.3) << std::endl;

    // String operations
    std::cout << "Add (\"Hello\" + \" World\"): " << add(std::string("Hello"), std::string(" World")) << std::endl;

    // Find max
    int intArr[] = {5, 12, 8, 3, 15, 2};
    double doubleArr[] = {3.5, 8.9, 2.1, 9.8};

    std::cout << "Max int: " << findMax(intArr, 6) << std::endl;
    std::cout << "Max double: " << findMax(doubleArr, 4) << std::endl;

    return 0;
}
