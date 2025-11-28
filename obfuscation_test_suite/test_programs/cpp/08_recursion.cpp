/**
 * Test Program 8: Recursive Functions
 * Purpose: Test recursive call chains
 * Expected output: Recursive computation results
 */
#include <iostream>

long long factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int ackermann(int m, int n) {
    if (m == 0) return n + 1;
    if (n == 0) return ackermann(m - 1, 1);
    return ackermann(m - 1, ackermann(m, n - 1));
}

int main() {
    // Factorial
    std::cout << "Factorials:" << std::endl;
    for (int i = 0; i <= 10; i++) {
        std::cout << i << "! = " << factorial(i) << std::endl;
    }

    std::cout << "\nAckermann function:" << std::endl;
    for (int m = 0; m <= 3; m++) {
        for (int n = 0; n <= 3; n++) {
            if (m <= 2 || n <= 1) {  // Limit to avoid long computation
                std::cout << "A(" << m << "," << n << ") = " << ackermann(m, n) << std::endl;
            }
        }
    }

    return 0;
}
