/**
 * Test Program 5: Exception Handling
 * Purpose: Test C++ exception mechanisms
 * Expected output: Exception handling results
 */
#include <iostream>
#include <stdexcept>
#include <string>

class Calculator {
public:
    double divide(double a, double b) {
        if (b == 0) {
            throw std::invalid_argument("Division by zero");
        }
        return a / b;
    }

    int modulo(int a, int b) {
        if (b == 0) {
            throw std::domain_error("Modulo by zero");
        }
        return a % b;
    }
};

int main() {
    Calculator calc;

    // Valid operations
    try {
        std::cout << "10 / 2 = " << calc.divide(10, 2) << std::endl;
        std::cout << "7 % 3 = " << calc.modulo(7, 3) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    // Division by zero
    try {
        std::cout << "10 / 0 = " << calc.divide(10, 0) << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }

    // Modulo by zero
    try {
        std::cout << "7 % 0 = " << calc.modulo(7, 0) << std::endl;
    } catch (const std::domain_error& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }

    std::cout << "Program completed successfully" << std::endl;
    return 0;
}
