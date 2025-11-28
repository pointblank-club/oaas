/**
 * Test Program 4: STL Containers and Algorithms
 * Purpose: Test STL usage and algorithm application
 * Expected output: Container operations results
 */
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>

int main() {
    // Vector operations
    std::vector<int> numbers;
    for (int i = 1; i <= 10; i++) {
        numbers.push_back(i * i);
    }

    std::cout << "Vector: ";
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Vector algorithms
    std::sort(numbers.rbegin(), numbers.rend());
    std::cout << "Sorted (descending): ";
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Map operations
    std::map<std::string, int> ages;
    ages["Alice"] = 28;
    ages["Bob"] = 35;
    ages["Charlie"] = 22;
    ages["Diana"] = 30;

    std::cout << "Ages:" << std::endl;
    for (const auto& pair : ages) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // Find operation
    if (ages.find("Bob") != ages.end()) {
        std::cout << "Bob is " << ages["Bob"] << " years old" << std::endl;
    }

    return 0;
}
