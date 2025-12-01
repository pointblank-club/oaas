/**
 * Test Program 10: Advanced Algorithms
 * Purpose: Test complex algorithmic implementations
 * Expected output: Algorithm results
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

class AlgorithmTest {
public:
    // Binary search
    static bool binarySearch(std::vector<int>& arr, int target) {
        return std::binary_search(arr.begin(), arr.end(), target);
    }

    // Merge sorted arrays
    static std::vector<int> mergeSorted(const std::vector<int>& a, const std::vector<int>& b) {
        std::vector<int> result;
        std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
        return result;
    }

    // Calculate statistics
    static void printStats(const std::vector<int>& arr) {
        if (arr.empty()) return;
        int sum = std::accumulate(arr.begin(), arr.end(), 0);
        int min = *std::min_element(arr.begin(), arr.end());
        int max = *std::max_element(arr.begin(), arr.end());

        std::cout << "Sum: " << sum << ", Min: " << min << ", Max: " << max;
        std::cout << ", Avg: " << (double)sum / arr.size() << std::endl;
    }
};

int main() {
    std::vector<int> arr1 = {1, 3, 5, 7, 9};
    std::vector<int> arr2 = {2, 4, 6, 8, 10};

    std::cout << "Array 1: ";
    for (int x : arr1) std::cout << x << " ";
    std::cout << std::endl;

    std::cout << "Array 2: ";
    for (int x : arr2) std::cout << x << " ";
    std::cout << std::endl;

    // Binary search test
    std::cout << "Search for 5 in arr1: " << (AlgorithmTest::binarySearch(arr1, 5) ? "Found" : "Not found") << std::endl;
    std::cout << "Search for 6 in arr1: " << (AlgorithmTest::binarySearch(arr1, 6) ? "Found" : "Not found") << std::endl;

    // Merge test
    std::vector<int> merged = AlgorithmTest::mergeSorted(arr1, arr2);
    std::cout << "Merged: ";
    for (int x : merged) std::cout << x << " ";
    std::cout << std::endl;

    // Statistics
    std::cout << "Statistics for merged: ";
    AlgorithmTest::printStats(merged);

    return 0;
}
