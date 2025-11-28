/**
 * Test Program 9: String Manipulation
 * Purpose: Test string operations and plaintext detection
 * Expected output: String operation results
 */
#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>

int main() {
    std::string text = "The Quick Brown Fox Jumps Over The Lazy Dog";
    std::string secret = "SecurePassword123!@#";

    std::cout << "Original: " << text << std::endl;
    std::cout << "Length: " << text.length() << std::endl;

    // Uppercase
    std::string upper = text;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    std::cout << "Uppercase: " << upper << std::endl;

    // Lowercase
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    std::cout << "Lowercase: " << lower << std::endl;

    // Replace
    std::string replaced = text;
    size_t pos = replaced.find("Fox");
    if (pos != std::string::npos) {
        replaced.replace(pos, 3, "Cat");
    }
    std::cout << "Replaced: " << replaced << std::endl;

    // Substring
    std::string sub = text.substr(10, 5);
    std::cout << "Substring: " << sub << std::endl;

    // Character count
    int vowels = 0;
    for (char c : lower) {
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
            vowels++;
        }
    }
    std::cout << "Vowels: " << vowels << std::endl;

    // Reverse
    std::string reversed = text;
    std::reverse(reversed.begin(), reversed.end());
    std::cout << "Reversed: " << reversed << std::endl;

    return 0;
}
