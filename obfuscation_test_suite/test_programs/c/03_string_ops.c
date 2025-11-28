/**
 * Test Program 3: String Operations
 * Purpose: Test string manipulation and plaintext detection
 * Expected output: Various string operations results
 */
#include <stdio.h>
#include <string.h>
#include <ctype.h>

int main() {
    char str[256] = "The Quick Brown Fox Jumps Over The Lazy Dog";
    char password[256] = "SecurePassword123!@#";
    char encrypted[256];
    int i;

    printf("Original: %s\n", str);
    printf("Length: %lu\n", strlen(str));

    // Simple XOR encryption for obfuscation testing
    strcpy(encrypted, password);
    for (i = 0; i < (int)strlen(password); i++) {
        encrypted[i] ^= 0xAA;
    }

    printf("Password length: %lu\n", strlen(password));

    // Count letters, digits
    int letters = 0, digits = 0;
    for (i = 0; i < (int)strlen(str); i++) {
        if (isalpha(str[i])) letters++;
        if (isdigit(str[i])) digits++;
    }
    printf("Letters: %d, Digits: %d\n", letters, digits);

    return 0;
}
