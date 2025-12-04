// Simple authentication example for Polygeist obfuscation testing
// This demonstrates how Polygeist preserves high-level constructs

#include <stdio.h>
#include <string.h>

#define MAX_ATTEMPTS 3
#define PASSWORD "secret123"

int failed_attempts = 0;

// Validate user password
int validate_password(const char *input) {
    if (strcmp(input, PASSWORD) == 0) {
        failed_attempts = 0;
        return 1;  // Success
    }

    failed_attempts++;

    if (failed_attempts >= MAX_ATTEMPTS) {
        printf("Account locked!\n");
        return -1;  // Locked
    }

    return 0;  // Failed
}

// Check authentication status
int check_auth_status(void) {
    if (failed_attempts >= MAX_ATTEMPTS) {
        return -1;  // Locked
    }
    return 0;  // OK
}

// Main authentication loop
int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <password>\n", argv[0]);
        return 1;
    }

    // Check if account is locked
    if (check_auth_status() < 0) {
        printf("Authentication disabled\n");
        return 1;
    }

    // Validate password
    int result = validate_password(argv[1]);

    if (result == 1) {
        printf("Access granted!\n");
        return 0;
    } else if (result == -1) {
        printf("Account locked!\n");
        return 1;
    } else {
        printf("Access denied! (%d/%d attempts)\n",
               failed_attempts, MAX_ATTEMPTS);
        return 1;
    }
}
