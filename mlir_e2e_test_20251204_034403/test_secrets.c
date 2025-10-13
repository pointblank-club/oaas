/**
 * OAAS MLIR Obfuscation E2E Test Program
 * This file tests all layers of the obfuscation pipeline:
 * - String encryption
 * - Symbol obfuscation  
 * - OLLVM passes (control flow flattening, bogus CF)
 * - LLVM compiler flags
 */
#include <stdio.h>
#include <string.h>

// SECRET STRINGS - These should be encrypted by Layer 3a (string-encrypt)
const char* MASTER_PASSWORD = "TopSecret2024!";
const char* API_KEY = "sk_live_xyz123_secret";
const char* DB_CONN = "postgres://admin:password@db/main";

// SECRET CONSTANTS - These should be obfuscated
const int MAGIC_NUMBER = 0xDEADBEEF;
const float PI_FACTOR = 3.14159f;

// FUNCTION 1 - Should be renamed by Layer 3b (symbol-obfuscate)
int validate_password(const char* input) {
    return strcmp(input, MASTER_PASSWORD) == 0;
}

// FUNCTION 2 - Should be renamed
int check_api_key(const char* key) {
    return strcmp(key, API_KEY) == 0;
}

// FUNCTION 3 - Should be renamed
int process_magic(int value) {
    return value ^ MAGIC_NUMBER;
}

// FUNCTION 4 - Should be renamed
float compute_value(float input) {
    return input * PI_FACTOR;
}

// MAIN AUTHENTICATION FUNCTION - Complex control flow
int authenticate_user(const char* password, const char* key) {
    // Control flow that will be flattened by OLLVM
    if (validate_password(password)) {
        if (check_api_key(key)) {
            int magic = process_magic(42);
            float result = compute_value((float)magic);
            printf("ACCESS GRANTED! Result: %f\n", result);
            return 1;
        } else {
            printf("Invalid API key!\n");
            return 0;
        }
    }
    printf("Invalid password!\n");
    return 0;
}

int main(int argc, char** argv) {
    printf("=== OAAS E2E Test ===\n");
    
    // Test with correct credentials
    int result1 = authenticate_user("TopSecret2024!", "sk_live_xyz123_secret");
    printf("Test 1 (valid): %d\n", result1);
    
    // Test with wrong credentials
    int result2 = authenticate_user("wrong", "wrong");
    printf("Test 2 (invalid): %d\n", result2);
    
    return (result1 == 1 && result2 == 0) ? 0 : 1;
}
