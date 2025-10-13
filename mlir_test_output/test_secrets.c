/**
 * End-to-End MLIR Obfuscation Pipeline Test
 * This file contains sensitive data and complex control flow to test all layers
 */
#include <stdio.h>
#include <string.h>

// LAYER 3 TEST: These secrets should be encrypted by string-encrypt pass
const char* MASTER_PASSWORD = "SuperSecretPassword2024!";
const char* API_KEY = "sk_live_abc123_secret_key_xyz789";
const char* DATABASE_URL = "postgresql://admin:secret@localhost:5432/db";
const int MAGIC_NUMBER = 0xDEADBEEF;
const float ENCRYPTION_FACTOR = 3.14159f;

// LAYER 2 TEST: These functions should be renamed by symbol-obfuscate/crypto-hash pass
int validate_password(const char* input) {
    return strcmp(input, MASTER_PASSWORD) == 0;
}

int check_api_key(const char* key) {
    return strcmp(key, API_KEY) == 0;
}

int process_magic(int value) {
    return value ^ MAGIC_NUMBER;
}

float compute_encrypted_value(float input) {
    return input * ENCRYPTION_FACTOR;
}

// Complex function for control flow testing
int authenticate_user(const char* password, const char* key) {
    if (validate_password(password)) {
        if (check_api_key(key)) {
            int magic = process_magic(42);
            float encrypted = compute_encrypted_value((float)magic);
            printf("Access GRANTED! Magic: %d, Encrypted: %f\n", magic, encrypted);
            return 1;
        } else {
            printf("Invalid API key\n");
            return 0;
        }
    }
    printf("Invalid password\n");
    return 0;
}

int main() {
    printf("=== MLIR Obfuscation Test ===\n");
    printf("Testing authentication...\n");
    
    // This should work
    int result = authenticate_user("SuperSecretPassword2024!", "sk_live_abc123_secret_key_xyz789");
    printf("Authentication result: %d\n", result);
    
    // This should fail
    int bad_result = authenticate_user("wrong_password", "wrong_key");
    printf("Bad auth result: %d\n", bad_result);
    
    printf("Test complete!\n");
    return 0;
}
