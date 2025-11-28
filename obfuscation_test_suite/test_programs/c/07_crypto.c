/**
 * Test Program 7: Simple Encryption
 * Purpose: Test string handling and cryptographic operations
 * Expected output: Encrypted and decrypted text
 */
#include <stdio.h>
#include <string.h>

void xor_encrypt(char *plaintext, char *key, char *ciphertext) {
    int i, key_len = strlen(key);
    for (i = 0; plaintext[i] != '\0'; i++) {
        ciphertext[i] = plaintext[i] ^ key[i % key_len];
    }
    ciphertext[i] = '\0';
}

int main() {
    char plaintext[] = "This is a secret message for testing obfuscation";
    char key[] = "MySecretKey";
    char ciphertext[256];
    char decrypted[256];

    // Encrypt
    xor_encrypt(plaintext, key, ciphertext);
    printf("Plaintext: %s\n", plaintext);
    printf("Ciphertext length: %lu\n", strlen(ciphertext));

    // Decrypt
    xor_encrypt(ciphertext, key, decrypted);
    printf("Decrypted: %s\n", decrypted);
    printf("Match: %d\n", strcmp(plaintext, decrypted) == 0 ? 1 : 0);

    return 0;
}
