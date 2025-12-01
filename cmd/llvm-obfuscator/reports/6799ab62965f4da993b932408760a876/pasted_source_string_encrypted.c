/**
 * Test program for header obfuscation
 * Contains both stdlib and custom function calls
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <stdlib.h>
#include <string.h>

/* XOR String Decryption Helper */
static char* _xor_decrypt(const unsigned char* enc, int len, unsigned char key) {
    char* dec = (char*)malloc(len + 1);
    if (!dec) return NULL;
    for (int i = 0; i < len; i++) {
        dec[i] = enc[i] ^ key;
    }
    dec[len] = '\0';
    return dec;
}

static void _secure_free(char* ptr) {
    if (ptr) {
        memset(ptr, 0, strlen(ptr));
        free(ptr);
    }
}


/* Indirect Call Obfuscation - Function Pointers */

/* Forward declarations for custom functions */
int calculate_score(int a, int b);
int validate_password(const char* input);

/* Function pointer declarations */
static int (*__fptr_calculate_score)(int a, int b) = NULL;
static void (*__fptr_free)(void*) = NULL;
static void* (*__fptr_malloc)(size_t) = NULL;
static int (*__fptr_printf)(const char*, ...) = NULL;
static int (*__fptr_strcmp)(const char*, const char*) = NULL;
static char* (*__fptr_strcpy)(char*, const char*) = NULL;

/* String decryption initialization (runs before main) */
__attribute__((constructor)) static void _init_encrypted_strings(void) {
    correct = _xor_decrypt((const unsigned char[]){0x0d,0x3b,0x3d,0x2c,0x3b,0x2a,0x0e,0x3f,0x2d,0x2d,0x6f,0x6c,0x6d}, 13, 0x5e);
}

static size_t (*__fptr_strlen)(const char*) = NULL;
static int (*__fptr_validate_password)(const char* input) = NULL;

/* Initialize function pointers */
__attribute__((constructor)) static void __init_function_pointers(void) {
    __fptr_calculate_score = (void*)&calculate_score;
    __fptr_free = (void*)&free;
    __fptr_malloc = (void*)&malloc;
    __fptr_printf = (void*)&printf;
    __fptr_strcmp = (void*)&strcmp;
    __fptr_strcpy = (void*)&strcpy;
    __fptr_strlen = (void*)&strlen;
    __fptr_validate_password = (void*)&validate_password;
}


// Custom function
int validate_password(const char* input) {
char* correct = NULL;
    return __fptr_strcmp(input, correct) == 0;
}

// Custom function
int calculate_score(int a, int b) {
    return a * b + 42;
}

int main() {
    // Standard library calls that should be obfuscated
    __fptr_printf("Testing header obfuscation\n");

    char* buffer = (char*)__fptr_malloc(100);
    __fptr_strcpy(buffer, _xor_decrypt((const unsigned char[]){0xde,0xf3,0xfa,0xfa,0xf9}, 5, 0x96));

    int len = __fptr_strlen(buffer);
    __fptr_printf("Length: %d\n", len);

    // Custom function calls that should be obfuscated
    int valid = __fptr_validate_password(_xor_decrypt((const unsigned char[]){0xeb,0xce,0xd3,0xd2,0xdb,0xec,0xdd,0xcf,0xcf,0xcb,0xd3,0xce,0xd8}, 13, 0xbc));
    __fptr_printf("Password valid: %d\n", valid);

    int score = __fptr_calculate_score(10, 20);
    __fptr_printf("Score: %d\n", score);

    __fptr_free(buffer);
    return 0;
}
