/**
 * Test program for header obfuscation
 * Contains both stdlib and custom function calls
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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
    const char* correct = "SecretPass123";
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
    __fptr_strcpy(buffer, "Hello");

    int len = __fptr_strlen(buffer);
    __fptr_printf("Length: %d\n", len);

    // Custom function calls that should be obfuscated
    int valid = __fptr_validate_password("WrongPassword");
    __fptr_printf("Password valid: %d\n", valid);

    int score = __fptr_calculate_score(10, 20);
    __fptr_printf("Score: %d\n", score);

    __fptr_free(buffer);
    return 0;
}
