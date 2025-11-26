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
static int    (*__fptr_calculate_score)(int a, int b) = NULL;
static void   (*__fptr_free)(void*) = NULL;
static void*  (*__fptr_malloc)(size_t) = NULL;
static int    (*__fptr_printf)(const char*, ...) = NULL;
static int    (*__fptr_strcmp)(const char*, const char*) = NULL;
static char*  (*__fptr_strcpy)(char*, const char*) = NULL;
static size_t (*__fptr_strlen)(const char*) = NULL;
static int    (*__fptr_validate_password)(const char* input) = NULL;

/* Initialize function pointers */
__attribute__((constructor))
static void __init_function_pointers(void) {
    __fptr_calculate_score     = (void*)&calculate_score;
    __fptr_free                = (void*)&free;
    __fptr_malloc              = (void*)&malloc;
    __fptr_printf              = (void*)&printf;
    __fptr_strcmp              = (void*)&strcmp;
    __fptr_strcpy              = (void*)&strcpy;
    __fptr_strlen              = (void*)&strlen;
    __fptr_validate_password   = (void*)&validate_password;
}

/* Custom function */
int validate_password(const char* input) {
    /* IMPORTANT:
       Declaration and assignment must be separate
       to avoid obfuscators deleting the variable. */
    const char* correct;
    correct = _xor_decrypt((const unsigned char[]){0x87,0xb1,0xb7,0xa6,0xb1,0xa0,0x84,0xb5,0xa7,0xa7,0xe5,0xe6,0xe7}, 13, 0xd4);

    return __fptr_strcmp(input, correct) == 0;
}

/* Custom function */
int calculate_score(int a, int b) {
    return a * b + 42;
}

int main() {
    __fptr_printf("Testing header obfuscation\n");

    char* buffer = (char*)__fptr_malloc(100);
    __fptr_strcpy(buffer, _xor_decrypt((const unsigned char[]){0x1e,0x33,0x3a,0x3a,0x39}, 5, 0x56));

    int len = __fptr_strlen(buffer);
    __fptr_printf("Length: %d\n", len);

    int valid = __fptr_validate_password(_xor_decrypt((const unsigned char[]){0x34,0x11,0x0c,0x0d,0x04,0x33,0x02,0x10,0x10,0x14,0x0c,0x11,0x07}, 13, 0x63));
    __fptr_printf("Password valid: %d\n", valid);

    int score = __fptr_calculate_score(10, 20);
    __fptr_printf("Score: %d\n", score);

    __fptr_free(buffer);
    return 0;
}
