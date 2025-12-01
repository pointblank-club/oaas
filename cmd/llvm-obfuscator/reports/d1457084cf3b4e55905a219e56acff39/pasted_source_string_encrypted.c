#include <stdio.h>
#include <string.h>

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


int main() {
char* msg = NULL;
    printf("%s (length: %zu)\n", msg, strlen(msg));
    return 0;
}


/* String decryption initialization (runs before main) */
__attribute__((constructor)) static void _init_encrypted_strings(void) {
    msg = _xor_decrypt((const unsigned char[]){0x63,0x4e,0x4a,0x4f,0x4e,0x59,0x0b,0x4d,0x42,0x53,0x0b,0x5c,0x44,0x59,0x40,0x58,0x0a}, 17, 0x2b);
}
