#include <stdio.h>
#include <string.h>

int main() {
    const char* msg = "Header fix works!";
    printf("%s (length: %zu)\n", msg, strlen(msg));
    return 0;
}
