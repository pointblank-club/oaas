#include <stdio.h>
#include <string.h>

const char* SECRET_PASSWORD = "MySecret123!";
const char* API_KEY = "sk_live_12345";
const char* DATABASE_URL = "postgres://admin:pass@localhost/db";

int validate_password(const char* input) {
    return strcmp(input, SECRET_PASSWORD) == 0;
}

int check_api_key(const char* key) {
    return strcmp(key, API_KEY) == 0;
}

int main() {
    printf("Testing obfuscation\n");
    
    if (validate_password("MySecret123!")) {
        printf("Password valid\n");
    }
    
    if (check_api_key("sk_live_12345")) {
        printf("API key valid\n");
    }
    
    return 0;
}