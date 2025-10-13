// Simple test without vararg functions
const char* SECRET = "MyPassword123";
int magic_value = 0xDEADBEEF;

int validate(const char* input, const char* secret) {
    while (*input && *secret) {
        if (*input != *secret) return 0;
        input++;
        secret++;
    }
    return *input == *secret;
}

int process(int x) {
    return x ^ magic_value;
}

int main() {
    int result = validate("MyPassword123", SECRET);
    return process(result);
}
