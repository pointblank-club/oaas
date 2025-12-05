import { useState, useCallback, useEffect } from 'react';
import './App.css';
import { GitHubIntegration, FileTree, TestResults } from './components';
import { MetricsDashboard } from './components/MetricsDashboard';
import githubLogo from '../assets/github.png';
import { DATABASE_ENGINE_C, GAME_ENGINE_CPP } from './largeDemos';

type Platform = 'linux' | 'windows' | 'macos';
type Architecture = 'x86_64' | 'arm64' | 'i686';

// Demo Program Categories
type DemoCategory = 'basic' | 'function' | 'recursion' | 'mathematical' | 'exception' | 'oop' | 'large';

const DEMO_CATEGORIES: Record<DemoCategory, { name: string; description: string }> = {
  'basic': { name: '🟢 Basic', description: 'Simple programs for getting started' },
  'function': { name: '🔵 Function Based', description: 'Programs with multiple functions' },
  'recursion': { name: '🟣 Recursion Based', description: 'Programs using recursive algorithms' },
  'mathematical': { name: '🔴 Heavy Mathematical', description: 'Computationally intensive programs' },
  'exception': { name: '🟠 Exception Based', description: 'Programs with error handling (C++)' },
  'oop': { name: '🟡 OOP Based', description: 'Object-oriented programs (C++)' },
  'large': { name: '⚫ Large Program', description: 'Programs with 1000+ lines' },
};

// Demo Programs
const DEMO_PROGRAMS = {
  // ========== BASIC CATEGORY ==========
  'hello_world_c': {
    name: 'Hello World (C)',
    category: 'basic' as DemoCategory,
    language: 'c' as const,
    code: `#include <stdio.h>

const char* SECRET_MESSAGE = "Hello from OAAS Obfuscator!";
const char* BUILD_KEY = "BUILD_2024_XYZ_SECRET";

int main() {
    printf("=== Hello World Demo ===\\n\\n");
    printf("%s\\n", SECRET_MESSAGE);
    printf("[BUILD] Key: %s\\n", BUILD_KEY);
    printf("\\n[SUCCESS] Program completed!\\n");
    return 0;
}
`,
  },
  'hello_world_cpp': {
    name: 'Hello World (C++)',
    category: 'basic' as DemoCategory,
    language: 'cpp' as const,
    code: `#include <iostream>
#include <string>

const std::string SECRET_MESSAGE = "Hello from OAAS Obfuscator!";
const std::string BUILD_KEY = "BUILD_2024_XYZ_SECRET";
const std::string VERSION = "1.0.0-secret";

int main() {
    std::cout << "=== Hello World Demo (C++) ===" << std::endl << std::endl;
    std::cout << SECRET_MESSAGE << std::endl;
    std::cout << "[BUILD] Key: " << BUILD_KEY << std::endl;
    std::cout << "[VERSION] " << VERSION << std::endl;
    std::cout << std::endl << "[SUCCESS] Program completed!" << std::endl;
    return 0;
}
`,
  },

  // ========== FUNCTION BASED CATEGORY ==========
  'demo_auth_c': {
    name: 'Authentication System (C)',
    category: 'function' as DemoCategory,
    language: 'c' as const,
    code: `#include <stdio.h>
#include <string.h>

char* ADMIN_PASSWORD = "Admin@SecurePass2024!";
char* API_KEY = "sk_live_a1b2c3d4e5f6g7h8i9j0";
char* DB_CONNECTION = "postgresql://admin:dbpass123@db.internal:5432/prod";

int authenticate(char* username, char* password) {
    printf("[AUTH] Checking credentials for: %s\\n", username);
    if (strcmp(username, "admin") == 0 && strcmp(password, ADMIN_PASSWORD) == 0) {
        printf("[AUTH] Admin access granted\\n");
        return 1;
    }
    printf("[AUTH] Authentication failed\\n");
    return 0;
}

void connect_database() {
    printf("[DB] Connecting to: %s\\n", DB_CONNECTION);
    printf("[DB] Connection established\\n");
}

int main() {
    char* username = "admin";
    char* password = ADMIN_PASSWORD;

    printf("=== Authentication System v1.0 ===\\n\\n");

    if (authenticate(username, password)) {
        printf("\\n[SUCCESS] Access granted\\n");
        printf("[API] Using API key: %s\\n", API_KEY);
        connect_database();
        printf("\\n[USER] Profile loaded\\n");
        printf("  Username: admin\\n");
        printf("  Role: Administrator\\n");
        printf("  Access Level: 10\\n");
        return 0;
    }

    printf("\\n[FAIL] Access denied\\n");
    return 1;
}
`,
  },
  'password_checker': {
    name: 'Password Strength Checker (C)',
    category: 'function' as DemoCategory,
    language: 'c' as const,
    code: `#include <stdio.h>
#include <string.h>
#include <ctype.h>

char* MASTER_BYPASS = "Admin@Override2024!";
char* SECRET_SALT = "s3cr3t_salt_2024";

int check_strength(char* password) {
    int length = strlen(password);
    int has_upper = 0;
    int has_lower = 0;
    int has_digit = 0;
    int has_special = 0;
    int i;
    int score;

    for (i = 0; password[i] != 0; i++) {
        if (isupper(password[i])) has_upper = 1;
        if (islower(password[i])) has_lower = 1;
        if (isdigit(password[i])) has_digit = 1;
        if (!isalnum(password[i])) has_special = 1;
    }

    score = 0;
    if (length >= 8) score = score + 25;
    if (length >= 12) score = score + 25;
    if (has_upper) score = score + 15;
    if (has_lower) score = score + 15;
    if (has_digit) score = score + 10;
    if (has_special) score = score + 10;

    printf("[ANALYSIS] Password: %s\\n", password);
    printf("  Length: %d characters\\n", length);
    printf("  Uppercase: %s\\n", has_upper ? "Yes" : "No");
    printf("  Lowercase: %s\\n", has_lower ? "Yes" : "No");
    printf("  Digits: %s\\n", has_digit ? "Yes" : "No");
    printf("  Special: %s\\n", has_special ? "Yes" : "No");
    printf("\\n[RESULT] Strength: %d/100\\n", score);

    return score;
}

int main() {
    char* password = "TestPass123!";

    printf("=== Password Strength Checker v1.0 ===\\n\\n");

    if (strcmp(password, MASTER_BYPASS) == 0) {
        printf("[BYPASS] Master password detected!\\n");
        printf("[SECRET] Salt: %s\\n", SECRET_SALT);
        printf("\\n[ADMIN] Full access granted\\n");
        return 0;
    }

    check_strength(password);
    return 0;
}
`,
  },

  // ========== RECURSION BASED CATEGORY ==========
  'fibonacci_recursive': {
    name: 'Fibonacci Calculator (C)',
    category: 'recursion' as DemoCategory,
    language: 'c' as const,
    code: `#include <stdio.h>

const char* SECRET_SEQUENCE = "FIB_SECRET_2024_GOLDEN_RATIO";
const char* CACHE_KEY = "cache_key_fibonacci_xyz";

// Recursive Fibonacci - demonstrates deep recursion
long long fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Tail-recursive helper
long long fib_tail_helper(int n, long long a, long long b) {
    if (n == 0) return a;
    if (n == 1) return b;
    return fib_tail_helper(n - 1, b, a + b);
}

long long fibonacci_tail(int n) {
    return fib_tail_helper(n, 0, 1);
}

void print_sequence(int count) {
    printf("[SEQUENCE] First %d Fibonacci numbers:\\n", count);
    for (int i = 0; i < count; i++) {
        printf("  F(%d) = %lld\\n", i, fibonacci_tail(i));
    }
}

int main() {
    printf("=== Fibonacci Calculator v1.0 ===\\n\\n");
    printf("[SECRET] Key: %s\\n\\n", SECRET_SEQUENCE);

    print_sequence(15);

    printf("\\n[COMPUTE] F(30) using recursion: %lld\\n", fibonacci(30));
    printf("[COMPUTE] F(50) using tail recursion: %lld\\n", fibonacci_tail(50));

    printf("\\n[CACHE] Using key: %s\\n", CACHE_KEY);
    printf("[SUCCESS] Calculation complete!\\n");
    return 0;
}
`,
  },
  'quicksort_recursive': {
    name: 'QuickSort Algorithm (C++)',
    category: 'recursion' as DemoCategory,
    language: 'cpp' as const,
    code: `#include <iostream>
#include <vector>
#include <string>

const std::string SORT_KEY = "QUICKSORT_SECRET_PIVOT_2024";
const std::string BENCHMARK_TOKEN = "bench_token_xyz_123";

// Recursive partition function
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Recursive QuickSort
void quickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);   // Recursive call left
        quickSort(arr, pi + 1, high);  // Recursive call right
    }
}

// Recursive binary search
int binarySearch(const std::vector<int>& arr, int target, int low, int high) {
    if (low > high) return -1;
    int mid = low + (high - low) / 2;
    if (arr[mid] == target) return mid;
    if (arr[mid] > target) return binarySearch(arr, target, low, mid - 1);
    return binarySearch(arr, target, mid + 1, high);
}

void printArray(const std::vector<int>& arr, const std::string& label) {
    std::cout << label << ": [";
    for (size_t i = 0; i < arr.size(); i++) {
        std::cout << arr[i];
        if (i < arr.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    std::cout << "=== QuickSort Algorithm Demo ===" << std::endl << std::endl;
    std::cout << "[SECRET] Sort Key: " << SORT_KEY << std::endl << std::endl;

    std::vector<int> arr = {64, 34, 25, 12, 22, 11, 90, 45, 33, 77};

    printArray(arr, "[UNSORTED]");

    quickSort(arr, 0, arr.size() - 1);

    printArray(arr, "[SORTED]  ");

    int target = 45;
    int index = binarySearch(arr, target, 0, arr.size() - 1);
    std::cout << std::endl << "[SEARCH] Found " << target << " at index: " << index << std::endl;

    std::cout << "[BENCHMARK] Token: " << BENCHMARK_TOKEN << std::endl;
    std::cout << "[SUCCESS] Sorting complete!" << std::endl;
    return 0;
}
`,
  },

  // ========== HEAVY MATHEMATICAL CATEGORY ==========
  'matrix_operations': {
    name: 'Matrix Operations (C)',
    category: 'mathematical' as DemoCategory,
    language: 'c' as const,
    code: `#include <stdio.h>
#include <math.h>

#define SIZE 4

const char* MATRIX_KEY = "MATRIX_CRYPTO_KEY_2024_SECRET";
const char* TRANSFORM_SECRET = "transform_secret_xyz";
const double PI_CONSTANT = 3.14159265358979323846;

typedef struct {
    double data[SIZE][SIZE];
} Matrix;

// Matrix multiplication - O(n^3) complexity
Matrix multiply(Matrix a, Matrix b) {
    Matrix result = {{{0}}};
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                result.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return result;
}

// Matrix transpose
Matrix transpose(Matrix m) {
    Matrix result;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result.data[i][j] = m.data[j][i];
        }
    }
    return result;
}

// Calculate determinant (recursive for sub-matrices)
double determinant(double mat[SIZE][SIZE], int n) {
    if (n == 1) return mat[0][0];
    if (n == 2) return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];

    double det = 0;
    double submat[SIZE][SIZE];

    for (int x = 0; x < n; x++) {
        int subi = 0;
        for (int i = 1; i < n; i++) {
            int subj = 0;
            for (int j = 0; j < n; j++) {
                if (j == x) continue;
                submat[subi][subj] = mat[i][j];
                subj++;
            }
            subi++;
        }
        det += (x % 2 == 0 ? 1 : -1) * mat[0][x] * determinant(submat, n - 1);
    }
    return det;
}

// Trigonometric transform
void apply_transform(Matrix* m, double angle) {
    double c = cos(angle);
    double s = sin(angle);
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m->data[i][j] = m->data[i][j] * c + (i + j) * s;
        }
    }
}

void print_matrix(Matrix m, const char* label) {
    printf("%s:\\n", label);
    for (int i = 0; i < SIZE; i++) {
        printf("  [");
        for (int j = 0; j < SIZE; j++) {
            printf("%8.2f", m.data[i][j]);
        }
        printf(" ]\\n");
    }
}

int main() {
    printf("=== Matrix Operations v1.0 ===\\n\\n");
    printf("[SECRET] Key: %s\\n\\n", MATRIX_KEY);

    Matrix a = {{
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    }};

    Matrix b = {{
        {2, 0, 1, 0},
        {0, 2, 0, 1},
        {1, 0, 2, 0},
        {0, 1, 0, 2}
    }};

    print_matrix(a, "[MATRIX A]");
    print_matrix(b, "[MATRIX B]");

    Matrix c = multiply(a, b);
    print_matrix(c, "[A x B]");

    Matrix t = transpose(c);
    print_matrix(t, "[TRANSPOSE]");

    apply_transform(&t, PI_CONSTANT / 4);
    print_matrix(t, "[ROTATED 45°]");

    printf("\\n[DETERMINANT] det(A) = %.2f\\n", determinant(a.data, SIZE));
    printf("[TRANSFORM] Secret: %s\\n", TRANSFORM_SECRET);
    printf("[SUCCESS] Matrix operations complete!\\n");
    return 0;
}
`,
  },
  'signal_processing': {
    name: 'Signal Processing DSP (C)',
    category: 'mathematical' as DemoCategory,
    language: 'c' as const,
    code: `#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define SAMPLE_RATE 44100
#define BUFFER_SIZE 1024

const char* DSP_SECRET_KEY = "DSP_AUDIO_KEY_2024_ENCRYPTED";
const char* FILTER_COEFFS = "filter_coeff_secret_abc123";
const double NYQUIST_FREQ = SAMPLE_RATE / 2.0;

typedef struct {
    double real;
    double imag;
} Complex;

// Fast Fourier Transform (Cooley-Tukey)
void fft(Complex* x, int n) {
    if (n <= 1) return;

    Complex* even = (Complex*)malloc(n/2 * sizeof(Complex));
    Complex* odd = (Complex*)malloc(n/2 * sizeof(Complex));

    for (int i = 0; i < n/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i + 1];
    }

    fft(even, n/2);
    fft(odd, n/2);

    for (int k = 0; k < n/2; k++) {
        double angle = -2.0 * M_PI * k / n;
        Complex t;
        t.real = cos(angle) * odd[k].real - sin(angle) * odd[k].imag;
        t.imag = cos(angle) * odd[k].imag + sin(angle) * odd[k].real;

        x[k].real = even[k].real + t.real;
        x[k].imag = even[k].imag + t.imag;
        x[k + n/2].real = even[k].real - t.real;
        x[k + n/2].imag = even[k].imag - t.imag;
    }

    free(even);
    free(odd);
}

// Low-pass filter (IIR Butterworth)
double lowpass_filter(double input, double* state, double cutoff) {
    double rc = 1.0 / (2.0 * M_PI * cutoff);
    double dt = 1.0 / SAMPLE_RATE;
    double alpha = dt / (rc + dt);

    *state = *state + alpha * (input - *state);
    return *state;
}

// Generate sine wave
void generate_sine(double* buffer, int samples, double freq, double amplitude) {
    for (int i = 0; i < samples; i++) {
        buffer[i] = amplitude * sin(2.0 * M_PI * freq * i / SAMPLE_RATE);
    }
}

// Calculate RMS power
double calculate_rms(double* buffer, int samples) {
    double sum = 0;
    for (int i = 0; i < samples; i++) {
        sum += buffer[i] * buffer[i];
    }
    return sqrt(sum / samples);
}

// Compute magnitude spectrum
void magnitude_spectrum(Complex* fft_result, double* magnitude, int n) {
    for (int i = 0; i < n/2; i++) {
        magnitude[i] = sqrt(fft_result[i].real * fft_result[i].real +
                           fft_result[i].imag * fft_result[i].imag);
    }
}

int main() {
    printf("=== Signal Processing DSP Demo ===\\n\\n");
    printf("[SECRET] DSP Key: %s\\n\\n", DSP_SECRET_KEY);

    double signal[BUFFER_SIZE];
    Complex fft_buffer[BUFFER_SIZE];
    double magnitude[BUFFER_SIZE/2];

    // Generate 440Hz sine wave (A4 note)
    generate_sine(signal, BUFFER_SIZE, 440.0, 1.0);
    printf("[GENERATE] 440Hz sine wave, %d samples\\n", BUFFER_SIZE);

    // Calculate RMS
    double rms = calculate_rms(signal, BUFFER_SIZE);
    printf("[ANALYSIS] RMS Power: %.4f\\n", rms);

    // Prepare FFT input
    for (int i = 0; i < BUFFER_SIZE; i++) {
        fft_buffer[i].real = signal[i];
        fft_buffer[i].imag = 0;
    }

    // Perform FFT
    fft(fft_buffer, BUFFER_SIZE);
    magnitude_spectrum(fft_buffer, magnitude, BUFFER_SIZE);

    // Find peak frequency
    int peak_bin = 0;
    double peak_mag = 0;
    for (int i = 0; i < BUFFER_SIZE/2; i++) {
        if (magnitude[i] > peak_mag) {
            peak_mag = magnitude[i];
            peak_bin = i;
        }
    }
    double peak_freq = (double)peak_bin * SAMPLE_RATE / BUFFER_SIZE;
    printf("[FFT] Peak frequency: %.1f Hz (bin %d)\\n", peak_freq, peak_bin);

    // Apply low-pass filter
    double filter_state = 0;
    double filtered[BUFFER_SIZE];
    for (int i = 0; i < BUFFER_SIZE; i++) {
        filtered[i] = lowpass_filter(signal[i], &filter_state, 500.0);
    }
    printf("[FILTER] Applied 500Hz low-pass filter\\n");

    double filtered_rms = calculate_rms(filtered, BUFFER_SIZE);
    printf("[ANALYSIS] Filtered RMS: %.4f\\n", filtered_rms);

    printf("\\n[COEFFS] %s\\n", FILTER_COEFFS);
    printf("[SUCCESS] DSP processing complete!\\n");
    return 0;
}
`,
  },

  // ========== EXCEPTION BASED CATEGORY ==========
  'exception_handler': {
    name: 'Exception Handler (C++)',
    category: 'exception' as DemoCategory,
    language: 'cpp' as const,
    code: `#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

const std::string ERROR_KEY = "ERROR_HANDLER_SECRET_2024";
const std::string RECOVERY_TOKEN = "recovery_token_xyz_secure";
const std::string FALLBACK_CONFIG = "fallback_config_secret";

// Custom exception classes
class DatabaseException : public std::runtime_error {
private:
    int error_code;
public:
    DatabaseException(const std::string& msg, int code)
        : std::runtime_error(msg), error_code(code) {}
    int getErrorCode() const { return error_code; }
};

class NetworkException : public std::runtime_error {
private:
    std::string endpoint;
public:
    NetworkException(const std::string& msg, const std::string& ep)
        : std::runtime_error(msg), endpoint(ep) {}
    std::string getEndpoint() const { return endpoint; }
};

class AuthenticationException : public std::exception {
private:
    std::string message;
    std::string username;
public:
    AuthenticationException(const std::string& msg, const std::string& user)
        : message(msg), username(user) {}
    const char* what() const noexcept override { return message.c_str(); }
    std::string getUsername() const { return username; }
};

// RAII resource manager
class ResourceGuard {
private:
    std::string resource_name;
    bool acquired;
public:
    ResourceGuard(const std::string& name) : resource_name(name), acquired(true) {
        std::cout << "[RESOURCE] Acquired: " << resource_name << std::endl;
    }
    ~ResourceGuard() {
        if (acquired) {
            std::cout << "[RESOURCE] Released: " << resource_name << std::endl;
        }
    }
    void release() { acquired = false; }
};

// Functions that may throw
void connectDatabase(const std::string& conn_string) {
    if (conn_string.empty()) {
        throw DatabaseException("Empty connection string", 1001);
    }
    if (conn_string.find("invalid") != std::string::npos) {
        throw DatabaseException("Invalid database host", 1002);
    }
    std::cout << "[DB] Connected successfully" << std::endl;
}

void fetchFromNetwork(const std::string& url) {
    if (url.find("timeout") != std::string::npos) {
        throw NetworkException("Connection timed out", url);
    }
    if (url.find("404") != std::string::npos) {
        throw NetworkException("Resource not found", url);
    }
    std::cout << "[NET] Fetched from: " << url << std::endl;
}

void authenticateUser(const std::string& user, const std::string& pass) {
    if (user.empty() || pass.empty()) {
        throw AuthenticationException("Missing credentials", user);
    }
    if (pass.length() < 8) {
        throw AuthenticationException("Password too short", user);
    }
    std::cout << "[AUTH] User authenticated: " << user << std::endl;
}

int main() {
    std::cout << "=== Exception Handler Demo ===" << std::endl << std::endl;
    std::cout << "[SECRET] Error Key: " << ERROR_KEY << std::endl << std::endl;

    // Test 1: Database exception
    std::cout << "[TEST 1] Database connection..." << std::endl;
    try {
        ResourceGuard dbGuard("DatabaseConnection");
        connectDatabase("invalid_host:5432");
    } catch (const DatabaseException& e) {
        std::cout << "[CAUGHT] DatabaseException: " << e.what()
                  << " (code: " << e.getErrorCode() << ")" << std::endl;
        std::cout << "[RECOVERY] Using token: " << RECOVERY_TOKEN << std::endl;
    }

    // Test 2: Network exception
    std::cout << std::endl << "[TEST 2] Network fetch..." << std::endl;
    try {
        ResourceGuard netGuard("NetworkSocket");
        fetchFromNetwork("https://api.example.com/timeout");
    } catch (const NetworkException& e) {
        std::cout << "[CAUGHT] NetworkException: " << e.what()
                  << " (endpoint: " << e.getEndpoint() << ")" << std::endl;
    }

    // Test 3: Authentication exception
    std::cout << std::endl << "[TEST 3] Authentication..." << std::endl;
    try {
        authenticateUser("admin", "short");
    } catch (const AuthenticationException& e) {
        std::cout << "[CAUGHT] AuthenticationException: " << e.what()
                  << " (user: " << e.getUsername() << ")" << std::endl;
    }

    // Test 4: Nested try-catch with rethrow
    std::cout << std::endl << "[TEST 4] Nested exception handling..." << std::endl;
    try {
        try {
            throw std::runtime_error("Inner exception");
        } catch (const std::exception& e) {
            std::cout << "[INNER] Caught: " << e.what() << std::endl;
            throw; // Rethrow
        }
    } catch (const std::exception& e) {
        std::cout << "[OUTER] Re-caught: " << e.what() << std::endl;
    }

    // Test 5: Exception-safe resource management
    std::cout << std::endl << "[TEST 5] RAII cleanup..." << std::endl;
    try {
        ResourceGuard guard1("Resource1");
        ResourceGuard guard2("Resource2");
        throw std::runtime_error("Simulated failure");
    } catch (...) {
        std::cout << "[CAUGHT] Exception - resources auto-cleaned" << std::endl;
    }

    std::cout << std::endl << "[FALLBACK] Config: " << FALLBACK_CONFIG << std::endl;
    std::cout << "[SUCCESS] Exception handling complete!" << std::endl;
    return 0;
}
`,
  },

  // ========== OOP BASED CATEGORY ==========
  'demo_license_cpp': {
    name: 'License Validator (C++)',
    category: 'oop' as DemoCategory,
    language: 'cpp' as const,
    code: `#include <iostream>
#include <string>
#include <vector>
#include <map>

// Hardcoded secrets (anti-pattern for demo)
const std::string MASTER_KEY = "ENTERPRISE-MASTER-2024-A1B2C3D4E5F6";
const std::string RSA_KEY = "-----BEGIN RSA PRIVATE KEY-----\\nMIIE...";
const std::string AES_KEY = "AES256_PROD_KEY_2024_DO_NOT_SHARE";
const std::string ACTIVATION_SECRET = "activation_secret_xyz_2024";

enum class LicenseType { TRIAL, STANDARD, PROFESSIONAL, ENTERPRISE };

template<typename T>
class SecureContainer {
private:
    std::vector<T> data;
    std::string key;
public:
    SecureContainer(const std::string& k) : key(k) {
        std::cout << "[SECURE] Container initialized\\n";
    }
    void add(const T& item) { data.push_back(item); }
    size_t size() const { return data.size(); }
};

class License {
protected:
    std::string license_key;
    std::string owner;
    LicenseType type;
    bool activated;
public:
    License(const std::string& key, const std::string& own, LicenseType t)
        : license_key(key), owner(own), type(t), activated(false) {}

    virtual bool validate() const {
        std::cout << "[LICENSE] Validating: " << license_key << "\\n";
        if (license_key == MASTER_KEY) {
            std::cout << "[LICENSE] Master key detected\\n";
            return true;
        }
        return activated;
    }

    void activate(const std::string& code) {
        if (code == ACTIVATION_SECRET) {
            activated = true;
            std::cout << "[LICENSE] Activation successful\\n";
        }
    }

    bool is_activated() const { return activated; }
};

class EnterpriseLicense : public License {
private:
    int max_users;
    std::vector<std::string> features;
public:
    EnterpriseLicense(const std::string& key, const std::string& own, int users)
        : License(key, own, LicenseType::ENTERPRISE), max_users(users) {
        features = {"Analytics", "Support", "Cloud", "API"};
    }

    bool validate() const override {
        if (!License::validate()) return false;
        std::cout << "[ENTERPRISE] Max users: " << max_users << "\\n";
        return true;
    }
};

int main(int argc, char** argv) {
    std::cout << "=== License Validator v2.0 ===\\n\\n";

    std::string key = (argc >= 2) ? argv[1] : MASTER_KEY;
    std::string code = (argc >= 3) ? argv[2] : ACTIVATION_SECRET;

    EnterpriseLicense* license = new EnterpriseLicense(key, "Acme Corp", 100);
    license->activate(code);

    bool valid = license->validate();

    if (valid && license->is_activated()) {
        std::cout << "\\n[SUCCESS] License valid\\n";
        std::cout << "[CRYPTO] AES Key: " << AES_KEY << "\\n";
        delete license;
        return 0;
    }

    std::cout << "\\n[FAIL] License invalid\\n";
    delete license;
    return 1;
}
`,
  },

  // ========== LARGE PROGRAM CATEGORY ==========
  'config_system': {
    name: 'Configuration Manager (C++, ~1500 lines)',
    category: 'large' as DemoCategory,
    language: 'cpp' as const,
    code: `#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <sstream>
#include <algorithm>
#include <functional>
#include <ctime>
#include <cstdlib>

// ============================================================================
// SECRET CONFIGURATION VALUES - These should be obfuscated
// ============================================================================
namespace Secrets {
    const std::string MASTER_API_KEY = "sk_live_master_2024_abcdef123456789";
    const std::string DATABASE_URL = "postgresql://admin:SuperSecretPass123@db.internal:5432/production";
    const std::string REDIS_PASSWORD = "redis_secret_password_2024_xyz";
    const std::string JWT_SECRET = "jwt_signing_secret_do_not_share_ever_2024";
    const std::string ENCRYPTION_KEY = "AES256_MASTER_KEY_32_BYTES_LONG!";
    const std::string AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE";
    const std::string AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
    const std::string STRIPE_SECRET = "sk_live_stripe_secret_key_2024";
    const std::string SENDGRID_KEY = "SG.sendgrid_api_key_secret_2024";
    const std::string GITHUB_TOKEN = "ghp_github_personal_access_token_secret";
}

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================
class ConfigValue;
class ConfigSection;
class ConfigValidator;
class ConfigSerializer;
class ConfigManager;
class Logger;
class EventEmitter;

// ============================================================================
// ENUMERATIONS
// ============================================================================
enum class LogLevel { DEBUG, INFO, WARNING, ERROR, CRITICAL };
enum class ConfigType { STRING, INTEGER, FLOAT, BOOLEAN, ARRAY, OBJECT };
enum class ValidationResult { VALID, INVALID, WARNING };
enum class Environment { DEVELOPMENT, STAGING, PRODUCTION };

// ============================================================================
// LOGGER CLASS
// ============================================================================
class Logger {
private:
    LogLevel min_level;
    std::string prefix;
    bool timestamps_enabled;

    std::string getLevelString(LogLevel level) const {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARN";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::CRITICAL: return "CRIT";
            default: return "UNKNOWN";
        }
    }

    std::string getTimestamp() const {
        time_t now = time(nullptr);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));
        return std::string(buf);
    }

public:
    Logger(const std::string& p = "", LogLevel level = LogLevel::INFO)
        : min_level(level), prefix(p), timestamps_enabled(true) {}

    void setLevel(LogLevel level) { min_level = level; }
    void setPrefix(const std::string& p) { prefix = p; }
    void enableTimestamps(bool enable) { timestamps_enabled = enable; }

    void log(LogLevel level, const std::string& message) {
        if (level < min_level) return;

        std::cout << "[" << getLevelString(level) << "]";
        if (timestamps_enabled) std::cout << " " << getTimestamp();
        if (!prefix.empty()) std::cout << " [" << prefix << "]";
        std::cout << " " << message << std::endl;
    }

    void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
    void info(const std::string& msg) { log(LogLevel::INFO, msg); }
    void warning(const std::string& msg) { log(LogLevel::WARNING, msg); }
    void error(const std::string& msg) { log(LogLevel::ERROR, msg); }
    void critical(const std::string& msg) { log(LogLevel::CRITICAL, msg); }
};

// ============================================================================
// EVENT EMITTER CLASS
// ============================================================================
class EventEmitter {
public:
    using Callback = std::function<void(const std::string&)>;

private:
    std::map<std::string, std::vector<Callback>> listeners;
    Logger logger;

public:
    EventEmitter() : logger("EventEmitter") {}

    void on(const std::string& event, Callback callback) {
        listeners[event].push_back(callback);
        logger.debug("Registered listener for: " + event);
    }

    void emit(const std::string& event, const std::string& data = "") {
        logger.debug("Emitting event: " + event);
        if (listeners.find(event) != listeners.end()) {
            for (auto& callback : listeners[event]) {
                callback(data);
            }
        }
    }

    void removeAllListeners(const std::string& event) {
        listeners.erase(event);
    }
};

// ============================================================================
// CONFIG VALUE CLASS
// ============================================================================
class ConfigValue {
private:
    ConfigType type;
    std::string string_value;
    int int_value;
    double float_value;
    bool bool_value;
    std::vector<std::string> array_value;
    std::map<std::string, std::string> object_value;
    bool is_secret;
    std::string description;

public:
    ConfigValue() : type(ConfigType::STRING), int_value(0), float_value(0.0),
                    bool_value(false), is_secret(false) {}

    // Factory methods
    static ConfigValue fromString(const std::string& value, bool secret = false) {
        ConfigValue cv;
        cv.type = ConfigType::STRING;
        cv.string_value = value;
        cv.is_secret = secret;
        return cv;
    }

    static ConfigValue fromInt(int value) {
        ConfigValue cv;
        cv.type = ConfigType::INTEGER;
        cv.int_value = value;
        return cv;
    }

    static ConfigValue fromFloat(double value) {
        ConfigValue cv;
        cv.type = ConfigType::FLOAT;
        cv.float_value = value;
        return cv;
    }

    static ConfigValue fromBool(bool value) {
        ConfigValue cv;
        cv.type = ConfigType::BOOLEAN;
        cv.bool_value = value;
        return cv;
    }

    static ConfigValue fromArray(const std::vector<std::string>& value) {
        ConfigValue cv;
        cv.type = ConfigType::ARRAY;
        cv.array_value = value;
        return cv;
    }

    // Getters
    ConfigType getType() const { return type; }
    bool isSecret() const { return is_secret; }
    void setSecret(bool secret) { is_secret = secret; }
    void setDescription(const std::string& desc) { description = desc; }
    std::string getDescription() const { return description; }

    std::string asString() const {
        switch (type) {
            case ConfigType::STRING: return string_value;
            case ConfigType::INTEGER: return std::to_string(int_value);
            case ConfigType::FLOAT: return std::to_string(float_value);
            case ConfigType::BOOLEAN: return bool_value ? "true" : "false";
            default: return "";
        }
    }

    int asInt() const { return (type == ConfigType::INTEGER) ? int_value : 0; }
    double asFloat() const { return (type == ConfigType::FLOAT) ? float_value : 0.0; }
    bool asBool() const { return (type == ConfigType::BOOLEAN) ? bool_value : false; }
    std::vector<std::string> asArray() const { return array_value; }

    std::string getMaskedValue() const {
        if (!is_secret) return asString();
        std::string val = asString();
        if (val.length() <= 4) return "****";
        return val.substr(0, 2) + std::string(val.length() - 4, '*') + val.substr(val.length() - 2);
    }
};

// ============================================================================
// CONFIG SECTION CLASS
// ============================================================================
class ConfigSection {
private:
    std::string name;
    std::map<std::string, ConfigValue> values;
    std::map<std::string, std::shared_ptr<ConfigSection>> subsections;
    Logger logger;

public:
    ConfigSection(const std::string& n = "") : name(n), logger("ConfigSection:" + n) {}

    void set(const std::string& key, const ConfigValue& value) {
        values[key] = value;
        logger.debug("Set key: " + key);
    }

    ConfigValue get(const std::string& key, const ConfigValue& default_value = ConfigValue()) const {
        auto it = values.find(key);
        return (it != values.end()) ? it->second : default_value;
    }

    bool has(const std::string& key) const {
        return values.find(key) != values.end();
    }

    void remove(const std::string& key) {
        values.erase(key);
    }

    std::shared_ptr<ConfigSection> createSubsection(const std::string& subsection_name) {
        auto section = std::make_shared<ConfigSection>(name + "." + subsection_name);
        subsections[subsection_name] = section;
        return section;
    }

    std::shared_ptr<ConfigSection> getSubsection(const std::string& subsection_name) const {
        auto it = subsections.find(subsection_name);
        return (it != subsections.end()) ? it->second : nullptr;
    }

    std::vector<std::string> getKeys() const {
        std::vector<std::string> keys;
        for (const auto& pair : values) {
            keys.push_back(pair.first);
        }
        return keys;
    }

    std::string getName() const { return name; }
    size_t size() const { return values.size(); }
};

// ============================================================================
// CONFIG VALIDATOR CLASS
// ============================================================================
class ConfigValidator {
public:
    using ValidationRule = std::function<ValidationResult(const ConfigValue&)>;

private:
    std::map<std::string, std::vector<ValidationRule>> rules;
    Logger logger;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

public:
    ConfigValidator() : logger("Validator") {}

    void addRule(const std::string& key, ValidationRule rule) {
        rules[key].push_back(rule);
    }

    void addRequiredRule(const std::string& key) {
        addRule(key, [](const ConfigValue& v) {
            return v.asString().empty() ? ValidationResult::INVALID : ValidationResult::VALID;
        });
    }

    void addMinLengthRule(const std::string& key, size_t min_length) {
        addRule(key, [min_length](const ConfigValue& v) {
            return v.asString().length() >= min_length ? ValidationResult::VALID : ValidationResult::INVALID;
        });
    }

    void addRangeRule(const std::string& key, int min, int max) {
        addRule(key, [min, max](const ConfigValue& v) {
            int val = v.asInt();
            return (val >= min && val <= max) ? ValidationResult::VALID : ValidationResult::INVALID;
        });
    }

    bool validate(const ConfigSection& section) {
        errors.clear();
        warnings.clear();
        bool all_valid = true;

        for (const auto& rule_pair : rules) {
            const std::string& key = rule_pair.first;
            if (!section.has(key)) {
                errors.push_back("Missing required key: " + key);
                all_valid = false;
                continue;
            }

            ConfigValue value = section.get(key);
            for (const auto& rule : rule_pair.second) {
                ValidationResult result = rule(value);
                if (result == ValidationResult::INVALID) {
                    errors.push_back("Validation failed for: " + key);
                    all_valid = false;
                } else if (result == ValidationResult::WARNING) {
                    warnings.push_back("Validation warning for: " + key);
                }
            }
        }

        return all_valid;
    }

    std::vector<std::string> getErrors() const { return errors; }
    std::vector<std::string> getWarnings() const { return warnings; }
};

// ============================================================================
// CONFIG SERIALIZER CLASS
// ============================================================================
class ConfigSerializer {
private:
    Logger logger;

public:
    ConfigSerializer() : logger("Serializer") {}

    std::string toJSON(const ConfigSection& section, bool mask_secrets = true) {
        std::ostringstream oss;
        oss << "{\\n";

        std::vector<std::string> keys = section.getKeys();
        for (size_t i = 0; i < keys.size(); i++) {
            ConfigValue value = section.get(keys[i]);
            std::string display_value = mask_secrets ? value.getMaskedValue() : value.asString();

            oss << "  \\"" << keys[i] << "\\": \\"" << display_value << "\\"";
            if (i < keys.size() - 1) oss << ",";
            oss << "\\n";
        }

        oss << "}";
        return oss.str();
    }

    std::string toINI(const ConfigSection& section, bool mask_secrets = true) {
        std::ostringstream oss;
        oss << "[" << section.getName() << "]\\n";

        for (const auto& key : section.getKeys()) {
            ConfigValue value = section.get(key);
            std::string display_value = mask_secrets ? value.getMaskedValue() : value.asString();
            oss << key << "=" << display_value << "\\n";
        }

        return oss.str();
    }
};

// ============================================================================
// CONFIG MANAGER CLASS (Main Orchestrator)
// ============================================================================
class ConfigManager {
private:
    std::map<std::string, std::shared_ptr<ConfigSection>> sections;
    ConfigValidator validator;
    ConfigSerializer serializer;
    EventEmitter events;
    Logger logger;
    Environment environment;
    bool initialized;
    std::string version;

    void setupDefaultValidation() {
        validator.addRequiredRule("api_key");
        validator.addMinLengthRule("api_key", 10);
        validator.addRequiredRule("database_url");
        validator.addRangeRule("port", 1, 65535);
        validator.addRangeRule("max_connections", 1, 1000);
    }

    void loadSecrets() {
        auto secrets = createSection("secrets");
        secrets->set("master_api_key", ConfigValue::fromString(Secrets::MASTER_API_KEY, true));
        secrets->set("database_url", ConfigValue::fromString(Secrets::DATABASE_URL, true));
        secrets->set("redis_password", ConfigValue::fromString(Secrets::REDIS_PASSWORD, true));
        secrets->set("jwt_secret", ConfigValue::fromString(Secrets::JWT_SECRET, true));
        secrets->set("encryption_key", ConfigValue::fromString(Secrets::ENCRYPTION_KEY, true));
        secrets->set("aws_access_key", ConfigValue::fromString(Secrets::AWS_ACCESS_KEY, true));
        secrets->set("aws_secret_key", ConfigValue::fromString(Secrets::AWS_SECRET_KEY, true));
        secrets->set("stripe_secret", ConfigValue::fromString(Secrets::STRIPE_SECRET, true));
        secrets->set("sendgrid_key", ConfigValue::fromString(Secrets::SENDGRID_KEY, true));
        secrets->set("github_token", ConfigValue::fromString(Secrets::GITHUB_TOKEN, true));
        logger.info("Loaded " + std::to_string(secrets->size()) + " secret values");
    }

    void loadDefaults() {
        auto app = createSection("application");
        app->set("name", ConfigValue::fromString("OAAS Configuration Demo"));
        app->set("version", ConfigValue::fromString("1.0.0"));
        app->set("debug", ConfigValue::fromBool(false));
        app->set("log_level", ConfigValue::fromString("INFO"));

        auto server = createSection("server");
        server->set("host", ConfigValue::fromString("0.0.0.0"));
        server->set("port", ConfigValue::fromInt(8080));
        server->set("max_connections", ConfigValue::fromInt(100));
        server->set("timeout", ConfigValue::fromInt(30));
        server->set("ssl_enabled", ConfigValue::fromBool(true));

        auto database = createSection("database");
        database->set("pool_size", ConfigValue::fromInt(10));
        database->set("max_idle", ConfigValue::fromInt(5));
        database->set("timeout", ConfigValue::fromInt(10));

        auto cache = createSection("cache");
        cache->set("enabled", ConfigValue::fromBool(true));
        cache->set("ttl", ConfigValue::fromInt(3600));
        cache->set("max_size", ConfigValue::fromInt(1000));

        auto features = createSection("features");
        features->set("auth_enabled", ConfigValue::fromBool(true));
        features->set("rate_limiting", ConfigValue::fromBool(true));
        features->set("analytics", ConfigValue::fromBool(true));
        features->set("webhooks", ConfigValue::fromBool(false));

        logger.info("Loaded default configuration");
    }

public:
    ConfigManager() : logger("ConfigManager"), initialized(false), version("1.0.0") {
        logger.info("Initializing Configuration Manager v" + version);
    }

    void initialize(Environment env = Environment::DEVELOPMENT) {
        environment = env;

        std::string env_name;
        switch (env) {
            case Environment::DEVELOPMENT: env_name = "DEVELOPMENT"; break;
            case Environment::STAGING: env_name = "STAGING"; break;
            case Environment::PRODUCTION: env_name = "PRODUCTION"; break;
        }
        logger.info("Environment: " + env_name);

        setupDefaultValidation();
        loadSecrets();
        loadDefaults();

        events.on("config.changed", [this](const std::string& key) {
            logger.debug("Configuration changed: " + key);
        });

        events.on("config.validated", [this](const std::string& section) {
            logger.info("Validated section: " + section);
        });

        initialized = true;
        events.emit("config.initialized");
        logger.info("Configuration Manager initialized successfully");
    }

    std::shared_ptr<ConfigSection> createSection(const std::string& name) {
        auto section = std::make_shared<ConfigSection>(name);
        sections[name] = section;
        return section;
    }

    std::shared_ptr<ConfigSection> getSection(const std::string& name) const {
        auto it = sections.find(name);
        return (it != sections.end()) ? it->second : nullptr;
    }

    bool validateSection(const std::string& name) {
        auto section = getSection(name);
        if (!section) {
            logger.error("Section not found: " + name);
            return false;
        }

        bool valid = validator.validate(*section);

        for (const auto& error : validator.getErrors()) {
            logger.error(error);
        }
        for (const auto& warning : validator.getWarnings()) {
            logger.warning(warning);
        }

        if (valid) {
            events.emit("config.validated", name);
        }

        return valid;
    }

    void printSection(const std::string& name, bool show_secrets = false) {
        auto section = getSection(name);
        if (!section) {
            logger.error("Section not found: " + name);
            return;
        }

        std::cout << "\\n" << serializer.toJSON(*section, !show_secrets) << std::endl;
    }

    void printAllSections(bool show_secrets = false) {
        std::cout << "\\n=== Configuration Dump ===" << std::endl;
        for (const auto& pair : sections) {
            std::cout << "\\n[" << pair.first << "]" << std::endl;
            printSection(pair.first, show_secrets);
        }
    }

    std::string getSecret(const std::string& key) const {
        auto secrets = getSection("secrets");
        if (secrets && secrets->has(key)) {
            return secrets->get(key).asString();
        }
        return "";
    }

    void setConfigValue(const std::string& section_name, const std::string& key, const ConfigValue& value) {
        auto section = getSection(section_name);
        if (section) {
            section->set(key, value);
            events.emit("config.changed", section_name + "." + key);
        }
    }

    Environment getEnvironment() const { return environment; }
    bool isInitialized() const { return initialized; }
    std::string getVersion() const { return version; }

    void shutdown() {
        logger.info("Shutting down Configuration Manager");
        events.emit("config.shutdown");
        sections.clear();
        initialized = false;
    }
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
void printBanner() {
    std::cout << "============================================================" << std::endl;
    std::cout << "     OAAS Configuration Manager Demo - Large Program" << std::endl;
    std::cout << "============================================================" << std::endl;
}

void runDiagnostics(ConfigManager& manager) {
    std::cout << "\\n=== Running Diagnostics ===" << std::endl;

    std::cout << "[CHECK] Manager initialized: " << (manager.isInitialized() ? "YES" : "NO") << std::endl;
    std::cout << "[CHECK] Version: " << manager.getVersion() << std::endl;

    std::string env;
    switch (manager.getEnvironment()) {
        case Environment::DEVELOPMENT: env = "DEVELOPMENT"; break;
        case Environment::STAGING: env = "STAGING"; break;
        case Environment::PRODUCTION: env = "PRODUCTION"; break;
    }
    std::cout << "[CHECK] Environment: " << env << std::endl;

    // Test secret access
    std::string api_key = manager.getSecret("master_api_key");
    std::cout << "[CHECK] API Key accessible: " << (!api_key.empty() ? "YES" : "NO") << std::endl;
    std::cout << "[CHECK] API Key (masked): " << api_key.substr(0, 8) << "..." << std::endl;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================
int main() {
    printBanner();

    ConfigManager manager;
    manager.initialize(Environment::PRODUCTION);

    std::cout << "\\n=== Configuration Loaded ===" << std::endl;
    manager.printAllSections(false);

    std::cout << "\\n=== Secret Values (DEMO - Normally Hidden) ===" << std::endl;
    std::cout << "[SECRET] Master API Key: " << manager.getSecret("master_api_key") << std::endl;
    std::cout << "[SECRET] Database URL: " << manager.getSecret("database_url") << std::endl;
    std::cout << "[SECRET] JWT Secret: " << manager.getSecret("jwt_secret") << std::endl;
    std::cout << "[SECRET] AWS Access Key: " << manager.getSecret("aws_access_key") << std::endl;
    std::cout << "[SECRET] Stripe Secret: " << manager.getSecret("stripe_secret") << std::endl;

    runDiagnostics(manager);

    std::cout << "\\n=== Modifying Configuration ===" << std::endl;
    manager.setConfigValue("server", "port", ConfigValue::fromInt(9000));
    manager.setConfigValue("features", "webhooks", ConfigValue::fromBool(true));

    std::cout << "\\n=== Updated Server Section ===" << std::endl;
    manager.printSection("server");

    std::cout << "\\n=== Validation Test ===" << std::endl;
    auto test_section = manager.createSection("test");
    test_section->set("api_key", ConfigValue::fromString("short"));
    test_section->set("port", ConfigValue::fromInt(99999));

    bool valid = manager.validateSection("test");
    std::cout << "[VALIDATION] Test section valid: " << (valid ? "YES" : "NO") << std::endl;

    manager.shutdown();

    std::cout << "\\n============================================================" << std::endl;
    std::cout << "[SUCCESS] Configuration Manager Demo Complete!" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
`,
  },
  // ========== LARGE PROGRAMS (imported from separate file) ==========
  'database_engine_c': {
    name: DATABASE_ENGINE_C.name,
    category: DATABASE_ENGINE_C.category as DemoCategory,
    language: DATABASE_ENGINE_C.language,
    code: DATABASE_ENGINE_C.code,
  },
  'game_engine_cpp': {
    name: GAME_ENGINE_CPP.name,
    category: GAME_ENGINE_CPP.category as DemoCategory,
    language: GAME_ENGINE_CPP.language,
    code: GAME_ENGINE_CPP.code,
  },
};

interface ReportData {
  input_parameters?: {
    source_file: string;
    platform: string;
    obfuscation_level: number;
    requested_passes: string[];
    applied_passes: string[];  // ✅ Fixed: was "enabled_passes" in PDF, now consistent
    compiler_flags: string[];
    timestamp: string;
  };
  warnings?: string[];
  baseline_status?: "success" | "failed";  // ✅ NEW: indicates if baseline compiled successfully
  comparison_valid?: boolean;  // ✅ NEW: false if baseline failed, don't show comparison
  baseline_metrics?: {
    file_size: number;
    binary_format: string;
    sections: Record<string, number>;
    symbols_count: number;
    functions_count: number;
    entropy: number;
  };
  // ✅ NEW: Baseline compiler metadata for reproducibility
  baseline_compiler?: {
    compiler: string;
    version: string;
    optimization_level: string;
    compilation_method: string;
    compiler_flags: string[];
    passes_applied: string[];
  };
  output_attributes: {
    file_size: number;
    binary_format: string;
    sections: Record<string, number>;
    symbols_count: number;
    functions_count: number;
    entropy: number;
    obfuscation_methods: string[];
  };
  comparison?: {
    size_change: number;
    size_change_percent: number;
    symbols_removed: number;
    symbols_removed_percent: number;
    functions_removed: number;
    functions_removed_percent: number;
    entropy_increase: number;
    entropy_increase_percent: number;
  };
  bogus_code_info: {
    dead_code_blocks: number;
    opaque_predicates: number;
    junk_instructions: number;
    code_bloat_percentage: number;
  };
  cycles_completed: {
    total_cycles: number;
    per_cycle_metrics: Array<{
      cycle: number;
      passes_applied: string[];
      duration_ms: number;
    }>;
  };
  string_obfuscation: {
    total_strings: number;
    encrypted_strings: number;
    encryption_method: string;
    encryption_percentage: number;
  };
  fake_loops_inserted: {
    count: number;
    types: string[];
    locations: string[];
  };
  symbol_obfuscation: {
    enabled: boolean;
    symbols_obfuscated?: number;
    algorithm?: string;
  };
  obfuscation_score: number;
  symbol_reduction: number;
  function_reduction: number;
  size_reduction: number;
  entropy_increase: number;
  estimated_re_effort: string;
  // ✅ NEW: Comprehensive metrics fields
  total_passes_applied?: number;
  total_obfuscation_overhead?: number;
  code_complexity_factor?: number;
  detection_difficulty_rating?: string;
  protections_applied?: {
    control_flow_flattening?: boolean;
    bogus_control_flow?: boolean;
    symbol_obfuscation?: boolean;
    function_hiding?: boolean;
    fake_loops_injected?: boolean;
    string_encryption?: boolean;
    indirect_calls?: boolean;
    total_protections_enabled?: number;
  };
  // ✅ NEW: LLVM IR Analysis Metrics
  control_flow_metrics?: {
    baseline: {
      basic_blocks: number;
      cfg_edges: number;
      cyclomatic_complexity: number;
      functions: number;
      loops: number;
      avg_bb_per_function: number;
    };
    obfuscated: {
      basic_blocks: number;
      cfg_edges: number;
      cyclomatic_complexity: number;
      functions: number;
      loops: number;
      avg_bb_per_function: number;
    };
    comparison: {
      complexity_increase_percent: number;
      basic_blocks_added: number;
      cfg_edges_added: number;
      instruction_growth_percent: number;
      mba_expressions_added: number;
      arithmetic_complexity_increase: number;
    };
  };
  instruction_metrics?: {
    baseline: {
      total_instructions: number;
      instruction_distribution: {
        load: number;
        store: number;
        call: number;
        br: number;
        phi: number;
        arithmetic: number;
        other: number;
      };
      call_instruction_count: number;
      indirect_call_count: number;
    };
    obfuscated: {
      total_instructions: number;
      instruction_distribution: {
        load: number;
        store: number;
        call: number;
        br: number;
        phi: number;
        arithmetic: number;
        other: number;
      };
      call_instruction_count: number;
      indirect_call_count: number;
    };
    comparison: {
      instruction_growth_percent: number;
      mba_expressions_added: number;
      substituted_instructions: number;
      arithmetic_complexity_increase: number;
    };
  };
}

interface Modal {
  type: 'error' | 'warning' | 'success';
  title: string;
  message: string;
  details?: string;  // Optional detailed error message (shown on expand)
}

interface RepoFile {
  path: string;
  content: string;
  is_binary: boolean;
}

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [serverStatus, setServerStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [file, setFile] = useState<File | null>(null);
  const [inputMode, setInputMode] = useState<'file' | 'paste' | 'github'>('file');
  const [pastedSource, setPastedSource] = useState('');
  const [selectedDemo, setSelectedDemo] = useState<string>('');
  const [jobId, setJobId] = useState<string | null>(null);

  // GitHub integration state
  const [repoFiles, setRepoFiles] = useState<RepoFile[]>([]);
  const [selectedRepoFile, setSelectedRepoFile] = useState<RepoFile | null>(null);
  const [repoName, setRepoName] = useState<string>('');
  const [repoBranch, setRepoBranch] = useState<string>('');
  const [repoSessionId, setRepoSessionId] = useState<string | null>(null);  // Fast clone session
  const [repoFileCount, setRepoFileCount] = useState<number>(0);  // File count from fast clone
  const [showGitHubModal, setShowGitHubModal] = useState(false);
  const [loadingSession, setLoadingSession] = useState(false);  // Loading saved session
  const [uploadingFiles, setUploadingFiles] = useState(false);  // Uploading local files
  const [downloadUrls, setDownloadUrls] = useState<Record<Platform, string | null>>({
    linux: null,
    windows: null,
    macos: null,
  });
  const [binaryName, setBinaryName] = useState<string | null>(null);
  const [report, setReport] = useState<ReportData | null>(null);
  const [loading, setLoading] = useState(false);
  const [modal, setModal] = useState<Modal | null>(null);
  const [showErrorDetails, setShowErrorDetails] = useState(false);
  const [progress, setProgress] = useState<{ message: string; percent: number } | null>(null);
  const [detectedLanguage, setDetectedLanguage] = useState<'c' | 'cpp' | null>(null);

  // Layer states (execution order: 1→2→3→4→5)
  const [layer1, setLayer1] = useState(false); // Symbol obfuscation (PRE-COMPILE, FIRST)
  const [layer2, setLayer2] = useState(false); // String encryption (PRE-COMPILE, SECOND)
  const [layer2_5, setLayer2_5] = useState(false); // Indirect calls (PRE-COMPILE, 2.5)
  const [layer3, setLayer3] = useState(false); // OLLVM passes (COMPILE, THIRD - optional)
  const [layer4, setLayer4] = useState(false); // Compiler flags (COMPILE, FOURTH)
  const [layer5, setLayer5] = useState(false); // UPX packing (POST-COMPILE, FINAL)

  // Layer 1: Symbol Obfuscation sub-options
  const [symbolAlgorithm, setSymbolAlgorithm] = useState('sha256');
  const [symbolHashLength, setSymbolHashLength] = useState(12);
  const [symbolPrefix, setSymbolPrefix] = useState('typed');
  const [symbolSalt, setSymbolSalt] = useState('');

  // Layer 2: String Encryption sub-options
  const [fakeLoops, setFakeLoops] = useState<number | string>(0);

  // Layer 2.5: Indirect Call Obfuscation sub-options
  const [indirectStdlib, setIndirectStdlib] = useState(true);
  const [indirectCustom, setIndirectCustom] = useState(true);

  // Layer 3: OLLVM Passes sub-options
  const [passFlattening, setPassFlattening] = useState(false);
  const [passSubstitution, setPassSubstitution] = useState(false);
  const [passBogusControlFlow, setPassBogusControlFlow] = useState(false);
  const [passSplitBasicBlocks, setPassSplitBasicBlocks] = useState(false);
  const [passLinearMBA, setPassLinearMBA] = useState(false);
  const [cycles, setCycles] = useState<number | string>(1);

  // Layer 4: Compiler Flags sub-options
  const [flagLTO, setFlagLTO] = useState(false);
  const [flagSymbolHiding, setFlagSymbolHiding] = useState(false);
  const [flagOmitFramePointer, setFlagOmitFramePointer] = useState(false);
  const [flagSpeculativeLoadHardening, setFlagSpeculativeLoadHardening] = useState(false);
  const [flagO3, setFlagO3] = useState(false);
  const [flagStripSymbols, setFlagStripSymbols] = useState(false);
  const [flagNoBuiltin, setFlagNoBuiltin] = useState(false);

  // Layer 5: UPX Packing sub-options
  const [upxCompression, setUpxCompression] = useState<'fast' | 'default' | 'best' | 'brute'>('best');
  const [upxLzma, setUpxLzma] = useState(true);
  const [upxPreserveOriginal, setUpxPreserveOriginal] = useState(false);

  // Configuration states
  const [obfuscationLevel, setObfuscationLevel] = useState(5);
  const [targetPlatform, setTargetPlatform] = useState<Platform>('linux');
  const [targetArchitecture, setTargetArchitecture] = useState<Architecture>('x86_64');
  const [entrypointCommand, setEntrypointCommand] = useState<string>('./a.out');

  // Build system configuration (for complex projects like CURL)
  type BuildSystem = 'simple' | 'cmake' | 'make' | 'autotools' | 'custom';
  const [buildSystem, setBuildSystem] = useState<BuildSystem>('simple');
  const [customBuildCommand, setCustomBuildCommand] = useState<string>('');
  const [outputBinaryPath, setOutputBinaryPath] = useState<string>('');
  const [cmakeOptions, setCmakeOptions] = useState<string>('');  // Extra cmake flags like -DFOO=OFF

  useEffect(() => {
    document.body.className = darkMode ? 'dark' : 'light';
  }, [darkMode]);

  useEffect(() => {
    fetch('/api/health')
      .then(res => setServerStatus(res.ok ? 'online' : 'offline'))
      .catch(() => setServerStatus('offline'));
  }, []);

  // Restore session on mount
  useEffect(() => {
    const savedSessionId = sessionStorage.getItem('repoSessionId');
    if (savedSessionId) {
      setLoadingSession(true);
      fetch(`/api/github/repo/session/${savedSessionId}`)
        .then(res => {
          if (res.ok) {
            return res.json();
          }
          throw new Error('Session expired or not found');
        })
        .then(data => {
          setRepoSessionId(data.session_id);
          setRepoName(data.repo_name);
          setRepoBranch(data.branch);
          setRepoFileCount(data.file_count || 0);
          // Set input mode based on whether it's a local upload or GitHub clone
          // Local uploads have branch === "local"
          setInputMode(data.branch === 'local' ? 'file' : 'github');
          setLoadingSession(false);
        })
        .catch(err => {
          console.log('No valid session to restore:', err);
          sessionStorage.removeItem('repoSessionId');
          setLoadingSession(false);
        });
    }
  }, []);

  // Auto-detect language
  const detectLanguage = useCallback((filename: string, content?: string): 'c' | 'cpp' => {
    const ext = filename.toLowerCase().split('.').pop();
    if (ext === 'cpp' || ext === 'cc' || ext === 'cxx' || ext === 'c++') return 'cpp';
    if (ext === 'c') return 'c';

    if (content) {
      const cppIndicators = [
        /\bclass\b/, /\bnamespace\b/, /\btemplate\s*</, /\bstd::/,
        /\b(public|private|protected):/, /#include\s*<(iostream|string|vector)>/,
        /\bvirtual\b/, /\boperator\s*\(/
      ];
      if (cppIndicators.some(regex => regex.test(content))) return 'cpp';
    }
    return 'c';
  }, []);

  const onPick = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const nextFile = e.target.files?.[0] ?? null;
    setFile(nextFile);
    if (nextFile) {
      setInputMode('file');
      const lang = detectLanguage(nextFile.name);
      setDetectedLanguage(lang);
      // Clear any existing session since we're switching to single file mode
      if (repoSessionId) {
        setRepoSessionId(null);
        setRepoName('');
        setRepoBranch('');
        setRepoFileCount(0);
        sessionStorage.removeItem('repoSessionId');
      }
    }
  }, [detectLanguage, repoSessionId]);

  // Handle folder/multiple file upload
  const onPickFolder = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploadingFiles(true);
    setFile(null);  // Clear single file

    try {
      // Create FormData with all files
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        // webkitRelativePath contains the relative path including folder name
        const relativePath = (file as any).webkitRelativePath || file.name;
        formData.append('files', file, relativePath);
      }

      // Upload to backend
      const response = await fetch('/api/local/folder/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Upload failed: ${response.status}`);
      }

      const data = await response.json();

      // Use the same state management as GitHub clone
      setRepoSessionId(data.session_id);
      setRepoName(data.repo_name);
      setRepoBranch(data.branch);
      setRepoFileCount(data.file_count);
      setRepoFiles([]);  // Clear any old files
      setInputMode('file');  // Stay in file mode but show as project

      // Save session ID for refresh persistence
      sessionStorage.setItem('repoSessionId', data.session_id);

      // Detect language from first C/C++ file
      const cppExtensions = ['cpp', 'cc', 'cxx', 'c++'];
      let detectedLang: 'c' | 'cpp' = 'c';
      for (let i = 0; i < files.length; i++) {
        const ext = files[i].name.split('.').pop()?.toLowerCase();
        if (ext && cppExtensions.includes(ext)) {
          detectedLang = 'cpp';
          break;
        }
      }
      setDetectedLanguage(detectedLang);

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      setModal({
        type: 'error',
        title: 'Upload Failed',
        message: errorMsg,
      });
    } finally {
      setUploadingFiles(false);
      // Reset the input so the same folder can be selected again
      e.target.value = '';
    }
  }, []);

  // Handle folder upload from FileList (used by drag and drop)
  const uploadFolderFiles = useCallback(async (files: FileList) => {
    if (!files || files.length === 0) return;

    setUploadingFiles(true);
    setFile(null);  // Clear single file

    try {
      // Create FormData with all files
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        // For drag and drop, we need to preserve the folder structure
        const relativePath = (file as any).webkitRelativePath || file.name;
        formData.append('files', file, relativePath);
      }

      // Upload to backend
      const response = await fetch('/api/local/folder/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Upload failed: ${response.status}`);
      }

      const data = await response.json();

      // Use the same state management as GitHub clone
      setRepoSessionId(data.session_id);
      setRepoName(data.repo_name);
      setRepoBranch(data.branch);
      setRepoFileCount(data.file_count);
      setRepoFiles([]);  // Clear any old files
      setInputMode('file');  // Stay in file mode but show as project

      // Save session ID for refresh persistence
      sessionStorage.setItem('repoSessionId', data.session_id);

      // Detect language from first C/C++ file
      const cppExtensions = ['cpp', 'cc', 'cxx', 'c++'];
      let detectedLang: 'c' | 'cpp' = 'c';
      for (let i = 0; i < files.length; i++) {
        const ext = files[i].name.split('.').pop()?.toLowerCase();
        if (ext && cppExtensions.includes(ext)) {
          detectedLang = 'cpp';
          break;
        }
      }
      setDetectedLanguage(detectedLang);

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      setModal({
        type: 'error',
        title: 'Upload Failed',
        message: errorMsg,
      });
    } finally {
      setUploadingFiles(false);
    }
  }, []);

  // Drag and drop state
  const [isDraggingOver, setIsDraggingOver] = useState(false);

  // Handle drag and drop events
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingOver(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingOver(false);

    const files = e.dataTransfer.files;
    
    if (files.length === 0) {
      return;
    }

    // Helper to check if file has valid C/C++ extension
    const validExtensions = ['c', 'cpp', 'cc', 'cxx', 'c++', 'h', 'hpp', 'hxx', 'h++', 'txt'];
    const hasValidExtension = (filename: string) => {
      const ext = filename.toLowerCase().split('.').pop();
      return ext && validExtensions.includes(ext);
    };

    // If multiple files are dropped, treat as folder/project upload
    if (files.length > 1) {
      // Check if there's at least one valid C/C++ file
      let hasValidFiles = false;
      for (let i = 0; i < files.length; i++) {
        if (hasValidExtension(files[i].name)) {
          hasValidFiles = true;
          break;
        }
      }

      if (!hasValidFiles) {
        setModal({
          type: 'error',
          title: 'No Valid Files Found',
          message: `No C/C++ source files found in the dropped files. Please drop files with extensions: ${validExtensions.slice(0, -1).join(', ')}`
        });
        return;
      }

      // Upload all files (backend will filter)
      await uploadFolderFiles(files);
    } else {
      // Single file - check if it's a valid C/C++ file
      const file = files[0];
      const ext = file.name.toLowerCase().split('.').pop();
      
      if (ext && validExtensions.includes(ext)) {
        setFile(file);
        setInputMode('file');
        const lang = detectLanguage(file.name);
        setDetectedLanguage(lang);
        // Clear any existing session
        if (repoSessionId) {
          setRepoSessionId(null);
          setRepoName('');
          setRepoBranch('');
          setRepoFileCount(0);
          sessionStorage.removeItem('repoSessionId');
        }
      } else {
        setModal({
          type: 'error',
          title: 'Invalid File Type',
          message: `File type '.${ext}' is not supported. Please drop a C/C++ source file (.c, .cpp, .cc, .cxx, .c++, .txt).`
        });
      }
    }
  }, [uploadFolderFiles, detectLanguage, repoSessionId]);

  const onGitHubFilesLoaded = useCallback((files: RepoFile[], repoName: string, branch: string) => {
    setRepoFiles(files);
    setRepoName(repoName);
    setRepoBranch(branch);
    setInputMode('github');
    setShowGitHubModal(false);

    // Auto-select file containing main() function
    const mainFile = files.find(f => {
      const ext = f.path.toLowerCase().split('.').pop();
      if (ext && ['c', 'cpp', 'cc', 'cxx', 'c++'].includes(ext)) {
        // Check if file contains main function
        return /\bmain\s*\(/.test(f.content);
      }
      return false;
    });

    // If no main file found, select first C/C++ file
    const sourceFile = mainFile || files.find(f => {
      const ext = f.path.toLowerCase().split('.').pop();
      return ext && ['c', 'cpp', 'cc', 'cxx', 'c++'].includes(ext);
    });

    if (sourceFile) {
      setSelectedRepoFile(sourceFile);
      const lang = detectLanguage(sourceFile.path, sourceFile.content);
      setDetectedLanguage(lang);
    }
  }, [detectLanguage]);

  // Callback for fast clone - repo stays on backend with all files (including build system files)
  const onRepoCloned = useCallback((sessionId: string, name: string, branch: string, fileCount: number) => {
    setRepoSessionId(sessionId);
    setRepoName(name);
    setRepoBranch(branch);
    setRepoFileCount(fileCount);
    setRepoFiles([]);  // Clear any old files - we don't need them for fast clone
    setInputMode('github');
    setShowGitHubModal(false);

    // Save session ID to sessionStorage for persistence across page refreshes
    sessionStorage.setItem('repoSessionId', sessionId);
  }, []);

  // Helper function to parse errors into summary and details
  const parseErrorForDisplay = useCallback((error: string): { summary: string; details?: string } => {
    const lines = error.split('\n').filter(line => line.trim());

    // Check for build failure pattern
    if (error.includes('Build failed:') || error.includes('Compilation failed')) {
      const summaryLine = lines.find(l =>
        l.includes('Build failed') ||
        l.includes('Compilation failed') ||
        l.includes('error:')
      ) || lines[0];

      // Extract a short summary
      let summary = 'Compilation failed';
      const errorMatch = error.match(/error:.*$/m);
      if (errorMatch) {
        summary = errorMatch[0].slice(0, 100) + (errorMatch[0].length > 100 ? '...' : '');
      } else if (summaryLine) {
        summary = summaryLine.slice(0, 100) + (summaryLine.length > 100 ? '...' : '');
      }

      return {
        summary,
        details: lines.length > 1 ? error : undefined
      };
    }

    // For other errors, use first line as summary if multi-line
    if (lines.length > 1) {
      return {
        summary: lines[0].slice(0, 150) + (lines[0].length > 150 ? '...' : ''),
        details: error
      };
    }

    // Single line error - show as is if short, otherwise truncate with details
    if (error.length > 200) {
      return {
        summary: error.slice(0, 200) + '...',
        details: error
      };
    }

    return { summary: error };
  }, []);

  // Helper to show error modal with summary/details pattern
  const showErrorModal = useCallback((title: string, error: string) => {
    const { summary, details } = parseErrorForDisplay(error);
    setShowErrorDetails(false);
    setModal({
      type: 'error',
      title,
      message: summary,
      details
    });
  }, [parseErrorForDisplay]);

  const onGitHubError = useCallback((error: string) => {
    showErrorModal('GitHub Error', error);
  }, [showErrorModal]);

  const onSelectDemo = useCallback((demoKey: string) => {
    if (!demoKey) {
      setSelectedDemo('');
      setPastedSource('');
      setDetectedLanguage(null);
      return;
    }

    const demo = DEMO_PROGRAMS[demoKey as keyof typeof DEMO_PROGRAMS];
    if (demo) {
      setSelectedDemo(demoKey);
      setPastedSource(demo.code);
      setDetectedLanguage(demo.language);
      setInputMode('paste');
    }
  }, []);

  // Count active obfuscation layers
  const countLayers = useCallback(() => {
    let count = 0;
    if (layer1) count++; // Symbol obfuscation
    if (layer2) count++; // String encryption
    if (layer2_5) count++; // Indirect calls
    if (layer3 && (passFlattening || passSubstitution || passBogusControlFlow || passSplitBasicBlocks || passLinearMBA)) count++; // OLLVM passes
    if (layer4 && (flagLTO || flagSymbolHiding || flagOmitFramePointer || flagSpeculativeLoadHardening || flagO3 || flagStripSymbols || flagNoBuiltin)) count++; // Compiler flags
    if (layer5) count++; // UPX packing
    return count;
  }, [layer1, layer2, layer2_5, layer3, layer4, layer5, passFlattening, passSubstitution, passBogusControlFlow, passSplitBasicBlocks, passLinearMBA, flagLTO, flagSymbolHiding, flagOmitFramePointer, flagSpeculativeLoadHardening, flagO3, flagStripSymbols, flagNoBuiltin]);

  // Validate source code syntax
  const validateCode = (code: string, _language: 'c' | 'cpp'): { valid: boolean; error?: string } => {
    if (!code || code.trim().length === 0) {
      return { valid: false, error: 'Source code is empty' };
    }

    // Basic syntax checks
    const hasMainFunction = /\bmain\s*\(/.test(code);
    if (!hasMainFunction) {
      return { valid: false, error: 'No main() function found. Invalid C/C++ program.' };
    }

    // Check for basic C/C++ structure
    const hasInclude = /#include/.test(code);
    const hasFunction = /\w+\s+\w+\s*\([^)]*\)\s*\{/.test(code);

    if (!hasInclude && !hasFunction) {
      return { valid: false, error: 'Invalid C/C++ syntax: missing includes or function definitions' };
    }

    // Check for balanced braces
    const openBraces = (code.match(/\{/g) || []).length;
    const closeBraces = (code.match(/\}/g) || []).length;
    if (openBraces !== closeBraces) {
      return { valid: false, error: `Syntax error: Unbalanced braces (${openBraces} opening, ${closeBraces} closing)` };
    }

    // Check for balanced parentheses in preprocessor directives
    const includeLines = code.match(/#include\s*[<"][^>"]+[>"]/g);
    if (code.includes('#include') && !includeLines) {
      return { valid: false, error: 'Syntax error: Invalid #include directive' };
    }

    return { valid: true };
  };

  const fileToBase64 = (f: File) =>
    new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result;
        if (typeof result === 'string') {
          const [, base64] = result.split(',');
          resolve(base64 ?? result);
        } else {
          reject(new Error('Failed to read file'));
        }
      };
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(f);
    });

  const stringToBase64 = (value: string) => {
    const encoder = new TextEncoder();
    const bytes = encoder.encode(value);
    let binary = '';
    bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
    return btoa(binary);
  };

  // Handle cascading layer selection
  const handleLayerChange = (layer: number, value: boolean) => {
    if (layer === 1) setLayer1(value);
    if (layer === 2) setLayer2(value);
    if (layer === 3) {
      setLayer3(value);
      // LTO now works with Layer 3 (OLLVM) thanks to -fuse-ld=lld fix
    }
    if (layer === 4) setLayer4(value);
    if (layer === 5) setLayer5(value);
  };

  const onSubmit = useCallback(async () => {
    // Validation: Check if source is provided
    // For file mode: allow either single file OR repoSessionId (folder upload)
    if (inputMode === 'file' && !file && !repoSessionId) {
      setModal({
        type: 'error',
        title: 'No File Selected',
        message: 'Please select a C or C++ source file or folder to obfuscate.'
      });
      return;
    }

    if (inputMode === 'paste' && pastedSource.trim().length === 0) {
      setModal({
        type: 'error',
        title: 'Empty Source Code',
        message: 'Please paste valid C/C++ source code before submitting.'
      });
      return;
    }

    // For GitHub mode: require either repoSessionId (fast clone) or repoFiles (legacy)
    if (inputMode === 'github' && !repoSessionId && repoFiles.length === 0) {
      setModal({
        type: 'error',
        title: 'No Repository Loaded',
        message: 'Please load a GitHub repository first.'
      });
      return;
    }

    // Validation: Check if at least one layer is selected
    const layerCount = countLayers();
    if (layerCount === 0) {
      setModal({
        type: 'warning',
        title: 'No Obfuscation Layers Selected',
        message: 'Please enable at least one obfuscation layer (Layer 1-4) before proceeding.'
      });
      return;
    }

    // Get source code for validation
    let sourceCode: string;
    let filename: string;
    let language: 'c' | 'cpp';

    try {
      if (inputMode === 'file' && file && !repoSessionId) {
        // Single file mode
        sourceCode = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = () => reject(new Error('Failed to read file'));
          reader.readAsText(file as File);
        });
        filename = (file as File).name;
        language = detectLanguage(filename, sourceCode);
      } else if (inputMode === 'file' && repoSessionId) {
        // Folder upload mode: files are on backend (same as GitHub fast clone)
        sourceCode = '// Folder upload mode - files on server';
        filename = 'main.c';  // Placeholder - backend finds actual main file
        language = 'c';
      } else if (inputMode === 'paste') {
        sourceCode = pastedSource;
        language = detectLanguage('pasted_source', pastedSource);
        setDetectedLanguage(language);
        filename = language === 'cpp' ? 'pasted_source.cpp' : 'pasted_source.c';
      } else {
        // GitHub mode - for multi-file projects, we don't need a specific file selected
        if (repoSessionId) {
          // Fast clone mode: files are on backend, we can't inspect them locally
          // Use placeholder values - backend will find the main file
          sourceCode = '// Fast clone mode - files on server';
          filename = 'main.c';  // Placeholder - backend finds actual main file
          language = 'c';
        } else if (repoFiles.length > 0) {
          // Legacy mode: use first C/C++ file for language detection
          const firstCppFile = repoFiles.find(f => {
            const ext = f.path.toLowerCase().split('.').pop();
            return ext && ['c', 'cpp', 'cc', 'cxx', 'c++'].includes(ext);
          });

          if (firstCppFile) {
            sourceCode = firstCppFile.content;
            filename = firstCppFile.path.split('/').pop() || 'repo_file.c';
            language = detectLanguage(filename, sourceCode);
          } else {
            sourceCode = '';
            filename = 'repo_file.c';
            language = 'c';
          }
        } else {
          // Fallback
          sourceCode = '';
          filename = 'repo_file.c';
          language = 'c';
        }
      }

      // Validate code syntax
      if (inputMode === 'github' || (inputMode === 'file' && repoSessionId)) {
        // Skip validation for fast clone / folder upload mode - backend handles it
        if (inputMode === 'github' && !repoSessionId) {
          // Legacy mode: validate that at least one file has main()
          const hasMainFunction = repoFiles.some(f => {
            const ext = f.path.toLowerCase().split('.').pop();
            if (ext && ['c', 'cpp', 'cc', 'cxx', 'c++'].includes(ext)) {
              return /\bmain\s*\(/.test(f.content);
            }
            return false;
          });

          if (!hasMainFunction) {
            setModal({
              type: 'error',
              title: 'Invalid Repository',
              message: 'No main() function found in any C/C++ source file. The repository must contain at least one file with a main() function.'
            });
            return;
          }

          // Basic validation: check for C/C++ files
          const hasCppFiles = repoFiles.some(f => {
            const ext = f.path.toLowerCase().split('.').pop();
            return ext && ['c', 'cpp', 'cc', 'cxx', 'c++', 'h', 'hpp', 'hxx', 'h++'].includes(ext);
          });

          if (!hasCppFiles) {
            setModal({
              type: 'error',
              title: 'Invalid Repository',
              message: 'No C/C++ source files found in repository.'
            });
            return;
          }
        }
        // For fast clone/folder upload, backend will validate (it already checks file count > 0)
      } else {
        // For single file mode, validate the file itself
        const validation = validateCode(sourceCode, language);
        if (!validation.valid) {
          setModal({
            type: 'error',
            title: 'Invalid Source Code',
            message: validation.error || 'The provided source code contains syntax errors.'
          });
          return;
        }
      }

    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      showErrorModal('File Read Error', `Failed to read the source file: ${errorMsg}`);
      return;
    }

    setLoading(true);
    setReport(null);
    setDownloadUrls({ linux: null, windows: null, macos: null });
    setBinaryName(null);
    setProgress({ message: 'Initializing...', percent: 0 });

    // Prepare source files for multi-file projects (GitHub mode or folder upload)
    // Priority: 1) repoSessionId (fast clone/folder upload, keeps all files on backend)
    //           2) repoFiles (legacy, only C/C++ files)
    let sourceFiles = null;
    const useSessionId = repoSessionId !== null;  // Works for both GitHub and folder upload

    if (inputMode === 'github' && repoFiles.length > 0 && !repoSessionId) {
      // Legacy mode: filter only C/C++ source and header files
      const validExtensions = ['c', 'cpp', 'cc', 'cxx', 'c++', 'h', 'hpp', 'hxx', 'h++'];
      sourceFiles = repoFiles
        .filter(f => {
          const ext = f.path.toLowerCase().split('.').pop();
          return ext && validExtensions.includes(ext);
        })
        .map(f => ({
          path: f.path,
          content: f.content
        }));
    }

    try {
      const fileCountDisplay = useSessionId ? repoFileCount : (sourceFiles ? sourceFiles.length : 1);
      setProgress({
        message: useSessionId ? `Processing ${fileCountDisplay} file${fileCountDisplay > 1 ? 's' : ''}...` : (inputMode === 'file' ? 'Uploading file...' : inputMode === 'github' ? `Processing ${fileCountDisplay} repository file${fileCountDisplay > 1 ? 's' : ''}...` : 'Encoding source...'),
        percent: 10
      });
      // If using repo session ID (folder upload or GitHub fast clone), don't encode file
      // Files are already uploaded to backend
      const source_b64 = useSessionId ? '' : (inputMode === 'file' && file ? await fileToBase64(file) : stringToBase64(sourceCode));

      // Build compiler flags based on Layer 4 (Compiler Flags) - only selected flags
      const flags: string[] = [];
      if (layer4) {
        // Add non-LTO flags first
        if (flagO3) flags.push('-O3');
        if (flagNoBuiltin) flags.push('-fno-builtin');
        if (flagStripSymbols) flags.push('-Wl,-s');
        if (flagSymbolHiding) flags.push('-fvisibility=hidden');
        if (flagOmitFramePointer) flags.push('-fomit-frame-pointer');
        if (flagSpeculativeLoadHardening) flags.push('-mspeculative-load-hardening');
        // Add LTO at the end (as requested by user)
        if (flagLTO) flags.push('-flto', '-flto=thin');
      }

      // NOTE: Layer 3 OLLVM passes are handled via config.passes object below
      // The server uses wrapper scripts (clang-obfuscate) which apply passes via opt tool
      // No -mllvm flags needed here - they're passed via OLLVM_PASSES environment variable

      const payload = {
        source_code: source_b64,
        filename: filename,
        platform: targetPlatform,
        architecture: targetArchitecture,
        entrypoint_command: buildSystem === 'simple' ? (entrypointCommand.trim() || './a.out') : undefined,
        // Fast clone: use repo_session_id to keep all files on backend (including build system files)
        // Legacy: use source_files (filtered C/C++ only - missing CMakeLists.txt, configure, etc.)
        repo_session_id: useSessionId ? repoSessionId : undefined,
        source_files: useSessionId ? undefined : sourceFiles,
        // Build system configuration for complex projects
        build_system: buildSystem,
        build_command: buildSystem === 'custom' ? customBuildCommand : undefined,
        output_binary_path: buildSystem !== 'simple' && outputBinaryPath ? outputBinaryPath : undefined,
        cmake_options: buildSystem === 'cmake' && cmakeOptions ? cmakeOptions : undefined,
        config: {
          level: obfuscationLevel,
          passes: {
            flattening: layer3 && passFlattening,
            substitution: layer3 && passSubstitution,
            bogus_control_flow: layer3 && passBogusControlFlow,
            split: layer3 && passSplitBasicBlocks,
            linear_mba: layer3 && passLinearMBA
          },
          cycles: layer3 ? (typeof cycles === 'number' ? cycles : parseInt(String(cycles)) || 1) : 1,
          string_encryption: layer2,
          fake_loops: layer2 ? (typeof fakeLoops === 'number' ? fakeLoops : parseInt(String(fakeLoops)) || 0) : 0,
          symbol_obfuscation: {
            enabled: layer1,
            algorithm: layer1 ? symbolAlgorithm : 'sha256',
            hash_length: layer1 ? symbolHashLength : 12,
            prefix_style: layer1 ? symbolPrefix : 'typed',
            salt: layer1 && symbolSalt ? symbolSalt : null
          },
          indirect_calls: {
            enabled: layer2_5,
            obfuscate_stdlib: layer2_5 ? indirectStdlib : true,
            obfuscate_custom: layer2_5 ? indirectCustom : true
          },
          upx: {
            enabled: layer5,
            compression_level: layer5 ? upxCompression : 'best',
            use_lzma: layer5 ? upxLzma : true,
            preserve_original: layer5 ? upxPreserveOriginal : false
          }
        },
        report_formats: ['json', 'markdown'],
        custom_flags: Array.from(new Set(flags.flatMap(f => f.split(' ')).map(t => t.trim()).filter(t => t.length > 0)))
      };

      setProgress({ message: 'Processing obfuscation...', percent: 30 });

      const res = await fetch('/api/obfuscate/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText);
      }

      setProgress({ message: 'Finalizing...', percent: 90 });
      const data = await res.json();

      // Generate custom binary name based on layer count
      const customBinaryName = `obfuscated_${layerCount}_layer`;

      setJobId(data.job_id);

      // Handle response - multi-platform builds
      const downloadUrlsMap: Record<Platform, string | null> = {
        linux: null,
        windows: null,
        macos: null,
      };

      console.log('[DEBUG] Full response data:', JSON.stringify(data, null, 2));
      console.log('[DEBUG] data.download_urls:', data.download_urls);
      console.log('[DEBUG] data.download_url:', data.download_url);

      if (data.download_urls) {
        // Multi-platform build - use platform-specific URLs
        console.log('[DEBUG] Using multi-platform URLs');
        downloadUrlsMap.linux = data.download_urls.linux || null;
        downloadUrlsMap.windows = data.download_urls.windows || null;
        downloadUrlsMap.macos = data.download_urls.macos || null;
        console.log('[DEBUG] Download URLs map:', downloadUrlsMap);
      } else if (data.download_url) {
        // Legacy single platform build
        console.log('[DEBUG] Using legacy single platform URL:', data.download_url);
        downloadUrlsMap.linux = data.download_url;
      }

      console.log('[DEBUG] Final downloadUrlsMap before setState:', downloadUrlsMap);
      setDownloadUrls(downloadUrlsMap);
      setBinaryName(customBinaryName);
      setProgress({ message: 'Complete!', percent: 100 });

      // Clear session storage since backend cleaned up the session
      if (repoSessionId) {
        sessionStorage.removeItem('repoSessionId');
      }

      // Show success modal
      setModal({
        type: 'success',
        title: 'Obfuscation Complete',
        message: `Successfully applied ${layerCount} layer${layerCount > 1 ? 's' : ''} of obfuscation. Binary ready for download.`
      });

      // Auto-fetch report
      if (data.report_url) {
        try {
          const reportRes = await fetch(data.report_url);
          if (reportRes.ok) {
            const reportData = await reportRes.json();
            console.log('[DEBUG] Report data received:', reportData);
            setReport(reportData);
          } else {
            console.error('[DEBUG] Failed to fetch report: HTTP', reportRes.status);
            // Don't show error modal for report fetch failure - binary was still generated successfully
          }
        } catch (reportErr) {
          console.error('[DEBUG] Failed to fetch report:', reportErr);
          const reportErrMsg = reportErr instanceof Error ? reportErr.message : String(reportErr);
          // Show a warning (not error) since the obfuscation succeeded
          setModal({
            type: 'warning',
            title: 'Report Unavailable',
            message: 'Obfuscation completed successfully, but failed to fetch the report.',
            details: `Error: ${reportErrMsg}\n\nYou can still download the binary and try fetching the report manually.`
          });
        }
      }
    } catch (err) {
      console.error('[DEBUG] Obfuscation error:', err);
      setProgress(null);
      const errorMsg = err instanceof Error ? err.message : String(err);

      // Extract detailed error message
      let userFriendlyError = errorMsg;

      // Try to parse JSON error from backend (FastAPI sends {detail: "..."})
      try {
        const errorData = JSON.parse(errorMsg);
        if (errorData.detail) {
          userFriendlyError = errorData.detail;
        }
      } catch {
        // Not JSON, use the raw error message (which already contains the backend error)
      }

      // Only apply generic messages for specific non-compilation errors
      if (errorMsg.includes('timeout') || errorMsg.includes('timed out')) {
        userFriendlyError = 'Obfuscation timed out. Try reducing obfuscation complexity or file size.';
      } else if (errorMsg.includes('network') || errorMsg.includes('Failed to fetch')) {
        userFriendlyError = 'Network error. Please check your connection and try again.';
      }

      showErrorModal('Obfuscation Failed', userFriendlyError);
    } finally {
      setLoading(false);
      setProgress(null);
    }
  }, [
    file, inputMode, pastedSource, obfuscationLevel, cycles, targetPlatform, targetArchitecture, entrypointCommand,
    layer1, layer2, layer3, layer4, layer2_5, layer5,
    symbolAlgorithm, symbolHashLength, symbolPrefix, symbolSalt,
    fakeLoops, indirectStdlib, indirectCustom,
    passFlattening, passSubstitution, passBogusControlFlow, passSplitBasicBlocks, passLinearMBA,
    flagLTO, flagSymbolHiding, flagOmitFramePointer, flagSpeculativeLoadHardening,
    flagO3, flagStripSymbols, flagNoBuiltin,
    upxCompression, upxLzma, upxPreserveOriginal,
    buildSystem, customBuildCommand, outputBinaryPath, cmakeOptions,
    detectLanguage, countLayers, selectedRepoFile, repoSessionId, repoFileCount, repoFiles,
    showErrorModal
  ]);

  const onDownloadBinary = useCallback((platform: Platform) => {
    const url = downloadUrls[platform];
    if (!url) return;
    const link = document.createElement('a');
    link.href = url;
    link.download = binaryName || `obfuscated_${platform}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [downloadUrls, binaryName]);

  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1 className="title">LLVM-OBFUSCATOR</h1>
          <button className="dark-toggle" onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? '☀' : '☾'}
          </button>
        </div>
        <div className="status-bar">
          <span className={`status-indicator ${serverStatus}`}>
            [{serverStatus === 'online' ? '✓' : serverStatus === 'offline' ? '✗' : '...'}] Backend: {serverStatus}
          </span>
          {detectedLanguage && <span className="lang-indicator">[{detectedLanguage.toUpperCase()}]</span>}
        </div>
      </header>

      {/* Modal */}
      {modal && (
        <div className="modal-overlay" onClick={() => { setModal(null); setShowErrorDetails(false); }}>
          <div className={`modal ${modal.details ? 'modal-with-details' : ''}`} onClick={(e) => e.stopPropagation()}>
            <div className={`modal-header ${modal.type}`}>
              <h3>{modal.title}</h3>
              <button className="modal-close" onClick={() => { setModal(null); setShowErrorDetails(false); }}>×</button>
            </div>
            <div className="modal-body">
              <p>{modal.message}</p>
              {modal.details && (
                <div className="error-details-section">
                  <button
                    className="error-details-toggle"
                    onClick={() => setShowErrorDetails(!showErrorDetails)}
                  >
                    {showErrorDetails ? '▼ Hide Details' : '▶ Show Details'}
                  </button>
                  {showErrorDetails && (
                    <div className="error-details-content">
                      <pre>{modal.details}</pre>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="modal-btn" onClick={() => { setModal(null); setShowErrorDetails(false); }}>
                {modal.type === 'success' ? 'OK' : 'Close'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* GitHub Modal */}
      {showGitHubModal && (
        <div className="modal-overlay" onClick={() => setShowGitHubModal(false)}>
          <div className="modal github-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>GitHub Repository</h3>
              <button className="modal-close" onClick={() => setShowGitHubModal(false)}>×</button>
            </div>
            <div className="modal-body">
              <GitHubIntegration
                onFilesLoaded={onGitHubFilesLoaded}
                onRepoCloned={onRepoCloned}
                onError={onGitHubError}
              />
            </div>
          </div>
        </div>
      )}

      <main className="main-content">
        {/* Input Section */}
        <section className="section">
          <h2 className="section-title">[1] SOURCE INPUT</h2>
          <div className="input-section-header">
            <div className="input-mode-toggle">
              <button
                className={inputMode === 'file' ? 'active' : ''}
                onClick={() => setInputMode('file')}
              >
                FILE
              </button>
              <button
                className={inputMode === 'paste' ? 'active' : ''}
                onClick={() => setInputMode('paste')}
              >
                PASTE
              </button>
              <button
                className={inputMode === 'github' ? 'active' : ''}
                onClick={() => setInputMode('github')}
              >
                GITHUB
              </button>
            </div>
            <button
              className="github-btn"
              onClick={() => setShowGitHubModal(true)}
              title="Load from GitHub Repository"
            >
              <span className="github-logo">
                <img src={githubLogo} alt="GitHub Logo" className="github-logo" />
                </span>
              GitHub
            </button>
          </div>

          {inputMode === 'paste' && (
            <div className="config-grid" style={{ marginBottom: '15px' }}>
              <label>
                Load Demo Program:
                <select value={selectedDemo} onChange={(e) => onSelectDemo(e.target.value)}>
                  <option value="">-- Select Demo (10 programs) --</option>
                  {Object.entries(DEMO_CATEGORIES).map(([categoryKey, categoryInfo]) => {
                    const demosInCategory = Object.entries(DEMO_PROGRAMS).filter(
                      ([, demo]) => demo.category === categoryKey
                    );
                    if (demosInCategory.length === 0) return null;
                    return (
                      <optgroup key={categoryKey} label={categoryInfo.name}>
                        {demosInCategory.map(([key, demo]) => (
                          <option key={key} value={key}>{demo.name}</option>
                        ))}
                      </optgroup>
                    );
                  })}
                </select>
              </label>
            </div>
          )}

          {loadingSession && (
            <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-secondary)' }}>
              <p>Restoring previous session...</p>
            </div>
          )}

          {!loadingSession && inputMode === 'file' ? (
            <div className="file-input">
              {/* Show project info if session is active (local folder upload) */}
              {repoSessionId ? (
                <div className="github-repo-loaded">
                  <div className="repo-info">
                    <h4>📁 {repoName} ({repoBranch})</h4>
                    <p>{repoFileCount} C/C++ source files ready</p>
                    <p style={{ fontSize: '0.9em', color: 'var(--text-secondary)', marginTop: '5px' }}>
                      ✓ Files uploaded to server (includes build system files: CMakeLists.txt, configure, etc.)
                    </p>
                    <p style={{ fontSize: '0.9em', color: 'var(--text-secondary)', marginTop: '3px' }}>
                      ℹ️ All C/C++ files will be compiled together into a single obfuscated binary
                    </p>
                    <button
                      className="clear-repo-btn"
                      onClick={() => {
                        setRepoSessionId(null);
                        setRepoName('');
                        setRepoBranch('');
                        setRepoFileCount(0);
                        setFile(null);
                        sessionStorage.removeItem('repoSessionId');
                      }}
                      style={{
                        marginTop: '10px',
                        padding: '8px 16px',
                        backgroundColor: 'var(--danger)',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.9em'
                      }}
                    >
                      Clear Project
                    </button>
                  </div>
                  <div className="github-content">
                    <div className="file-preview" style={{ width: '100%' }}>
                      <div className="file-preview-header">
                        <h5>🚀 Ready to obfuscate</h5>
                      </div>
                      <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                        <p>Project is uploaded and ready for obfuscation.</p>
                        <p style={{ fontSize: '0.9em', marginTop: '10px' }}>
                          Select your obfuscation layers above and click "Obfuscate" to begin.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  {/* Single file and folder upload - side by side */}
                  <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <label className="file-label">
                      <input type="file" accept=".c,.cpp,.cc,.cxx,.txt" onChange={onPick} />
                      SELECT FILE
                    </label>
                    <label className="file-label" style={{ opacity: uploadingFiles ? 0.6 : 1 }}>
                      <input
                        type="file"
                        /* @ts-ignore - webkitdirectory is a non-standard but widely supported attribute */
                        webkitdirectory=""
                        directory=""
                        multiple
                        onChange={onPickFolder}
                        disabled={uploadingFiles}
                      />
                      {uploadingFiles ? 'UPLOADING...' : 'SELECT FOLDER'}
                    </label>
                  </div>
                  {file && <span className="file-name">{file.name}</span>}
                  
                  {/* Drag and drop zone */}
                  <div
                    className={`drag-drop-zone ${isDraggingOver ? 'dragging-over' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    style={{
                      marginTop: '20px',
                      padding: '40px 20px',
                      border: isDraggingOver ? '2px dashed var(--accent)' : '2px dashed var(--border)',
                      borderRadius: '8px',
                      textAlign: 'center',
                      backgroundColor: isDraggingOver ? 'var(--bg-secondary)' : 'transparent',
                      transition: 'all 0.2s ease',
                      cursor: 'pointer'
                    }}
                  >
                    <div style={{ fontSize: '2em', marginBottom: '10px' }}>
                      {isDraggingOver ? '📂' : '📁'}
                    </div>
                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.95em' }}>
                      {isDraggingOver ? (
                        <strong>Drop files or folder here</strong>
                      ) : (
                        <>
                          <strong>Drag & Drop files or folders here</strong>
                          <br />
                          <span style={{ fontSize: '0.85em' }}>
                            Single file: .c, .cpp, .cc, .cxx, .c++, .txt
                            <br />
                            Folder or multiple files: Uploaded as a project
                            <br />
                            <span style={{ opacity: 0.7 }}>Note: Non-C/C++ files will be included for build system support</span>
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          ) : !loadingSession && inputMode === 'paste' ? (
            <textarea
              className="code-input"
              placeholder="// Paste your C/C++ source code here..."
              value={pastedSource}
              onChange={(e) => setPastedSource(e.target.value)}
              rows={20}
              style={{ minHeight: '400px', fontFamily: 'monospace', fontSize: '14px' }}
            />
          ) : !loadingSession ? (
            <div className="github-input">
              {/* Fast clone mode: files are on backend */}
              {repoSessionId ? (
                <div className="github-repo-loaded">
                  <div className="repo-info">
                    <h4>📁 {repoName} ({repoBranch})</h4>
                    <p>{repoFileCount} C/C++ source files ready</p>
                    <p style={{ fontSize: '0.9em', color: 'var(--text-secondary)', marginTop: '5px' }}>
                      ✓ Repository cloned to server (includes build system files: CMakeLists.txt, configure, etc.)
                    </p>
                    <p style={{ fontSize: '0.9em', color: 'var(--text-secondary)', marginTop: '3px' }}>
                      ℹ️ All C/C++ files will be compiled together into a single obfuscated binary
                    </p>
                    <button
                      className="clear-repo-btn"
                      onClick={() => {
                        setRepoSessionId(null);
                        setRepoName('');
                        setRepoBranch('');
                        setRepoFileCount(0);
                        sessionStorage.removeItem('repoSessionId');
                      }}
                      style={{
                        marginTop: '10px',
                        padding: '8px 16px',
                        backgroundColor: 'var(--danger)',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.9em'
                      }}
                    >
                      Clear Repository
                    </button>
                  </div>
                  <div className="github-content">
                    <div className="file-preview" style={{ width: '100%' }}>
                      <div className="file-preview-header">
                        <h5>🚀 Ready to obfuscate</h5>
                      </div>
                      <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                        <p>Repository is cloned and ready for obfuscation.</p>
                        <p style={{ fontSize: '0.9em', marginTop: '10px' }}>
                          Select your obfuscation layers above and click "Obfuscate" to begin.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ) : repoFiles.length > 0 ? (
                <div className="github-repo-loaded">
                  <div className="repo-info">
                    <h4>📁 {repoName} ({repoBranch})</h4>
                    <p>{repoFiles.length} files loaded</p>
                    <p style={{ fontSize: '0.9em', color: 'var(--text-secondary)', marginTop: '5px' }}>
                      ℹ️ All C/C++ files will be compiled together into a single obfuscated binary
                    </p>
                  </div>
                  <div className="github-content">
                    <div className="file-tree-container">
                      <FileTree
                        files={repoFiles}
                        selectedFile={selectedRepoFile?.path || null}
                        onFileSelect={setSelectedRepoFile}
                      />
                    </div>
                    {selectedRepoFile && (
                      <div className="file-preview">
                        <div className="file-preview-header">
                          <h5>📄 {selectedRepoFile.path}</h5>
                        </div>
                        <textarea
                          className="code-input"
                          value={selectedRepoFile.content}
                          readOnly
                          rows={12}
                        />
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="github-empty">
                  <p>No repository loaded. Click the GitHub button to load a repository.</p>
                  <button
                    className="load-github-btn"
                    onClick={() => setShowGitHubModal(true)}
                  >
                    Load GitHub Repository
                  </button>
                </div>
              )}
            </div>
          ) : null}
        </section>

        {/* Layer Selection */}
        <section className="section">
          <h2 className="section-title">[2] OBFUSCATION LAYERS</h2>
          <div className="layer-description">
            Select any combination of layers and their individual options
          </div>
          <div style={{ display: 'flex', gap: '10px', marginBottom: '15px' }}>
            <button
              className="select-all-btn"
              onClick={() => {
                // LTO now works with Layer 3 (OLLVM) thanks to -fuse-ld=lld fix
                const allSelected = layer1 && layer2 && layer2_5 && layer3 && layer4 && layer5 &&
                  passFlattening && passSubstitution && passBogusControlFlow && passSplitBasicBlocks && passLinearMBA &&
                  flagSymbolHiding && flagOmitFramePointer && flagSpeculativeLoadHardening &&
                  flagO3 && flagStripSymbols && flagNoBuiltin && flagLTO;

                const newValue = !allSelected;
                setLayer1(newValue);
                setLayer2(newValue);
                setLayer2_5(newValue);
                setLayer3(newValue);
                setLayer4(newValue);
                setLayer5(newValue);
                setPassFlattening(newValue);
                setPassSubstitution(newValue);
                setPassBogusControlFlow(newValue);
                setPassSplitBasicBlocks(newValue);
                setPassLinearMBA(newValue);
                setFlagLTO(newValue);
                setFlagSymbolHiding(newValue);
                setFlagOmitFramePointer(newValue);
                setFlagSpeculativeLoadHardening(newValue);
                setFlagO3(newValue);
                setFlagStripSymbols(newValue);
                setFlagNoBuiltin(newValue);
              }}
            >
              {layer1 && layer2 && layer2_5 && layer3 && layer4 && layer5 &&
                passFlattening && passSubstitution && passBogusControlFlow && passSplitBasicBlocks && passLinearMBA &&
                flagSymbolHiding && flagOmitFramePointer && flagSpeculativeLoadHardening &&
                flagO3 && flagStripSymbols && flagNoBuiltin && flagLTO
                ? 'Deselect All' : 'Select All'}
            </button>
          </div>
          <div className="layers-grid">
            {/* Layer 1: Symbol Obfuscation */}
            <label className="layer-checkbox">
              <input
                type="checkbox"
                checked={layer1}
                onChange={(e) => handleLayerChange(1, e.target.checked)}
              />
              <span className="layer-label">
                [LAYER 1] Symbol Obfuscation (PRE-COMPILE, 1st)
                <small>Cryptographic hash renaming of all symbols</small>
              </span>
            </label>

            {layer1 && (
              <div className="layer-config">
                <label>
                  Symbol Hash Algorithm:
                  <select value={symbolAlgorithm} onChange={(e) => setSymbolAlgorithm(e.target.value)}>
                    <option value="sha256">SHA256</option>
                    <option value="blake2b">BLAKE2B</option>
                    <option value="siphash">SipHash</option>
                  </select>
                </label>
                <label>
                  Hash Length (8-32):
                  <input
                    type="number"
                    min="8"
                    max="32"
                    value={symbolHashLength}
                    onChange={(e) => setSymbolHashLength(parseInt(e.target.value) || 12)}
                  />
                </label>
                <label>
                  Prefix Style:
                  <select value={symbolPrefix} onChange={(e) => setSymbolPrefix(e.target.value)}>
                    <option value="none">none</option>
                    <option value="typed">typed (f_, v_)</option>
                    <option value="underscore">underscore (_)</option>
                  </select>
                </label>
              </div>
            )}

            {/* Layer 2: String Encryption */}
            <label className="layer-checkbox">
              <input
                type="checkbox"
                checked={layer2}
                onChange={(e) => handleLayerChange(2, e.target.checked)}
              />
              <span className="layer-label">
                [LAYER 2] String Encryption (PRE-COMPILE, 2nd)
                <small>XOR encryption of string literals + runtime decryption</small>
              </span>
            </label>

            {layer2 && (
              <div className="layer-config">
                <label>
                  Fake Loops (0-50):
                  <input
                    type="number"
                    min="0"
                    max="50"
                    value={fakeLoops}
                    onChange={(e) => setFakeLoops(e.target.value === '' ? '' : parseInt(e.target.value) || 0)}
                    onBlur={(e) => {
                      const val = parseInt(e.target.value) || 0;
                      setFakeLoops(Math.min(50, Math.max(0, val)));
                    }}
                  />
                </label>
              </div>
            )}

            {/* Layer 2.5: Indirect Call Obfuscation */}
            <label className="layer-checkbox">
              <input
                type="checkbox"
                checked={layer2_5}
                onChange={(e) => setLayer2_5(e.target.checked)}
              />
              <span className="layer-label">
                [LAYER 2.5] Indirect Call Obfuscation (PRE-COMPILE, 2.5)
                <small>Convert direct function calls to indirect calls via function pointers</small>
              </span>
            </label>

            {layer2_5 && (
              <div className="layer-config">
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={indirectStdlib}
                    onChange={(e) => setIndirectStdlib(e.target.checked)}
                  />
                  Obfuscate stdlib functions (printf, malloc, etc.)
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={indirectCustom}
                    onChange={(e) => setIndirectCustom(e.target.checked)}
                  />
                  Obfuscate custom functions
                </label>
              </div>
            )}

            {/* Layer 3: OLLVM Passes */}
            <label className="layer-checkbox">
              <input
                type="checkbox"
                checked={layer3}
                onChange={(e) => handleLayerChange(3, e.target.checked)}
              />
              <span className="layer-label">
                [LAYER 3] OLLVM Passes (COMPILE, 3rd - Optional)
                <small>Select individual control flow obfuscation passes</small>
              </span>
            </label>

            {layer3 && (
              <div className="layer-config">
                <button
                  className="select-all-btn"
                  style={{ marginBottom: '10px', fontSize: '0.9em' }}
                  onClick={() => {
                    const allPassesSelected = passFlattening && passSubstitution &&
                      passBogusControlFlow && passSplitBasicBlocks && passLinearMBA;
                    const newValue = !allPassesSelected;
                    setPassFlattening(newValue);
                    setPassSubstitution(newValue);
                    setPassBogusControlFlow(newValue);
                    setPassSplitBasicBlocks(newValue);
                    setPassLinearMBA(newValue);
                  }}
                >
                  {passFlattening && passSubstitution && passBogusControlFlow && passSplitBasicBlocks && passLinearMBA
                    ? 'Deselect All' : 'Select All'}
                </button>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={passFlattening}
                    onChange={(e) => setPassFlattening(e.target.checked)}
                  />
                  Control Flow Flattening
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={passSubstitution}
                    onChange={(e) => setPassSubstitution(e.target.checked)}
                  />
                  Instruction Substitution
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={passBogusControlFlow}
                    onChange={(e) => setPassBogusControlFlow(e.target.checked)}
                  />
                  Bogus Control Flow
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={passSplitBasicBlocks}
                    onChange={(e) => setPassSplitBasicBlocks(e.target.checked)}
                  />
                  Split Basic Blocks
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={passLinearMBA}
                    onChange={(e) => setPassLinearMBA(e.target.checked)}
                  />
                  Linear MBA (Mixed Boolean-Arithmetic)
                  <small style={{ display: 'block', color: '#888', marginTop: '2px', marginLeft: '20px' }}>
                    Replaces AND/OR/XOR with per-bit reconstruction
                  </small>
                </label>
                <label>
                  Cycle Count (1-5):
                  <input
                    type="number"
                    min="1"
                    max="5"
                    value={cycles}
                    onChange={(e) => setCycles(e.target.value === '' ? '' : parseInt(e.target.value) || 1)}
                    onBlur={(e) => {
                      const val = parseInt(e.target.value) || 1;
                      setCycles(Math.min(5, Math.max(1, val)));
                    }}
                    title="Number of times to apply OLLVM passes (higher = stronger obfuscation)"
                  />
                  <small style={{ display: 'block', color: '#888', marginTop: '4px' }}>
                    More cycles = stronger obfuscation but larger binary
                  </small>
                </label>
              </div>
            )}

            {/* Layer 4: Compiler Flags */}
            <label className="layer-checkbox">
              <input
                type="checkbox"
                checked={layer4}
                onChange={(e) => handleLayerChange(4, e.target.checked)}
              />
              <span className="layer-label">
                [LAYER 4] Compiler Flags (COMPILE, FINAL)
                <small>Select individual LLVM optimization and hardening flags</small>
              </span>
            </label>

            {layer4 && (
              <div className="layer-config">
                <button
                  className="select-all-btn"
                  style={{ marginBottom: '10px', fontSize: '0.9em' }}
                  onClick={() => {
                    // LTO now works with Layer 3 (OLLVM) thanks to -fuse-ld=lld fix
                    const allFlagsSelected = flagSymbolHiding &&
                      flagOmitFramePointer && flagSpeculativeLoadHardening &&
                      flagO3 && flagStripSymbols && flagNoBuiltin && flagLTO;
                    const newValue = !allFlagsSelected;
                    setFlagSymbolHiding(newValue);
                    setFlagOmitFramePointer(newValue);
                    setFlagSpeculativeLoadHardening(newValue);
                    setFlagO3(newValue);
                    setFlagStripSymbols(newValue);
                    setFlagNoBuiltin(newValue);
                    setFlagLTO(newValue);
                  }}
                >
                  {flagSymbolHiding && flagOmitFramePointer && flagSpeculativeLoadHardening &&
                    flagO3 && flagStripSymbols && flagNoBuiltin && flagLTO
                    ? 'Deselect All' : 'Select All'}
                </button>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={flagSymbolHiding}
                    onChange={(e) => setFlagSymbolHiding(e.target.checked)}
                  />
                  Symbol Hiding (-fvisibility=hidden)
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={flagOmitFramePointer}
                    onChange={(e) => setFlagOmitFramePointer(e.target.checked)}
                  />
                  Remove Frame Pointer (-fomit-frame-pointer)
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={flagSpeculativeLoadHardening}
                    onChange={(e) => setFlagSpeculativeLoadHardening(e.target.checked)}
                  />
                  Speculative Load Hardening
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={flagO3}
                    onChange={(e) => setFlagO3(e.target.checked)}
                  />
                  Maximum Optimization (-O3)
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={flagStripSymbols}
                    onChange={(e) => setFlagStripSymbols(e.target.checked)}
                  />
                  Strip Symbols (-Wl,-s)
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={flagNoBuiltin}
                    onChange={(e) => setFlagNoBuiltin(e.target.checked)}
                  />
                  Disable Built-in Functions (-fno-builtin)
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={flagLTO}
                    onChange={(e) => setFlagLTO(e.target.checked)}
                  />
                  Link-Time Optimization (-flto)
                </label>
              </div>
            )}

            {/* Layer 5: UPX Binary Packing */}
            <label className="layer-checkbox">
              <input
                type="checkbox"
                checked={layer5}
                onChange={(e) => handleLayerChange(5, e.target.checked)}
              />
              <span className="layer-label">
                [LAYER 5] UPX Binary Packing (POST-COMPILE, FINAL)
                <small>Compress binary 50-70% + additional obfuscation layer</small>
              </span>
            </label>

            {layer5 && (
              <div className="layer-config">
                <label>
                  Compression Level:
                  <select
                    value={upxCompression}
                    onChange={(e) => setUpxCompression(e.target.value as 'fast' | 'default' | 'best' | 'brute')}
                  >
                    <option value="fast">Fast (~40-50% compression)</option>
                    <option value="default">Default (~50-60% compression)</option>
                    <option value="best">Best (~60-70% compression)</option>
                    <option value="brute">Brute (~65-75%, very slow)</option>
                  </select>
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={upxLzma}
                    onChange={(e) => setUpxLzma(e.target.checked)}
                  />
                  Use LZMA Compression (better ratio, recommended)
                </label>
                <label className="sub-option">
                  <input
                    type="checkbox"
                    checked={upxPreserveOriginal}
                    onChange={(e) => setUpxPreserveOriginal(e.target.checked)}
                  />
                  Preserve Original Binary (.pre-upx backup)
                </label>
                <div className="layer-info" style={{ marginTop: '10px', fontSize: '0.85em', color: 'var(--text-secondary)' }}>
                  Note: Adds 10-50ms startup overhead. Some antivirus may flag UPX-packed binaries.
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Configuration */}
        <section className="section">
          <h2 className="section-title">[3] CONFIGURATION</h2>
          <div className="config-grid">
            <label>
              Obfuscation Level:
              <select
                value={obfuscationLevel}
                onChange={(e) => setObfuscationLevel(parseInt(e.target.value))}
              >
                <option value="1">1 - Minimal</option>
                <option value="2">2 - Low</option>
                <option value="3">3 - Medium</option>
                <option value="4">4 - High</option>
                <option value="5">5 - Maximum</option>
              </select>
            </label>

            <label>
              Target Platform:
              <select
                value={targetPlatform}
                onChange={(e) => setTargetPlatform(e.target.value as Platform)}
              >
                <option value="linux">Linux</option>
                <option value="windows">Windows</option>
                <option value="macos">macOS (ARM64)</option>
              </select>
            </label>

            <label>
              Target Architecture:
              <select
                value={targetArchitecture}
                onChange={(e) => setTargetArchitecture(e.target.value as Architecture)}
              >
                <option value="x86_64">x86_64 (64-bit Intel/AMD)</option>
                <option value="arm64">ARM64 (Apple M1/M2, ARM servers)</option>
                <option value="i686">i686 (32-bit x86)</option>
              </select>
            </label>

            <label>
              Build System:
              <select
                value={buildSystem}
                onChange={(e) => setBuildSystem(e.target.value as BuildSystem)}
                title="How to compile the project. Use 'Simple' for single files, or select the project's build system for complex projects."
              >
                <option value="simple">Simple (Direct Compilation)</option>
                <option value="cmake">CMake</option>
                <option value="make">Make</option>
                <option value="autotools">Autotools (configure + make)</option>
                <option value="custom">Custom Command</option>
              </select>
            </label>

            {buildSystem === 'custom' && (
              <label>
                Custom Build Command:
                <input
                  type="text"
                  placeholder="e.g., meson build && ninja -C build"
                  value={customBuildCommand}
                  onChange={(e) => setCustomBuildCommand(e.target.value)}
                  title="The exact command to build the project"
                />
              </label>
            )}

            {buildSystem !== 'simple' && (
              <label>
                Output Binary Path (optional):
                <input
                  type="text"
                  placeholder="e.g., build/bin/curl"
                  value={outputBinaryPath}
                  onChange={(e) => setOutputBinaryPath(e.target.value)}
                  title="Hint for where to find the compiled binary after build"
                />
              </label>
            )}

            {buildSystem === 'cmake' && (
              <label>
                CMake Options (optional):
                <input
                  type="text"
                  placeholder="e.g., -DBUILD_TESTING=OFF -DCURL_USE_LIBPSL=OFF"
                  value={cmakeOptions}
                  onChange={(e) => setCmakeOptions(e.target.value)}
                  title="Extra CMake flags to disable features or customize the build"
                />
              </label>
            )}

            {buildSystem === 'simple' && (
              <label>
                Entrypoint Command:
                <input
                  type="text"
                  placeholder="./a.out or gcc main.c -o out && ./out"
                  value={entrypointCommand}
                  onChange={(e) => setEntrypointCommand(e.target.value)}
                  title="Command to run the compiled binary (e.g., ./a.out, make && ./bin/app)"
                />
              </label>
            )}
          </div>

          {buildSystem !== 'simple' && (
            <div className="build-system-info" style={{
              marginTop: '15px',
              padding: '12px',
              backgroundColor: 'var(--bg-secondary)',
              borderRadius: '4px',
              fontSize: '0.9em',
              color: 'var(--text-secondary)'
            }}>
              <strong>Custom Build Mode:</strong> Source-level obfuscation (Layers 1, 2, 2.5) will be applied in-place to all C/C++ files before building.
              Compiler-level obfuscation (Layers 3, 4) will be injected via CC/CXX environment variables.
              {buildSystem === 'cmake' && <span> Build command: <code>cmake -B build && cmake --build build</code></span>}
              {buildSystem === 'make' && <span> Build command: <code>make</code></span>}
              {buildSystem === 'autotools' && <span> Build command: <code>./configure && make</code></span>}
            </div>
          )}
        </section>

        {/* Submit */}
        <section className="section">
          <h2 className="section-title">[4] EXECUTE</h2>
          <button
            className="submit-btn"
            onClick={onSubmit}
            disabled={loading || (
              inputMode === 'file' ? (!file && !repoSessionId) :
                inputMode === 'paste' ? pastedSource.trim().length === 0 :
                  inputMode === 'github' ? (!repoSessionId && repoFiles.length === 0) : true
            )}
          >
            {loading ? 'PROCESSING...' : '► OBFUSCATE'}
          </button>

          {progress && (
            <div className="progress-container">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress.percent}%` }} />
              </div>
              <div className="progress-text">{progress.message} ({progress.percent}%)</div>
            </div>
          )}

          {(downloadUrls.linux || downloadUrls.windows) && (
            <div className="download-section">
              <h3>Download Obfuscated Binary:</h3>
              <div className="download-buttons">
                {(Object.keys(downloadUrls) as Platform[]).map(platform => (
                  downloadUrls[platform] && (
                    <button
                      key={platform}
                      className="download-btn"
                      onClick={() => onDownloadBinary(platform)}
                    >
                      ⬇ {platform.toUpperCase()}
                    </button>
                  )
                ))}
              </div>
              {binaryName && <div className="binary-name">File: {binaryName}</div>}

              {jobId && (
                <div style={{ marginTop: '15px' }}>
                  <h3>Download Report:</h3>
                  <div className="download-buttons">
                    <button
                      className="download-btn"
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = `/api/report/${jobId}?fmt=markdown`;
                        link.download = `report_${jobId}.md`;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                      }}
                    >
                      📄 MARKDOWN
                    </button>
                    <button
                      className="download-btn"
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = `/api/report/${jobId}?fmt=json`;
                        link.download = `report_${jobId}.json`;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                      }}
                    >
                      📊 JSON
                    </button>
                    <button
                      className="download-btn"
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = `/api/report/${jobId}?fmt=pdf`;
                        link.download = `report_${jobId}.pdf`;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                      }}
                    >
                      📋 PDF
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        {/* Report */}
        {report && (
          <section className="section report-section">
            <h2 className="section-title">[5] OBFUSCATION REPORT</h2>

            {/* ✅ FIX: Show warning if baseline failed */}
            {report.baseline_status === "failed" && (
              <div style={{
                backgroundColor: '#fff3cd',
                border: '1px solid #ffc107',
                borderRadius: '4px',
                padding: '12px',
                marginBottom: '16px',
                color: '#856404'
              }}>
                <strong>⚠️ Baseline Compilation Failed</strong>
                <p style={{ margin: '4px 0 0 0', fontSize: '14px' }}>
                  The baseline binary could not be compiled. Comparison metrics below may not be reliable.
                </p>
              </div>
            )}

            {/* ✅ FIX: Show warning if comparison is not valid */}
            {report.warnings && report.warnings.length > 0 && (
              <div style={{
                backgroundColor: '#f8d7da',
                border: '1px solid #f5c6cb',
                borderRadius: '4px',
                padding: '12px',
                marginBottom: '16px',
                color: '#721c24'
              }}>
                <strong>⚠️ Warnings:</strong>
                <ul style={{ margin: '8px 0 0 0', paddingLeft: '20px' }}>
                  {report.warnings.map((warning, idx) => (
                    <li key={idx} style={{ fontSize: '14px', marginBottom: '4px' }}>
                      {warning}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="report-grid">
              {/* Input Parameters */}
              <div className="report-block">
                <h3>INPUT PARAMETERS</h3>
                <div className="report-item">Source: {report.input_parameters?.source_file || 'N/A'}</div>
                <div className="report-item">Platform: {report.input_parameters?.platform || 'N/A'}</div>
                <div className="report-item">Level: {report.input_parameters?.obfuscation_level ?? 'N/A'}</div>
                <div className="report-item">Timestamp: {report.input_parameters?.timestamp || 'N/A'}</div>
                <div className="report-item">Compiler Flags: {report.input_parameters?.compiler_flags?.join(' ') || 'None'}</div>
              </div>

              {/* ✅ NEW: Baseline Compiler Metadata (Dark mode compatible) */}
              {report.baseline_compiler && (
                <div className="report-block" style={{
                  backgroundColor: 'var(--bg-secondary)',
                  borderLeft: '4px solid var(--accent)',
                  padding: '16px',
                  borderRadius: '4px'
                }}>
                  <h3 style={{ color: 'var(--text-primary)', marginTop: 0 }}>📋 BASELINE COMPILATION DETAILS</h3>
                  <div className="report-item">Compiler: {report.baseline_compiler.compiler || 'N/A'}</div>
                  <div className="report-item">Version: {report.baseline_compiler.version || 'N/A'}</div>
                  <div className="report-item">Optimization: {report.baseline_compiler.optimization_level || 'N/A'}</div>
                  <div className="report-item">Method: {report.baseline_compiler.compilation_method || 'N/A'}</div>
                  <div className="report-item">Flags: {report.baseline_compiler.compiler_flags?.join(' ') || 'None'}</div>
                  <div className="report-item">Passes: {report.baseline_compiler.passes_applied?.length ? report.baseline_compiler.passes_applied.join(', ') : 'None (unobfuscated)'}</div>
                </div>
              )}

              {/* ✅ FIX: Check comparison_valid instead of just baseline_metrics && comparison */}
              {report.comparison_valid && report.baseline_metrics && report.comparison && (
                <div className="report-block comparison-block" style={{ gridColumn: '1 / -1' }}>
                  <h3>⚖️ BEFORE / AFTER COMPARISON</h3>
                  <div className="comparison-grid-ui">
                    <div className="comparison-card">
                      <h4>📦 File Size</h4>
                      <div className="comparison-row">
                        <span className="comparison-label">Before:</span>
                        <span className="comparison-value">{(report.baseline_metrics.file_size / 1024).toFixed(2)} KB</span>
                      </div>
                      <div className="comparison-row">
                        <span className="comparison-label">After:</span>
                        <span className="comparison-value">{(report.output_attributes.file_size / 1024).toFixed(2)} KB</span>
                      </div>
                      <div className={`comparison-change ${report.comparison.size_change_percent > 0 ? 'negative' : report.comparison.size_change_percent < 0 ? 'positive' : 'neutral'}`}>
                        {report.comparison.size_change_percent > 0 ? '▲' : report.comparison.size_change_percent < 0 ? '▼' : '='} {Math.abs(report.comparison.size_change_percent).toFixed(2)}%
                        <small>({report.comparison.size_change > 0 ? '+' : ''}{report.comparison.size_change} bytes)</small>
                      </div>
                    </div>

                    <div className="comparison-card">
                      <h4>🏷️ Symbol Count</h4>
                      <div className="comparison-row">
                        <span className="comparison-label">Before:</span>
                        <span className="comparison-value">{report.baseline_metrics.symbols_count}</span>
                      </div>
                      <div className="comparison-row">
                        <span className="comparison-label">After:</span>
                        <span className="comparison-value">{report.output_attributes.symbols_count}</span>
                      </div>
                      <div className={`comparison-change ${report.comparison.symbols_removed_percent > 0 ? 'positive' : 'neutral'}`}>
                        {report.comparison.symbols_removed_percent > 0 ? '✓' : '•'} {report.comparison.symbols_removed} removed ({Math.abs(report.comparison.symbols_removed_percent).toFixed(1)}%)
                      </div>
                      <div className="progress-bar-container">
                        <div className="progress-bar-fill" style={{ width: `${Math.min(100, Math.abs(report.comparison.symbols_removed_percent))}%` }}>
                          {report.comparison.symbols_removed_percent.toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    <div className="comparison-card">
                      <h4>⚙️ Function Count</h4>
                      <div className="comparison-row">
                        <span className="comparison-label">Before:</span>
                        <span className="comparison-value">{report.baseline_metrics.functions_count}</span>
                      </div>
                      <div className="comparison-row">
                        <span className="comparison-label">After:</span>
                        <span className="comparison-value">{report.output_attributes.functions_count}</span>
                      </div>
                      <div className={`comparison-change ${report.comparison.functions_removed_percent > 0 ? 'positive' : 'neutral'}`}>
                        {report.comparison.functions_removed_percent > 0 ? '✓' : '•'} {report.comparison.functions_removed} hidden ({Math.abs(report.comparison.functions_removed_percent).toFixed(1)}%)
                      </div>
                      <div className="progress-bar-container">
                        <div className="progress-bar-fill" style={{ width: `${Math.min(100, Math.abs(report.comparison.functions_removed_percent))}%` }}>
                          {report.comparison.functions_removed_percent.toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    <div className="comparison-card">
                      <h4>🔒 Binary Entropy</h4>
                      <div className="comparison-row">
                        <span className="comparison-label">Before:</span>
                        <span className="comparison-value">{report.baseline_metrics.entropy.toFixed(3)}</span>
                      </div>
                      <div className="comparison-row">
                        <span className="comparison-label">After:</span>
                        <span className="comparison-value">{report.output_attributes.entropy.toFixed(3)}</span>
                      </div>
                      <div className={`comparison-change ${report.comparison.entropy_increase_percent > 0 ? 'positive' : 'neutral'}`}>
                        {report.comparison.entropy_increase_percent > 0 ? '✓' : '•'} {report.comparison.entropy_increase > 0 ? '+' : ''}{report.comparison.entropy_increase.toFixed(3)} ({report.comparison.entropy_increase_percent > 0 ? '+' : ''}{report.comparison.entropy_increase_percent.toFixed(1)}%)
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* ✅ FIX: Show message when comparison is not available */}
              {!report.comparison_valid && (
                <div className="report-block" style={{ gridColumn: '1 / -1', backgroundColor: '#fff3cd', borderLeft: '4px solid #ffc107' }}>
                  <h3 style={{ color: '#856404' }}>⚠️ Comparison Not Available</h3>
                  <p style={{ margin: '8px 0', color: '#856404' }}>
                    {report.baseline_status === "failed"
                      ? "Baseline compilation failed. Comparison metrics are unreliable. Please check the obfuscated binary metrics above instead."
                      : "Comparison metrics are not available for this report."}
                  </p>
                </div>
              )}

              {/* Output Attributes */}
              {report.output_attributes && (
                <div className="report-block">
                  <h3>OUTPUT ATTRIBUTES</h3>
                  <div className="report-item">Size: {report.output_attributes.file_size ? (report.output_attributes.file_size / 1024).toFixed(2) : '0'} KB</div>
                  <div className="report-item">Format: {report.output_attributes.binary_format || 'Unknown'}</div>
                  <div className="report-item">Symbols: {report.output_attributes.symbols_count ?? 0}</div>
                  <div className="report-item">Functions: {report.output_attributes.functions_count ?? 0}</div>
                  <div className="report-item">Entropy: {report.output_attributes.entropy?.toFixed(3) ?? 'N/A'}</div>
                  <div className="report-item">Methods: {report.output_attributes.obfuscation_methods?.join(', ') || 'None'}</div>
                </div>
              )}

              {/* Bogus Code */}
              {report.bogus_code_info && (
                <div className="report-block">
                  <h3>BOGUS CODE GENERATION</h3>
                  <div className="report-item">Dead Blocks: {report.bogus_code_info.dead_code_blocks ?? 0}</div>
                  <div className="report-item">Opaque Predicates: {report.bogus_code_info.opaque_predicates ?? 0}</div>
                  <div className="report-item">Junk Instructions: {report.bogus_code_info.junk_instructions ?? 0}</div>
                  <div className="report-item">Code Bloat: {report.bogus_code_info.code_bloat_percentage ?? 0}%</div>
                </div>
              )}

              {/* Cycles */}
              {report.cycles_completed && (
                <div className="report-block">
                  <h3>OBFUSCATION CYCLES</h3>
                  <div className="report-item">Total: {report.cycles_completed.total_cycles ?? 0}</div>
                  {report.cycles_completed.per_cycle_metrics?.map((cycle, idx) => (
                    <div key={idx} className="report-item">
                      Cycle {cycle.cycle}: {cycle.passes_applied?.join(', ') || 'N/A'} ({cycle.duration_ms ?? 0}ms)
                    </div>
                  ))}
                </div>
              )}

              {/* String Obfuscation */}
              {report.string_obfuscation && (
                <div className="report-block">
                  <h3>STRING ENCRYPTION</h3>
                  <div className="report-item">Total Strings: {report.string_obfuscation.total_strings ?? 0}</div>
                  <div className="report-item">Encrypted: {report.string_obfuscation.encrypted_strings ?? 0}</div>
                  <div className="report-item">Method: {report.string_obfuscation.encryption_method || 'None'}</div>
                  <div className="report-item">Rate: {report.string_obfuscation.encryption_percentage?.toFixed(1) ?? '0.0'}%</div>
                </div>
              )}

              {/* Fake Loops */}
              {report.fake_loops_inserted && (
                <div className="report-block">
                  <h3>FAKE LOOPS</h3>
                  <div className="report-item">Count: {report.fake_loops_inserted.count ?? 0}</div>
                  {report.fake_loops_inserted.locations && report.fake_loops_inserted.locations.length > 0 && (
                    <div className="report-item">Locations: {report.fake_loops_inserted.locations.join(', ')}</div>
                  )}
                </div>
              )}

              {/* Symbol Obfuscation */}
              {report.symbol_obfuscation && (
                <div className="report-block">
                  <h3>SYMBOL OBFUSCATION</h3>
                  <div className="report-item">Enabled: {report.symbol_obfuscation.enabled ? 'Yes' : 'No'}</div>
                  {report.symbol_obfuscation.enabled && report.symbol_obfuscation.symbols_obfuscated && (
                    <>
                      <div className="report-item">Symbols Renamed: {report.symbol_obfuscation.symbols_obfuscated}</div>
                      <div className="report-item">Algorithm: {report.symbol_obfuscation.algorithm || 'N/A'}</div>
                    </>
                  )}
                </div>
              )}

              {/* ✅ REDESIGNED: Comprehensive Metrics Section */}
              <div style={{ gridColumn: '1 / -1', marginTop: '20px' }}>
                {/* Main Score Card */}
                <div style={{
                  backgroundColor: 'var(--bg-secondary)',
                  border: `3px solid ${report.obfuscation_score >= 80 ? '#28a745' : report.obfuscation_score >= 60 ? '#ffc107' : '#dc3545'}`,
                  borderRadius: '8px',
                  padding: '24px',
                  marginBottom: '20px',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '8px' }}>OBFUSCATION SCORE</div>
                  <div style={{
                    fontSize: '48px',
                    fontWeight: 'bold',
                    color: report.obfuscation_score >= 80 ? '#28a745' : report.obfuscation_score >= 60 ? '#ffc107' : '#dc3545',
                    marginBottom: '8px'
                  }}>
                    {(report.obfuscation_score ?? 0).toFixed(1)}/100
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                    {report.obfuscation_score >= 80 ? '🟢 Excellent Protection' : report.obfuscation_score >= 60 ? '🟡 Good Protection' : '🔴 Moderate Protection'}
                  </div>
                  <div style={{ fontSize: '14px', color: 'var(--text-primary)', marginTop: '12px', fontWeight: 'bold' }}>
                    RE Difficulty: {report.estimated_re_effort || 'N/A'}
                  </div>
                  {report.detection_difficulty_rating && (
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '8px' }}>
                      Detection Difficulty: <strong>{report.detection_difficulty_rating}</strong>
                    </div>
                  )}
                </div>

                {/* Metrics Grid */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: '16px',
                  marginBottom: '20px'
                }}>
                  {/* Symbol Reduction */}
                  <div style={{
                    backgroundColor: 'var(--bg-secondary)',
                    border: `1px solid var(--border-color)`,
                    borderRadius: '6px',
                    padding: '16px'
                  }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px' }}>Symbol Reduction</div>
                    <div style={{
                      fontSize: '28px',
                      fontWeight: 'bold',
                      color: (report.symbol_reduction ?? 0) > 0 ? '#28a745' : 'var(--text-secondary)',
                      marginBottom: '4px'
                    }}>
                      {Math.abs(report.symbol_reduction ?? 0).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                      {(report.symbol_reduction ?? 0) > 0 ? '✓ Symbols hidden' : '• No reduction'}
                    </div>
                  </div>

                  {/* Function Reduction */}
                  <div style={{
                    backgroundColor: 'var(--bg-secondary)',
                    border: `1px solid var(--border-color)`,
                    borderRadius: '6px',
                    padding: '16px'
                  }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px' }}>Function Hiding</div>
                    <div style={{
                      fontSize: '28px',
                      fontWeight: 'bold',
                      color: (report.function_reduction ?? 0) > 0 ? '#28a745' : 'var(--text-secondary)',
                      marginBottom: '4px'
                    }}>
                      {Math.abs(report.function_reduction ?? 0).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                      {(report.function_reduction ?? 0) > 0 ? '✓ Functions hidden' : '• No reduction'}
                    </div>
                  </div>

                  {/* Entropy */}
                  <div style={{
                    backgroundColor: 'var(--bg-secondary)',
                    border: `1px solid var(--border-color)`,
                    borderRadius: '6px',
                    padding: '16px'
                  }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px' }}>Entropy Increase</div>
                    <div style={{
                      fontSize: '28px',
                      fontWeight: 'bold',
                      color: (report.entropy_increase ?? 0) > 0 ? '#28a745' : 'var(--text-secondary)',
                      marginBottom: '4px'
                    }}>
                      {((report.entropy_increase ?? 0) > 0 ? '+' : '')}{(report.entropy_increase ?? 0).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                      {(report.entropy_increase ?? 0) > 0 ? '✓ Code complexity' : '• No increase'}
                    </div>
                  </div>

                  {/* Size Change */}
                  <div style={{
                    backgroundColor: 'var(--bg-secondary)',
                    border: `1px solid var(--border-color)`,
                    borderRadius: '6px',
                    padding: '16px'
                  }}>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px' }}>Binary Size</div>
                    <div style={{
                      fontSize: '28px',
                      fontWeight: 'bold',
                      color: (report.size_reduction ?? 0) > 100 ? '#dc3545' : (report.size_reduction ?? 0) > 0 ? '#ffc107' : '#28a745',
                      marginBottom: '4px'
                    }}>
                      {((report.size_reduction ?? 0) > 0 ? '+' : '')}{(report.size_reduction ?? 0).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                      {(report.size_reduction ?? 0) > 100 ? '⚠ Extreme overhead' : (report.size_reduction ?? 0) > 0 ? 'ℹ Expected increase' : '✓ Size reduced'}
                    </div>
                  </div>

                  {/* Complexity Factor */}
                  {report.code_complexity_factor && (
                    <div style={{
                      backgroundColor: 'var(--bg-secondary)',
                      border: `1px solid var(--border-color)`,
                      borderRadius: '6px',
                      padding: '16px'
                    }}>
                      <div style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px' }}>Code Complexity</div>
                      <div style={{
                        fontSize: '28px',
                        fontWeight: 'bold',
                        color: report.code_complexity_factor > 1.3 ? '#28a745' : '#ffc107',
                        marginBottom: '4px'
                      }}>
                        {(report.code_complexity_factor ?? 1).toFixed(2)}x
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                        {report.code_complexity_factor > 1.3 ? '✓ High complexity' : 'ℹ Moderate'}
                      </div>
                    </div>
                  )}

                  {/* Passes Applied */}
                  {report.total_passes_applied && (
                    <div style={{
                      backgroundColor: 'var(--bg-secondary)',
                      border: `1px solid var(--border-color)`,
                      borderRadius: '6px',
                      padding: '16px'
                    }}>
                      <div style={{ fontSize: '11px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px' }}>Passes Applied</div>
                      <div style={{
                        fontSize: '28px',
                        fontWeight: 'bold',
                        color: '#007bff',
                        marginBottom: '4px'
                      }}>
                        {report.total_passes_applied}
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                        obfuscation layers
                      </div>
                    </div>
                  )}
                </div>

                {/* Protections Applied */}
                {report.protections_applied && (
                  <div style={{
                    backgroundColor: 'var(--bg-secondary)',
                    border: `1px solid var(--border-color)`,
                    borderRadius: '6px',
                    padding: '16px',
                    marginBottom: '20px'
                  }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '12px', fontWeight: 'bold' }}>
                      Protections Enabled ({report.protections_applied.total_protections_enabled || 0})
                    </div>
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
                      gap: '12px'
                    }}>
                      {report.protections_applied.control_flow_flattening && (
                        <div style={{ fontSize: '11px', color: '#28a745' }}>✓ Control Flow Flattening</div>
                      )}
                      {report.protections_applied.bogus_control_flow && (
                        <div style={{ fontSize: '11px', color: '#28a745' }}>✓ Bogus Control Flow</div>
                      )}
                      {report.protections_applied.symbol_obfuscation && (
                        <div style={{ fontSize: '11px', color: '#28a745' }}>✓ Symbol Obfuscation</div>
                      )}
                      {report.protections_applied.function_hiding && (
                        <div style={{ fontSize: '11px', color: '#28a745' }}>✓ Function Hiding</div>
                      )}
                      {report.protections_applied.fake_loops_injected && (
                        <div style={{ fontSize: '11px', color: '#28a745' }}>✓ Fake Loops</div>
                      )}
                      {report.protections_applied.string_encryption && (
                        <div style={{ fontSize: '11px', color: '#28a745' }}>✓ String Encryption</div>
                      )}
                      {report.protections_applied.indirect_calls && (
                        <div style={{ fontSize: '11px', color: '#28a745' }}>✓ Indirect Calls</div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}

        {/* ✅ NEW: Advanced Metrics Dashboard */}
        {report && (report.control_flow_metrics || report.instruction_metrics) && (
          <section className="section report-section">
            <h2 className="section-title">[6] ADVANCED METRICS DASHBOARD</h2>
            <MetricsDashboard report={report} />
          </section>
        )}

        {/* ✅ NEW: Test Suite Results Section */}
        {jobId && report && (
          <TestResults
            jobId={jobId}
            onError={(error) => {
              // Optionally show error for missing test results
              if (error.includes('not available') || error.includes('not in report')) {
                // Silently handle - test results may not be available for older jobs
              }
            }}
          />
        )}
      </main>

      <footer className="footer">
        <p>LLVM-OBFUSCATOR :: Research-backed binary hardening</p>
      </footer>
    </div>
  );
}

export default App;
