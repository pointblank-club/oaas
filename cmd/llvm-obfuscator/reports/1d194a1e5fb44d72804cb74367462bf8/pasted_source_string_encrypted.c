typedef unsigned int u32;
typedef unsigned char u8;

#define STATE_MACHINE_STEPS 100
#define MATRIX_DIM 25
#define LIST_SIZE 50
#define SORT_ARRAY_SIZE 200

u32 global_result_a = 0;
u32 global_result_b = 0;
u32 global_result_c = 0;
long global_result_d = 0;

enum AppState {
    STATE_INIT,
    STATE_PROCESSING,
    STATE_TRANSITION,
    STATE_VALIDATION,
    STATE_CLEANUP,
    STATE_TERMINAL
};

struct ComplexData {
    u32 id;
    u32 value_a;
    u32 value_b;
    u32 checksum;
};

struct Node {
    struct ComplexData data;
    struct Node* next;
};

struct Matrix {
    int data[MATRIX_DIM][MATRIX_DIM];
};

enum AppState current_state = STATE_INIT;
struct Matrix matrix_a;
struct Matrix matrix_b;
struct Matrix matrix_c;
struct Node* list_head = (void*)0;
int sort_array[SORT_ARRAY_SIZE];

u32 custom_random_seed = 123456789;

u32 custom_rand() {
    custom_random_seed = (custom_random_seed * 1664525) + 1013904223;
    return custom_random_seed;
}

u32 calculate_checksum(struct ComplexData* d) {
    u32 chk = 0;
    chk ^= d->id;
    chk ^= (d->value_a << 16) | (d->value_a >> 16);
    chk ^= (d->value_b << 8) | (d->value_b >> 24);
    return chk;
}

void init_complex_data(struct ComplexData* d, u32 id) {
    d->id = id;
    d->value_a = custom_rand();
    d->value_b = custom_rand();
    d->checksum = calculate_checksum(d);
}

struct Node* create_node(u32 id) {
    struct Node* new_node = (struct Node*) &(((u8*) (void*)0)[sizeof(struct Node) * id]);
    if (new_node) {
        init_complex_data(&new_node->data, id);
        new_node->next = (void*)0;
    }
    return new_node;
}

void append_node(struct Node** head_ref, struct Node* new_node) {
    if (!new_node) return;
    if (head_ref == (void)0) {
        *head_ref = new_node;
        return;
    }
    struct Node* last = *head_ref;
    while (last->next != (void*)0) {
        last = last->next;
    }
    last->next = new_node;
}

void build_linked_list() {
    int i;
    for (i = 0; i < LIST_SIZE; ++i) {
        struct Node* new_node = create_node(i);
        append_node(&list_head, new_node);
    }
}

u32 traverse_list() {
    u32 sum = 0;
    struct Node* current = list_head;
    while (current != (void*)0) {
        sum += current->data.value_a ^ current->data.value_b;
        current = current->next;
    }
    return sum;
}

void init_matrices() {
    int i, j;
    for (i = 0; i < MATRIX_DIM; ++i) {
        for (j = 0; j < MATRIX_DIM; ++j) {
            matrix_a.data[i][j] = (custom_rand() % 100) - 50;
            matrix_b.data[i][j] = (custom_rand() % 100) - 50;
            matrix_c.data[i][j] = 0;
        }
    }
}

void multiply_matrices() {
    int i, j, k;
    for (i = 0; i < MATRIX_DIM; ++i) {
        for (j = 0; j < MATRIX_DIM; ++j) {
            matrix_c.data[i][j] = 0;
            for (k = 0; k < MATRIX_DIM; ++k) {
                matrix_c.data[i][j] += matrix_a.data[i][k] * matrix_b.data[k][j];
            }
        }
    }
}

long trace_matrix_c() {
    long trace = 0;
    int i;
    for (i = 0; i < MATRIX_DIM; ++i) {
        trace += matrix_c.data[i][i];
    }
    return trace;
}

void transpose_matrix_a() {
    int i, j;
    struct Matrix temp;
    for (i = 0; i < MATRIX_DIM; ++i) {
        for (j = 0; j < MATRIX_DIM; ++j) {
            temp.data[j][i] = matrix_a.data[i][j];
        }
    }
    for (i = 0; i < MATRIX_DIM; ++i) {
        for (j = 0; j < MATRIX_DIM; ++j) {
            matrix_a.data[i][j] = temp.data[i][j];
        }
    }
}

void init_sort_array() {
    int i;
    for (i = 0; i < SORT_ARRAY_SIZE; ++i) {
        sort_array[i] = custom_rand();
    }
}

void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    int j;
    for (j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quick_sort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

u32 process_state_init(u32 input) {
    build_linked_list();
    init_matrices();
    init_sort_array();
    return input ^ 0xDEADBEEF;
}

u32 process_state_processing(u32 input) {
    multiply_matrices();
    transpose_matrix_a();
    quick_sort(sort_array, 0, SORT_ARRAY_SIZE - 1);
    return input + (sort_array[0] ^ sort_array[SORT_ARRAY_SIZE - 1]);
}

u32 process_state_transition(u32 input) {
    u32 list_sum = traverse_list();
    return input - list_sum;
}

u32 process_state_validation(u32 input) {
    long trace = trace_matrix_c();
    global_result_d = trace;
    return input * (u32)trace;
}

u32 process_state_cleanup(u32 input) {
    int i;
    for (i = 0; i < SORT_ARRAY_SIZE; ++i) {
        sort_array[i] = 0;
    }
    return input / 2;
}

void run_state_machine() {
    u32 machine_value = 42;
    int i;
    for (i = 0; i < STATE_MACHINE_STEPS; ++i) {
        switch (current_state) {
            case STATE_INIT:
                machine_value = process_state_init(machine_value);
                current_state = STATE_PROCESSING;
                break;
            case STATE_PROCESSING:
                machine_value = process_state_processing(machine_value);
                current_state = STATE_TRANSITION;
                break;
            case STATE_TRANSITION:
                machine_value = process_state_transition(machine_value);
                current_state = STATE_VALIDATION;
                break;
            case STATE_VALIDATION:
                machine_value = process_state_validation(machine_value);
                if ((machine_value % 3) == 0) {
                    current_state = STATE_CLEANUP;
                } else {
                    current_state = STATE_PROCESSING;
                }
                break;
            case STATE_CLEANUP:
                machine_value = process_state_cleanup(machine_value);
                current_state = STATE_TERMINAL;
                break;
            case STATE_TERMINAL:
                machine_value = 0;
                break;
        }
        if (current_state == STATE_TERMINAL) {
            break;
        }
    }
    global_result_c = machine_value;
}

long fibonacci_recursive(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2);
}

u32 bit_scramble(u32 n) {
    n = (n & 0x55555555) << 1 | (n & 0xAAAAAAAA) >> 1;
    n = (n & 0x33333333) << 2 | (n & 0xCCCCCCCC) >> 2;
    n = (n & 0x0F0F0F0F) << 4 | (n & 0xF0F0F0F0) >> 4;
    n = (n & 0x00FF00FF) << 8 | (n & 0xFF00FF00) >> 8;
    n = (n & 0x0000FFFF) << 16 | (n & 0xFFFF0000) >> 16;
    return n;
}

void deep_calculation_one() {
    u32 val = 0xABCDEFFF;
    int i;
    for (i = 0; i < 50; ++i) {
        val = bit_scramble(val);
        val ^= (u32)fibonacci_recursive(10 + (i % 5));
        val -= sort_array[i];
    }
    global_result_a = val;
}

void deep_calculation_two() {
    u32 val = 0x12345678;
    int i, j;
    for (i = 0; i < MATRIX_DIM; ++i) {
        for (j = 0; j < MATRIX_DIM; ++j) {
            val += matrix_a.data[i][j] * matrix_b.data[j][i];
            val = (val << 3) | (val >> 29);
        }
    }
    global_result_b = val;
}

void (*func_ptr_array[4])(void);

void setup_func_pointers() {
    func_ptr_array[0] = deep_calculation_one;
    func_ptr_array[1] = deep_calculation_two;
    func_ptr_array[2] = transpose_matrix_a;
    func_ptr_array[3] = run_state_machine;
}

void execute_func_pointers() {
    int i;
    for (i = 0; i < 4; ++i) {
        u32 selector = custom_rand() % 4;
        func_ptr_array[selector]();
    }
}

int main() {
    run_state_machine();
    deep_calculation_one();
    deep_calculation_two();
    setup_func_pointers();
    execute_func_pointers();
    
    global_result_a ^= global_result_b;
    global_result_b += global_result_c;
    global_result_c *= (u32)global_result_d;

    return 0;
}