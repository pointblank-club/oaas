/**
 * VM Interpreter - Public Interface
 *
 * This file defines the public interface for the embedded VM interpreter.
 * The interpreter executes custom bytecode and is designed to be embedded
 * in obfuscated binaries.
 *
 * Features:
 *   - Stack-based architecture
 *   - No dynamic memory allocation
 *   - Position-independent (no global state)
 *   - Bounds-checked stack operations
 */

#ifndef VM_INTERPRETER_H
#define VM_INTERPRETER_H

#include <stdint.h>

/**
 * VM Configuration Constants
 */
#define VM_STACK_SIZE   256  /* Maximum stack depth */
#define VM_REG_COUNT    8    /* Number of virtual registers */
#define VM_ARG_COUNT    8    /* Maximum function arguments */

/**
 * VM Error Codes
 */
#define VM_SUCCESS          0
#define VM_ERR_STACK_OVERFLOW   (-1)
#define VM_ERR_STACK_UNDERFLOW  (-2)
#define VM_ERR_INVALID_OPCODE   (-3)
#define VM_ERR_INVALID_REG      (-4)
#define VM_ERR_INVALID_ARG      (-5)
#define VM_ERR_NULL_BYTECODE    (-6)

/**
 * VM Context - Holds all VM state
 *
 * This structure is designed to fit on the stack (< 4KB).
 * Size calculation:
 *   - vstack: 256 * 8 = 2048 bytes
 *   - vregs:  8 * 8   = 64 bytes
 *   - args:   8 * 8   = 64 bytes
 *   - Other fields:   ~20 bytes
 *   - Total:          ~2200 bytes
 */
typedef struct {
    /* Virtual stack */
    int64_t vstack[VM_STACK_SIZE];  /* Stack storage */
    int32_t vsp;                     /* Stack pointer (index of next free slot) */

    /* Virtual registers */
    int64_t vregs[VM_REG_COUNT];    /* General-purpose registers */

    /* Program state */
    const uint8_t* bytecode;        /* Pointer to bytecode array */
    uint32_t vpc;                    /* Virtual program counter */
    uint32_t bytecode_len;           /* Length of bytecode (for bounds checking) */

    /* Function arguments (passed from native code) */
    int64_t args[VM_ARG_COUNT];     /* Argument values */
    int32_t arg_count;               /* Number of valid arguments */

    /* Error state */
    int32_t error;                   /* Last error code (0 = success) */
} VMContext;

/**
 * Execute bytecode with the VM interpreter.
 *
 * This is the main entry point for the VM. It initializes a VMContext
 * on the stack, executes bytecode, and returns the result.
 *
 * @param bytecode     Pointer to bytecode array
 * @param bytecode_len Length of bytecode in bytes
 * @param args         Pointer to argument array (can be NULL if arg_count is 0)
 * @param arg_count    Number of arguments (0-8)
 *
 * @return Result value from VM (top of stack at VM_RET)
 *         Returns VM error code on failure (negative values)
 *
 * Example:
 *   // Bytecode for: return arg[0] + arg[1]
 *   uint8_t code[] = {VM_LOAD_ARG, 0, VM_LOAD_ARG, 1, VM_ADD, VM_RET};
 *   int64_t args[] = {5, 3};
 *   int64_t result = vm_execute(code, sizeof(code), args, 2);
 *   // result == 8
 */
int64_t vm_execute(const uint8_t* bytecode, uint32_t bytecode_len,
                   const int64_t* args, int32_t arg_count);

/**
 * Initialize a VM context.
 *
 * This function is useful for advanced usage where you need direct
 * access to VM state (e.g., for debugging or step execution).
 *
 * @param ctx          Pointer to VMContext to initialize
 * @param bytecode     Pointer to bytecode array
 * @param bytecode_len Length of bytecode
 * @param args         Pointer to arguments
 * @param arg_count    Number of arguments
 */
void vm_init(VMContext* ctx, const uint8_t* bytecode, uint32_t bytecode_len,
             const int64_t* args, int32_t arg_count);

/**
 * Execute a single instruction.
 *
 * Useful for step-by-step debugging.
 *
 * @param ctx  Pointer to initialized VMContext
 * @return 1 if execution should continue, 0 if VM_RET was executed,
 *         negative on error
 */
int vm_step(VMContext* ctx);

/**
 * Get the current result (top of stack).
 *
 * @param ctx  Pointer to VMContext
 * @return Top of stack value, or 0 if stack is empty
 */
int64_t vm_get_result(const VMContext* ctx);

/**
 * Get error string for error code.
 *
 * @param error  Error code from VMContext or vm_execute
 * @return Static string describing the error
 */
const char* vm_error_string(int32_t error);

#endif /* VM_INTERPRETER_H */
