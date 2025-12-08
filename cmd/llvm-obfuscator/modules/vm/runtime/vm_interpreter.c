/**
 * VM Interpreter Implementation
 *
 * A minimal stack-based virtual machine for executing custom bytecode.
 * Designed to be embedded in obfuscated binaries.
 *
 * Features:
 *   - Pure C99, no external dependencies
 *   - No dynamic memory allocation
 *   - Position-independent (no global state)
 *   - Bounds-checked stack operations
 *   - Switch-case dispatcher
 */

#include "vm_interpreter.h"
#include "opcodes.h"
#include <string.h>

/* ========================================================================
 * Helper Macros for Stack Operations (with bounds checking)
 * ======================================================================== */

/**
 * Push a value onto the virtual stack.
 * Sets error and returns on overflow.
 */
#define VM_STACK_PUSH(ctx, val) do { \
    if ((ctx)->vsp >= VM_STACK_SIZE) { \
        (ctx)->error = VM_ERR_STACK_OVERFLOW; \
        return 0; \
    } \
    (ctx)->vstack[(ctx)->vsp++] = (val); \
} while(0)

/**
 * Pop a value from the virtual stack.
 * Sets error and returns on underflow.
 */
#define VM_STACK_POP(ctx, dest) do { \
    if ((ctx)->vsp <= 0) { \
        (ctx)->error = VM_ERR_STACK_UNDERFLOW; \
        return 0; \
    } \
    (dest) = (ctx)->vstack[--(ctx)->vsp]; \
} while(0)

/**
 * Peek at top of stack without popping.
 * Returns 0 if stack is empty.
 */
#define VM_STACK_PEEK(ctx) \
    (((ctx)->vsp > 0) ? (ctx)->vstack[(ctx)->vsp - 1] : 0)

/**
 * Read a single byte from bytecode and advance vpc.
 */
#define VM_READ_U8(ctx) \
    (((ctx)->vpc < (ctx)->bytecode_len) ? (ctx)->bytecode[(ctx)->vpc++] : 0)

/**
 * Read a 32-bit little-endian integer from bytecode.
 */
static inline int32_t vm_read_i32(VMContext* ctx) {
    if (ctx->vpc + 4 > ctx->bytecode_len) {
        ctx->error = VM_ERR_INVALID_OPCODE;
        return 0;
    }
    int32_t val = (int32_t)(
        ((uint32_t)ctx->bytecode[ctx->vpc]) |
        ((uint32_t)ctx->bytecode[ctx->vpc + 1] << 8) |
        ((uint32_t)ctx->bytecode[ctx->vpc + 2] << 16) |
        ((uint32_t)ctx->bytecode[ctx->vpc + 3] << 24)
    );
    ctx->vpc += 4;
    return val;
}

/* ========================================================================
 * VM Initialization
 * ======================================================================== */

void vm_init(VMContext* ctx, const uint8_t* bytecode, uint32_t bytecode_len,
             const int64_t* args, int32_t arg_count) {
    /* Zero-initialize the context */
    memset(ctx, 0, sizeof(VMContext));

    /* Set bytecode */
    ctx->bytecode = bytecode;
    ctx->bytecode_len = bytecode_len;
    ctx->vpc = 0;

    /* Initialize stack */
    ctx->vsp = 0;

    /* Copy arguments (bounds check) */
    ctx->arg_count = (arg_count > VM_ARG_COUNT) ? VM_ARG_COUNT : arg_count;
    if (args != NULL && ctx->arg_count > 0) {
        for (int32_t i = 0; i < ctx->arg_count; i++) {
            ctx->args[i] = args[i];
        }
    }

    ctx->error = VM_SUCCESS;
}

/* ========================================================================
 * Single-Step Execution
 * ======================================================================== */

int vm_step(VMContext* ctx) {
    /* Check for null bytecode */
    if (ctx->bytecode == NULL) {
        ctx->error = VM_ERR_NULL_BYTECODE;
        return -1;
    }

    /* Check if we've reached end of bytecode */
    if (ctx->vpc >= ctx->bytecode_len) {
        ctx->error = VM_ERR_INVALID_OPCODE;
        return -1;
    }

    /* Fetch opcode */
    uint8_t opcode = ctx->bytecode[ctx->vpc++];

    /* Dispatch based on opcode */
    switch (opcode) {

    /* ----------------------------------------------------------------
     * VM_NOP (0x00) - No operation
     * Stack: (no change)
     * ---------------------------------------------------------------- */
    case VM_NOP:
        /* Do nothing, continue execution */
        break;

    /* ----------------------------------------------------------------
     * VM_PUSH (0x01) - Push 32-bit immediate value
     * Operands: 4 bytes (little-endian i32)
     * Stack: -> val
     * ---------------------------------------------------------------- */
    case VM_PUSH: {
        int32_t imm = vm_read_i32(ctx);
        if (ctx->error != VM_SUCCESS) return -1;
        VM_STACK_PUSH(ctx, (int64_t)imm);
        break;
    }

    /* ----------------------------------------------------------------
     * VM_POP (0x02) - Pop and discard top of stack
     * Stack: val ->
     * ---------------------------------------------------------------- */
    case VM_POP: {
        int64_t discard;
        VM_STACK_POP(ctx, discard);
        (void)discard;  /* Suppress unused warning */
        break;
    }

    /* ----------------------------------------------------------------
     * VM_LOAD (0x03) - Load from virtual register to stack
     * Operands: 1 byte (register index 0-7)
     * Stack: -> vregs[idx]
     * ---------------------------------------------------------------- */
    case VM_LOAD: {
        uint8_t reg_idx = VM_READ_U8(ctx);
        if (reg_idx >= VM_REG_COUNT) {
            ctx->error = VM_ERR_INVALID_REG;
            return -1;
        }
        VM_STACK_PUSH(ctx, ctx->vregs[reg_idx]);
        break;
    }

    /* ----------------------------------------------------------------
     * VM_STORE (0x04) - Store from stack to virtual register
     * Operands: 1 byte (register index 0-7)
     * Stack: val -> (stored to vregs[idx])
     * ---------------------------------------------------------------- */
    case VM_STORE: {
        uint8_t reg_idx = VM_READ_U8(ctx);
        if (reg_idx >= VM_REG_COUNT) {
            ctx->error = VM_ERR_INVALID_REG;
            return -1;
        }
        int64_t val;
        VM_STACK_POP(ctx, val);
        ctx->vregs[reg_idx] = val;
        break;
    }

    /* ----------------------------------------------------------------
     * VM_LOAD_ARG (0x05) - Load function argument to stack
     * Operands: 1 byte (argument index 0-7)
     * Stack: -> args[idx]
     * ---------------------------------------------------------------- */
    case VM_LOAD_ARG: {
        uint8_t arg_idx = VM_READ_U8(ctx);
        if (arg_idx >= ctx->arg_count) {
            ctx->error = VM_ERR_INVALID_ARG;
            return -1;
        }
        VM_STACK_PUSH(ctx, ctx->args[arg_idx]);
        break;
    }

    /* ----------------------------------------------------------------
     * VM_ADD (0x10) - Add top two stack values
     * Stack: a, b -> (a + b)
     * ---------------------------------------------------------------- */
    case VM_ADD: {
        int64_t b, a;
        VM_STACK_POP(ctx, b);
        VM_STACK_POP(ctx, a);
        VM_STACK_PUSH(ctx, a + b);
        break;
    }

    /* ----------------------------------------------------------------
     * VM_SUB (0x11) - Subtract top two stack values
     * Stack: a, b -> (a - b)
     * Note: First pushed value minus second pushed value
     * ---------------------------------------------------------------- */
    case VM_SUB: {
        int64_t b, a;
        VM_STACK_POP(ctx, b);
        VM_STACK_POP(ctx, a);
        VM_STACK_PUSH(ctx, a - b);
        break;
    }

    /* ----------------------------------------------------------------
     * VM_XOR (0x12) - XOR top two stack values
     * Stack: a, b -> (a ^ b)
     * ---------------------------------------------------------------- */
    case VM_XOR: {
        int64_t b, a;
        VM_STACK_POP(ctx, b);
        VM_STACK_POP(ctx, a);
        VM_STACK_PUSH(ctx, a ^ b);
        break;
    }

    /* ----------------------------------------------------------------
     * VM_RET (0xFF) - Return result and exit VM
     * Stack: returns top of stack
     * ---------------------------------------------------------------- */
    case VM_RET:
        /* Return 0 to indicate VM_RET was executed */
        return 0;

    /* ----------------------------------------------------------------
     * Invalid opcode - Error
     * ---------------------------------------------------------------- */
    default:
        ctx->error = VM_ERR_INVALID_OPCODE;
        return -1;
    }

    /* Continue execution */
    return 1;
}

/* ========================================================================
 * Get Result
 * ======================================================================== */

int64_t vm_get_result(const VMContext* ctx) {
    if (ctx->vsp > 0) {
        return ctx->vstack[ctx->vsp - 1];
    }
    return 0;
}

/* ========================================================================
 * Main Execution Function
 * ======================================================================== */

int64_t vm_execute(const uint8_t* bytecode, uint32_t bytecode_len,
                   const int64_t* args, int32_t arg_count) {
    /* Check for null bytecode */
    if (bytecode == NULL || bytecode_len == 0) {
        return VM_ERR_NULL_BYTECODE;
    }

    /* Initialize VM context on stack (no heap allocation) */
    VMContext ctx;
    vm_init(&ctx, bytecode, bytecode_len, args, arg_count);

    /* Execute bytecode until VM_RET or error */
    int status;
    while ((status = vm_step(&ctx)) > 0) {
        /* Continue execution */
    }

    /* Check for error */
    if (ctx.error != VM_SUCCESS) {
        return ctx.error;
    }

    /* Return result (top of stack) */
    return vm_get_result(&ctx);
}

/* ========================================================================
 * Error String
 * ======================================================================== */

const char* vm_error_string(int32_t error) {
    switch (error) {
    case VM_SUCCESS:
        return "Success";
    case VM_ERR_STACK_OVERFLOW:
        return "Stack overflow";
    case VM_ERR_STACK_UNDERFLOW:
        return "Stack underflow";
    case VM_ERR_INVALID_OPCODE:
        return "Invalid opcode";
    case VM_ERR_INVALID_REG:
        return "Invalid register index";
    case VM_ERR_INVALID_ARG:
        return "Invalid argument index";
    case VM_ERR_NULL_BYTECODE:
        return "Null bytecode";
    default:
        return "Unknown error";
    }
}
