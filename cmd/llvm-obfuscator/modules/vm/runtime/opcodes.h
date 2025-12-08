/**
 * VM Opcodes - Custom bytecode instruction set
 *
 * This file defines the opcode enum for the embedded VM interpreter.
 * The VM uses a stack-based architecture with 9 basic opcodes sufficient
 * for virtualizing simple arithmetic functions.
 *
 * Bytecode Format:
 *   - 1 byte opcode
 *   - Variable length operands (depends on opcode)
 *
 * Stack notation: a, b -> c means pop a and b, push c
 */

#ifndef VM_OPCODES_H
#define VM_OPCODES_H

/**
 * VM Opcode enumeration
 *
 * | Opcode      | Value | Operands       | Stack Effect      |
 * |-------------|-------|----------------|-------------------|
 * | VM_NOP      | 0x00  | none           | (no change)       |
 * | VM_PUSH     | 0x01  | i32 immediate  | -> val            |
 * | VM_POP      | 0x02  | none           | val ->            |
 * | VM_LOAD     | 0x03  | u8 reg_idx     | -> vregs[idx]     |
 * | VM_STORE    | 0x04  | u8 reg_idx     | val -> (to reg)   |
 * | VM_LOAD_ARG | 0x05  | u8 arg_idx     | -> args[idx]      |
 * | VM_ADD      | 0x10  | none           | a, b -> (a+b)     |
 * | VM_SUB      | 0x11  | none           | a, b -> (a-b)     |
 * | VM_XOR      | 0x12  | none           | a, b -> (a^b)     |
 * | VM_RET      | 0xFF  | none           | return top        |
 */
typedef enum {
    /* Control opcodes */
    VM_NOP      = 0x00,  /* No operation */

    /* Stack manipulation opcodes */
    VM_PUSH     = 0x01,  /* Push 32-bit immediate onto stack */
    VM_POP      = 0x02,  /* Pop and discard top of stack */
    VM_LOAD     = 0x03,  /* Load from virtual register to stack */
    VM_STORE    = 0x04,  /* Store from stack to virtual register */
    VM_LOAD_ARG = 0x05,  /* Load function argument to stack */

    /* Arithmetic opcodes (0x10-0x1F reserved) */
    VM_ADD      = 0x10,  /* Pop two, push sum */
    VM_SUB      = 0x11,  /* Pop two, push difference (a - b) */
    VM_XOR      = 0x12,  /* Pop two, push XOR */

    /* Control flow opcodes (0xF0-0xFF reserved) */
    VM_RET      = 0xFF   /* Return top of stack, exit VM */
} VMOpcode;

/**
 * Bytecode format examples:
 *
 * VM_NOP:
 *   [0x00]
 *
 * VM_PUSH 5:
 *   [0x01, 0x05, 0x00, 0x00, 0x00]  (little-endian i32)
 *
 * VM_LOAD vreg[2]:
 *   [0x03, 0x02]
 *
 * VM_STORE vreg[0]:
 *   [0x04, 0x00]
 *
 * VM_LOAD_ARG arg[1]:
 *   [0x05, 0x01]
 *
 * VM_ADD:
 *   [0x10]
 *
 * VM_RET:
 *   [0xFF]
 *
 * Example: add(a, b) -> a + b
 *   [VM_LOAD_ARG, 0x00,   // push arg[0]
 *    VM_LOAD_ARG, 0x01,   // push arg[1]
 *    VM_ADD,              // pop two, push sum
 *    VM_RET]              // return result
 */

#endif /* VM_OPCODES_H */
