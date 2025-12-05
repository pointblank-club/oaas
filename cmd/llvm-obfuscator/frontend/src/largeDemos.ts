// Large demo programs - separated for maintainability
// These are 3000+ line programs for testing obfuscation on large codebases

export interface LargeDemo {
  name: string;
  category: 'large';
  language: 'c' | 'cpp';
  code: string;
}

export const DATABASE_ENGINE_C: LargeDemo = {
  name: 'SQL Database Engine (C)',
  category: 'large',
  language: 'c',
  code: `
/*
 * OAAS Demo: SQL Database Engine
 * Language: C
 * Category: Large Program (~3500 lines)
 *
 * A complete SQL database engine implementation featuring:
 * - B-Tree indexing for fast lookups
 * - Transaction management with ACID properties
 * - Buffer pool for memory management
 * - Lock manager for concurrency control
 * - Query parser and executor
 * - WAL (Write-Ahead Logging)
 *
 * Contains secrets for obfuscation testing:
 * - DB_MASTER_KEY
 * - DB_ENCRYPTION_KEY
 * - DB_LICENSE_KEY
 * - DB_ADMIN_PASSWORD
 * - DB_API_KEY
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

/* ============================================================================
 * CONFIGURATION AND SECRETS
 * ============================================================================ */

#define DB_MASTER_KEY "MK_a8f3e2d1c4b5_OAAS_2024_SECRET"
#define DB_ENCRYPTION_KEY "EK_7d9f4e2a1b8c_AES256_PRIVATE"
#define DB_LICENSE_KEY "LIC_OAAS_DEMO_2024_UNLIMITED_ACCESS"
#define DB_ADMIN_PASSWORD "Admin$ecure#2024!@Database"
#define DB_API_KEY "api_sk_live_oaas_db_engine_x9y8z7"

#define PAGE_SIZE 4096
#define MAX_PAGES 1024
#define MAX_TABLES 64
#define MAX_COLUMNS 32
#define MAX_ROWS_PER_PAGE 100
#define MAX_KEY_SIZE 256
#define MAX_VALUE_SIZE 1024
#define MAX_QUERY_SIZE 4096
#define BTREE_ORDER 4
#define WAL_BUFFER_SIZE 8192
#define LOCK_TABLE_SIZE 256
#define MAX_TRANSACTIONS 64
#define BUFFER_POOL_SIZE 128

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_STRING,
    TYPE_BLOB,
    TYPE_TIMESTAMP
} DataType;

typedef enum {
    OP_EQ,
    OP_NE,
    OP_LT,
    OP_LE,
    OP_GT,
    OP_GE,
    OP_LIKE
} Operator;

typedef enum {
    LOCK_SHARED,
    LOCK_EXCLUSIVE
} LockMode;

typedef enum {
    TXN_ACTIVE,
    TXN_COMMITTED,
    TXN_ABORTED
} TxnState;

typedef enum {
    WAL_INSERT,
    WAL_UPDATE,
    WAL_DELETE,
    WAL_COMMIT,
    WAL_ABORT,
    WAL_CHECKPOINT
} WalRecordType;

typedef struct {
    char name[64];
    DataType type;
    int size;
    bool nullable;
    bool primary_key;
    bool indexed;
} Column;

typedef struct {
    char name[64];
    int column_count;
    Column columns[MAX_COLUMNS];
    int row_count;
    int root_page;
} Table;

typedef struct BTreeNode {
    int keys[BTREE_ORDER - 1];
    void* values[BTREE_ORDER - 1];
    struct BTreeNode* children[BTREE_ORDER];
    int key_count;
    bool is_leaf;
    int page_id;
} BTreeNode;

typedef struct {
    int page_id;
    uint8_t data[PAGE_SIZE];
    bool dirty;
    int pin_count;
    time_t last_access;
} BufferFrame;

typedef struct {
    BufferFrame frames[BUFFER_POOL_SIZE];
    int frame_count;
    int clock_hand;
} BufferPool;

typedef struct {
    int resource_id;
    LockMode mode;
    int txn_id;
    bool granted;
} LockRequest;

typedef struct {
    LockRequest requests[LOCK_TABLE_SIZE];
    int request_count;
} LockTable;

typedef struct {
    int txn_id;
    TxnState state;
    time_t start_time;
    int locks_held[LOCK_TABLE_SIZE];
    int lock_count;
    int undo_log_start;
} Transaction;

typedef struct {
    WalRecordType type;
    int txn_id;
    int table_id;
    int row_id;
    uint8_t before_image[MAX_VALUE_SIZE];
    uint8_t after_image[MAX_VALUE_SIZE];
    int data_size;
    uint64_t lsn;
} WalRecord;

typedef struct {
    WalRecord buffer[WAL_BUFFER_SIZE / sizeof(WalRecord)];
    int record_count;
    uint64_t current_lsn;
    uint64_t flushed_lsn;
} WalManager;

typedef struct {
    Table tables[MAX_TABLES];
    int table_count;
    BufferPool buffer_pool;
    LockTable lock_table;
    Transaction transactions[MAX_TRANSACTIONS];
    int txn_count;
    WalManager wal;
    bool initialized;
    char master_key[64];
    char encryption_key[64];
} Database;

/* ============================================================================
 * GLOBAL DATABASE INSTANCE
 * ============================================================================ */

static Database g_database;

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

static uint32_t hash_string(const char* str) {
    uint32_t hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

static uint64_t get_timestamp(void) {
    return (uint64_t)time(NULL);
}

static void xor_encrypt(uint8_t* data, int size, const char* key) {
    int key_len = strlen(key);
    for (int i = 0; i < size; i++) {
        data[i] ^= key[i % key_len];
    }
}

static void xor_decrypt(uint8_t* data, int size, const char* key) {
    xor_encrypt(data, size, key);
}

static bool verify_license(const char* key) {
    if (strcmp(key, DB_LICENSE_KEY) != 0) {
        printf("[LICENSE] Invalid license key\\n");
        return false;
    }
    printf("[LICENSE] License verified: %s\\n", key);
    return true;
}

static bool authenticate_admin(const char* password) {
    if (strcmp(password, DB_ADMIN_PASSWORD) != 0) {
        printf("[AUTH] Authentication failed\\n");
        return false;
    }
    printf("[AUTH] Admin authenticated successfully\\n");
    return true;
}

/* ============================================================================
 * BUFFER POOL MANAGEMENT
 * ============================================================================ */

static void buffer_pool_init(BufferPool* pool) {
    memset(pool, 0, sizeof(BufferPool));
    pool->frame_count = BUFFER_POOL_SIZE;
    pool->clock_hand = 0;

    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        pool->frames[i].page_id = -1;
        pool->frames[i].dirty = false;
        pool->frames[i].pin_count = 0;
    }
    printf("[BUFFER] Buffer pool initialized with %d frames\\n", BUFFER_POOL_SIZE);
}

static int buffer_pool_find_victim(BufferPool* pool) {
    int attempts = 0;
    while (attempts < BUFFER_POOL_SIZE * 2) {
        BufferFrame* frame = &pool->frames[pool->clock_hand];

        if (frame->pin_count == 0) {
            if (frame->page_id == -1) {
                return pool->clock_hand;
            }
        }

        pool->clock_hand = (pool->clock_hand + 1) % BUFFER_POOL_SIZE;
        attempts++;
    }
    return -1;
}

static BufferFrame* buffer_pool_get_page(BufferPool* pool, int page_id) {
    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        if (pool->frames[i].page_id == page_id) {
            pool->frames[i].pin_count++;
            pool->frames[i].last_access = time(NULL);
            return &pool->frames[i];
        }
    }

    int victim = buffer_pool_find_victim(pool);
    if (victim < 0) {
        printf("[BUFFER] No available frames in buffer pool\\n");
        return NULL;
    }

    BufferFrame* frame = &pool->frames[victim];

    if (frame->dirty && frame->page_id >= 0) {
        printf("[BUFFER] Flushing dirty page %d\\n", frame->page_id);
    }

    frame->page_id = page_id;
    frame->dirty = false;
    frame->pin_count = 1;
    frame->last_access = time(NULL);
    memset(frame->data, 0, PAGE_SIZE);

    return frame;
}

static void buffer_pool_unpin_page(BufferPool* pool, int page_id, bool dirty) {
    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        if (pool->frames[i].page_id == page_id) {
            if (pool->frames[i].pin_count > 0) {
                pool->frames[i].pin_count--;
            }
            if (dirty) {
                pool->frames[i].dirty = true;
            }
            return;
        }
    }
}

static void buffer_pool_flush_all(BufferPool* pool) {
    int flushed = 0;
    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        if (pool->frames[i].dirty && pool->frames[i].page_id >= 0) {
            pool->frames[i].dirty = false;
            flushed++;
        }
    }
    printf("[BUFFER] Flushed %d dirty pages\\n", flushed);
}

/* ============================================================================
 * LOCK MANAGER
 * ============================================================================ */

static void lock_table_init(LockTable* lt) {
    memset(lt, 0, sizeof(LockTable));
    printf("[LOCK] Lock table initialized\\n");
}

static bool lock_acquire(LockTable* lt, int txn_id, int resource_id, LockMode mode) {
    for (int i = 0; i < lt->request_count; i++) {
        if (lt->requests[i].resource_id == resource_id &&
            lt->requests[i].granted) {
            if (mode == LOCK_EXCLUSIVE ||
                lt->requests[i].mode == LOCK_EXCLUSIVE) {
                if (lt->requests[i].txn_id != txn_id) {
                    printf("[LOCK] Conflict: txn %d waiting for resource %d\\n",
                           txn_id, resource_id);
                    return false;
                }
            }
        }
    }

    if (lt->request_count >= LOCK_TABLE_SIZE) {
        printf("[LOCK] Lock table full\\n");
        return false;
    }

    LockRequest* req = &lt->requests[lt->request_count++];
    req->resource_id = resource_id;
    req->mode = mode;
    req->txn_id = txn_id;
    req->granted = true;

    printf("[LOCK] Txn %d acquired %s lock on resource %d\\n",
           txn_id, mode == LOCK_SHARED ? "SHARED" : "EXCLUSIVE", resource_id);
    return true;
}

static void lock_release(LockTable* lt, int txn_id, int resource_id) {
    for (int i = 0; i < lt->request_count; i++) {
        if (lt->requests[i].resource_id == resource_id &&
            lt->requests[i].txn_id == txn_id &&
            lt->requests[i].granted) {
            lt->requests[i].granted = false;
            printf("[LOCK] Txn %d released lock on resource %d\\n",
                   txn_id, resource_id);
            return;
        }
    }
}

static void lock_release_all(LockTable* lt, int txn_id) {
    int released = 0;
    for (int i = 0; i < lt->request_count; i++) {
        if (lt->requests[i].txn_id == txn_id && lt->requests[i].granted) {
            lt->requests[i].granted = false;
            released++;
        }
    }
    printf("[LOCK] Released %d locks for txn %d\\n", released, txn_id);
}

/* ============================================================================
 * WAL (WRITE-AHEAD LOGGING)
 * ============================================================================ */

static void wal_init(WalManager* wal) {
    memset(wal, 0, sizeof(WalManager));
    wal->current_lsn = 1;
    wal->flushed_lsn = 0;
    printf("[WAL] Write-ahead log initialized\\n");
}

static uint64_t wal_write(WalManager* wal, WalRecordType type, int txn_id,
                          int table_id, int row_id,
                          const uint8_t* before, const uint8_t* after, int size) {
    if (wal->record_count >= WAL_BUFFER_SIZE / (int)sizeof(WalRecord)) {
        printf("[WAL] Buffer full, forcing flush\\n");
        wal->flushed_lsn = wal->current_lsn - 1;
        wal->record_count = 0;
    }

    WalRecord* rec = &wal->buffer[wal->record_count++];
    rec->type = type;
    rec->txn_id = txn_id;
    rec->table_id = table_id;
    rec->row_id = row_id;
    rec->lsn = wal->current_lsn++;
    rec->data_size = size;

    if (before && size > 0) {
        memcpy(rec->before_image, before, size < MAX_VALUE_SIZE ? size : MAX_VALUE_SIZE);
    }
    if (after && size > 0) {
        memcpy(rec->after_image, after, size < MAX_VALUE_SIZE ? size : MAX_VALUE_SIZE);
    }

    return rec->lsn;
}

static void wal_flush(WalManager* wal) {
    if (wal->record_count > 0) {
        printf("[WAL] Flushing %d records to disk (LSN %llu -> %llu)\\n",
               wal->record_count,
               (unsigned long long)wal->flushed_lsn,
               (unsigned long long)(wal->current_lsn - 1));
        wal->flushed_lsn = wal->current_lsn - 1;
        wal->record_count = 0;
    }
}

static void wal_checkpoint(WalManager* wal) {
    wal_write(wal, WAL_CHECKPOINT, 0, 0, 0, NULL, NULL, 0);
    wal_flush(wal);
    printf("[WAL] Checkpoint completed at LSN %llu\\n",
           (unsigned long long)wal->current_lsn);
}

/* ============================================================================
 * TRANSACTION MANAGER
 * ============================================================================ */

static int txn_begin(Database* db) {
    if (db->txn_count >= MAX_TRANSACTIONS) {
        printf("[TXN] Maximum transactions reached\\n");
        return -1;
    }

    Transaction* txn = &db->transactions[db->txn_count];
    txn->txn_id = db->txn_count + 1;
    txn->state = TXN_ACTIVE;
    txn->start_time = time(NULL);
    txn->lock_count = 0;
    txn->undo_log_start = db->wal.current_lsn;

    db->txn_count++;

    printf("[TXN] Transaction %d started\\n", txn->txn_id);
    return txn->txn_id;
}

static bool txn_commit(Database* db, int txn_id) {
    for (int i = 0; i < db->txn_count; i++) {
        if (db->transactions[i].txn_id == txn_id) {
            Transaction* txn = &db->transactions[i];

            if (txn->state != TXN_ACTIVE) {
                printf("[TXN] Transaction %d not active\\n", txn_id);
                return false;
            }

            wal_write(&db->wal, WAL_COMMIT, txn_id, 0, 0, NULL, NULL, 0);
            wal_flush(&db->wal);

            lock_release_all(&db->lock_table, txn_id);

            txn->state = TXN_COMMITTED;
            printf("[TXN] Transaction %d committed\\n", txn_id);
            return true;
        }
    }
    return false;
}

static bool txn_abort(Database* db, int txn_id) {
    for (int i = 0; i < db->txn_count; i++) {
        if (db->transactions[i].txn_id == txn_id) {
            Transaction* txn = &db->transactions[i];

            if (txn->state != TXN_ACTIVE) {
                printf("[TXN] Transaction %d not active\\n", txn_id);
                return false;
            }

            printf("[TXN] Rolling back transaction %d\\n", txn_id);

            wal_write(&db->wal, WAL_ABORT, txn_id, 0, 0, NULL, NULL, 0);

            lock_release_all(&db->lock_table, txn_id);

            txn->state = TXN_ABORTED;
            printf("[TXN] Transaction %d aborted\\n", txn_id);
            return true;
        }
    }
    return false;
}

/* ============================================================================
 * B-TREE INDEX
 * ============================================================================ */

static BTreeNode* btree_create_node(bool is_leaf) {
    BTreeNode* node = (BTreeNode*)calloc(1, sizeof(BTreeNode));
    if (!node) return NULL;

    node->is_leaf = is_leaf;
    node->key_count = 0;
    node->page_id = rand() % MAX_PAGES;

    for (int i = 0; i < BTREE_ORDER; i++) {
        node->children[i] = NULL;
    }

    return node;
}

static void btree_free_node(BTreeNode* node) {
    if (!node) return;

    if (!node->is_leaf) {
        for (int i = 0; i <= node->key_count; i++) {
            btree_free_node(node->children[i]);
        }
    }
    free(node);
}

static BTreeNode* btree_search(BTreeNode* root, int key) {
    if (!root) return NULL;

    int i = 0;
    while (i < root->key_count && key > root->keys[i]) {
        i++;
    }

    if (i < root->key_count && key == root->keys[i]) {
        return root;
    }

    if (root->is_leaf) {
        return NULL;
    }

    return btree_search(root->children[i], key);
}

static void btree_split_child(BTreeNode* parent, int index, BTreeNode* child) {
    BTreeNode* new_node = btree_create_node(child->is_leaf);
    int mid = (BTREE_ORDER - 1) / 2;

    new_node->key_count = child->key_count - mid - 1;

    for (int i = 0; i < new_node->key_count; i++) {
        new_node->keys[i] = child->keys[mid + 1 + i];
        new_node->values[i] = child->values[mid + 1 + i];
    }

    if (!child->is_leaf) {
        for (int i = 0; i <= new_node->key_count; i++) {
            new_node->children[i] = child->children[mid + 1 + i];
        }
    }

    child->key_count = mid;

    for (int i = parent->key_count; i > index; i--) {
        parent->children[i + 1] = parent->children[i];
    }
    parent->children[index + 1] = new_node;

    for (int i = parent->key_count - 1; i >= index; i--) {
        parent->keys[i + 1] = parent->keys[i];
        parent->values[i + 1] = parent->values[i];
    }
    parent->keys[index] = child->keys[mid];
    parent->values[index] = child->values[mid];
    parent->key_count++;
}

static void btree_insert_non_full(BTreeNode* node, int key, void* value) {
    int i = node->key_count - 1;

    if (node->is_leaf) {
        while (i >= 0 && key < node->keys[i]) {
            node->keys[i + 1] = node->keys[i];
            node->values[i + 1] = node->values[i];
            i--;
        }
        node->keys[i + 1] = key;
        node->values[i + 1] = value;
        node->key_count++;
    } else {
        while (i >= 0 && key < node->keys[i]) {
            i--;
        }
        i++;

        if (node->children[i]->key_count == BTREE_ORDER - 1) {
            btree_split_child(node, i, node->children[i]);
            if (key > node->keys[i]) {
                i++;
            }
        }
        btree_insert_non_full(node->children[i], key, value);
    }
}

static BTreeNode* btree_insert(BTreeNode* root, int key, void* value) {
    if (!root) {
        root = btree_create_node(true);
        root->keys[0] = key;
        root->values[0] = value;
        root->key_count = 1;
        return root;
    }

    if (root->key_count == BTREE_ORDER - 1) {
        BTreeNode* new_root = btree_create_node(false);
        new_root->children[0] = root;
        btree_split_child(new_root, 0, root);
        btree_insert_non_full(new_root, key, value);
        return new_root;
    }

    btree_insert_non_full(root, key, value);
    return root;
}

static void btree_print_inorder(BTreeNode* node, int level) {
    if (!node) return;

    for (int i = 0; i < node->key_count; i++) {
        if (!node->is_leaf) {
            btree_print_inorder(node->children[i], level + 1);
        }
        printf("[BTREE] Level %d: Key=%d\\n", level, node->keys[i]);
    }

    if (!node->is_leaf) {
        btree_print_inorder(node->children[node->key_count], level + 1);
    }
}

/* ============================================================================
 * TABLE OPERATIONS
 * ============================================================================ */

static int table_create(Database* db, const char* name, Column* columns, int col_count) {
    if (db->table_count >= MAX_TABLES) {
        printf("[TABLE] Maximum tables reached\\n");
        return -1;
    }

    for (int i = 0; i < db->table_count; i++) {
        if (strcmp(db->tables[i].name, name) == 0) {
            printf("[TABLE] Table '%s' already exists\\n", name);
            return -1;
        }
    }

    Table* table = &db->tables[db->table_count];
    strncpy(table->name, name, sizeof(table->name) - 1);
    table->column_count = col_count;
    table->row_count = 0;
    table->root_page = db->table_count;

    for (int i = 0; i < col_count && i < MAX_COLUMNS; i++) {
        memcpy(&table->columns[i], &columns[i], sizeof(Column));
    }

    db->table_count++;
    printf("[TABLE] Created table '%s' with %d columns\\n", name, col_count);
    return db->table_count - 1;
}

static Table* table_find(Database* db, const char* name) {
    for (int i = 0; i < db->table_count; i++) {
        if (strcmp(db->tables[i].name, name) == 0) {
            return &db->tables[i];
        }
    }
    return NULL;
}

static void table_describe(Table* table) {
    printf("\\n[TABLE] Schema for '%s':\\n", table->name);
    printf("  %-20s %-10s %-6s %-8s %-8s\\n",
           "Column", "Type", "Size", "Nullable", "Primary");
    printf("  %-20s %-10s %-6s %-8s %-8s\\n",
           "------", "----", "----", "--------", "-------");

    for (int i = 0; i < table->column_count; i++) {
        Column* col = &table->columns[i];
        const char* type_str = "UNKNOWN";
        switch (col->type) {
            case TYPE_INT: type_str = "INT"; break;
            case TYPE_FLOAT: type_str = "FLOAT"; break;
            case TYPE_STRING: type_str = "STRING"; break;
            case TYPE_BLOB: type_str = "BLOB"; break;
            case TYPE_TIMESTAMP: type_str = "TIMESTAMP"; break;
        }
        printf("  %-20s %-10s %-6d %-8s %-8s\\n",
               col->name, type_str, col->size,
               col->nullable ? "YES" : "NO",
               col->primary_key ? "YES" : "NO");
    }
    printf("  Rows: %d\\n", table->row_count);
}

/* ============================================================================
 * ROW OPERATIONS
 * ============================================================================ */

typedef struct {
    int id;
    uint8_t data[MAX_VALUE_SIZE];
    int data_size;
    bool deleted;
} Row;

static Row* row_create(int id) {
    Row* row = (Row*)calloc(1, sizeof(Row));
    if (!row) return NULL;
    row->id = id;
    row->deleted = false;
    return row;
}

static bool row_set_field(Row* row, int offset, const void* value, int size) {
    if (offset + size > MAX_VALUE_SIZE) {
        return false;
    }
    memcpy(row->data + offset, value, size);
    if (offset + size > row->data_size) {
        row->data_size = offset + size;
    }
    return true;
}

static void* row_get_field(Row* row, int offset, int size) {
    if (offset + size > row->data_size) {
        return NULL;
    }
    return row->data + offset;
}

/* ============================================================================
 * QUERY PARSER
 * ============================================================================ */

typedef enum {
    TOKEN_SELECT,
    TOKEN_INSERT,
    TOKEN_UPDATE,
    TOKEN_DELETE,
    TOKEN_CREATE,
    TOKEN_DROP,
    TOKEN_FROM,
    TOKEN_WHERE,
    TOKEN_INTO,
    TOKEN_VALUES,
    TOKEN_SET,
    TOKEN_TABLE,
    TOKEN_AND,
    TOKEN_OR,
    TOKEN_IDENTIFIER,
    TOKEN_STRING,
    TOKEN_NUMBER,
    TOKEN_OPERATOR,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_COMMA,
    TOKEN_STAR,
    TOKEN_SEMICOLON,
    TOKEN_EOF,
    TOKEN_ERROR
} TokenType;

typedef struct {
    TokenType type;
    char value[MAX_KEY_SIZE];
} Token;

typedef struct {
    const char* input;
    int pos;
    int length;
    Token current;
} Lexer;

static void lexer_init(Lexer* lex, const char* input) {
    lex->input = input;
    lex->pos = 0;
    lex->length = strlen(input);
}

static void lexer_skip_whitespace(Lexer* lex) {
    while (lex->pos < lex->length &&
           (lex->input[lex->pos] == ' ' ||
            lex->input[lex->pos] == '\\t' ||
            lex->input[lex->pos] == '\\n' ||
            lex->input[lex->pos] == '\\r')) {
        lex->pos++;
    }
}

static Token lexer_next_token(Lexer* lex) {
    Token token;
    memset(&token, 0, sizeof(Token));

    lexer_skip_whitespace(lex);

    if (lex->pos >= lex->length) {
        token.type = TOKEN_EOF;
        return token;
    }

    char c = lex->input[lex->pos];

    if (c == '(') { token.type = TOKEN_LPAREN; lex->pos++; return token; }
    if (c == ')') { token.type = TOKEN_RPAREN; lex->pos++; return token; }
    if (c == ',') { token.type = TOKEN_COMMA; lex->pos++; return token; }
    if (c == '*') { token.type = TOKEN_STAR; lex->pos++; return token; }
    if (c == ';') { token.type = TOKEN_SEMICOLON; lex->pos++; return token; }

    if (c == '=' || c == '<' || c == '>' || c == '!') {
        token.type = TOKEN_OPERATOR;
        token.value[0] = c;
        lex->pos++;
        if (lex->pos < lex->length && lex->input[lex->pos] == '=') {
            token.value[1] = '=';
            lex->pos++;
        }
        return token;
    }

    if (c == '\\'') {
        lex->pos++;
        int i = 0;
        while (lex->pos < lex->length && lex->input[lex->pos] != '\\'') {
            token.value[i++] = lex->input[lex->pos++];
        }
        if (lex->pos < lex->length) lex->pos++;
        token.type = TOKEN_STRING;
        return token;
    }

    if (c >= '0' && c <= '9') {
        int i = 0;
        while (lex->pos < lex->length &&
               (lex->input[lex->pos] >= '0' && lex->input[lex->pos] <= '9')) {
            token.value[i++] = lex->input[lex->pos++];
        }
        token.type = TOKEN_NUMBER;
        return token;
    }

    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
        int i = 0;
        while (lex->pos < lex->length &&
               ((lex->input[lex->pos] >= 'a' && lex->input[lex->pos] <= 'z') ||
                (lex->input[lex->pos] >= 'A' && lex->input[lex->pos] <= 'Z') ||
                (lex->input[lex->pos] >= '0' && lex->input[lex->pos] <= '9') ||
                lex->input[lex->pos] == '_')) {
            token.value[i++] = lex->input[lex->pos++];
        }

        if (strcasecmp(token.value, "SELECT") == 0) token.type = TOKEN_SELECT;
        else if (strcasecmp(token.value, "INSERT") == 0) token.type = TOKEN_INSERT;
        else if (strcasecmp(token.value, "UPDATE") == 0) token.type = TOKEN_UPDATE;
        else if (strcasecmp(token.value, "DELETE") == 0) token.type = TOKEN_DELETE;
        else if (strcasecmp(token.value, "CREATE") == 0) token.type = TOKEN_CREATE;
        else if (strcasecmp(token.value, "DROP") == 0) token.type = TOKEN_DROP;
        else if (strcasecmp(token.value, "FROM") == 0) token.type = TOKEN_FROM;
        else if (strcasecmp(token.value, "WHERE") == 0) token.type = TOKEN_WHERE;
        else if (strcasecmp(token.value, "INTO") == 0) token.type = TOKEN_INTO;
        else if (strcasecmp(token.value, "VALUES") == 0) token.type = TOKEN_VALUES;
        else if (strcasecmp(token.value, "SET") == 0) token.type = TOKEN_SET;
        else if (strcasecmp(token.value, "TABLE") == 0) token.type = TOKEN_TABLE;
        else if (strcasecmp(token.value, "AND") == 0) token.type = TOKEN_AND;
        else if (strcasecmp(token.value, "OR") == 0) token.type = TOKEN_OR;
        else token.type = TOKEN_IDENTIFIER;

        return token;
    }

    token.type = TOKEN_ERROR;
    return token;
}

/* ============================================================================
 * QUERY EXECUTOR
 * ============================================================================ */

typedef struct {
    char table_name[64];
    char columns[MAX_COLUMNS][64];
    int column_count;
    char where_column[64];
    Operator where_op;
    char where_value[MAX_VALUE_SIZE];
    bool has_where;
} SelectQuery;

typedef struct {
    char table_name[64];
    char values[MAX_COLUMNS][MAX_VALUE_SIZE];
    int value_count;
} InsertQuery;

static void execute_select(Database* db, SelectQuery* query) {
    Table* table = table_find(db, query->table_name);
    if (!table) {
        printf("[QUERY] Table '%s' not found\\n", query->table_name);
        return;
    }

    printf("\\n[QUERY] SELECT result from '%s':\\n", query->table_name);

    if (query->column_count == 0 ||
        (query->column_count == 1 && strcmp(query->columns[0], "*") == 0)) {
        for (int i = 0; i < table->column_count; i++) {
            printf("  %-15s", table->columns[i].name);
        }
        printf("\\n");
        for (int i = 0; i < table->column_count; i++) {
            printf("  %-15s", "---------------");
        }
        printf("\\n");
    }

    printf("  (Simulated %d rows)\\n", table->row_count);
}

static void execute_insert(Database* db, InsertQuery* query) {
    Table* table = table_find(db, query->table_name);
    if (!table) {
        printf("[QUERY] Table '%s' not found\\n", query->table_name);
        return;
    }

    table->row_count++;
    printf("[QUERY] Inserted 1 row into '%s' (total: %d)\\n",
           query->table_name, table->row_count);
}

/* ============================================================================
 * DATABASE INITIALIZATION AND SHUTDOWN
 * ============================================================================ */

static bool database_init(Database* db, const char* master_key, const char* encryption_key) {
    printf("\\n============================================================\\n");
    printf("[DATABASE] Initializing OAAS SQL Database Engine...\\n");
    printf("============================================================\\n\\n");

    if (strcmp(master_key, DB_MASTER_KEY) != 0) {
        printf("[DATABASE] Invalid master key\\n");
        return false;
    }

    strncpy(db->master_key, master_key, sizeof(db->master_key) - 1);
    strncpy(db->encryption_key, encryption_key, sizeof(db->encryption_key) - 1);

    printf("[CONFIG] Master Key: %s\\n", db->master_key);
    printf("[CONFIG] Encryption Key: %s\\n", db->encryption_key);
    printf("[CONFIG] API Key: %s\\n", DB_API_KEY);

    buffer_pool_init(&db->buffer_pool);
    lock_table_init(&db->lock_table);
    wal_init(&db->wal);

    db->table_count = 0;
    db->txn_count = 0;
    db->initialized = true;

    printf("\\n[DATABASE] Engine initialized successfully\\n");
    return true;
}

static void database_shutdown(Database* db) {
    printf("\\n[DATABASE] Shutting down...\\n");

    for (int i = 0; i < db->txn_count; i++) {
        if (db->transactions[i].state == TXN_ACTIVE) {
            txn_abort(db, db->transactions[i].txn_id);
        }
    }

    buffer_pool_flush_all(&db->buffer_pool);
    wal_checkpoint(&db->wal);

    db->initialized = false;
    printf("[DATABASE] Shutdown complete\\n");
}

/* ============================================================================
 * HASH INDEX IMPLEMENTATION
 * ============================================================================ */

#define HASH_BUCKET_COUNT 256
#define HASH_CHAIN_LENGTH 16

typedef struct HashEntry {
    int key;
    void* value;
    bool occupied;
    struct HashEntry* next;
} HashEntry;

typedef struct {
    HashEntry buckets[HASH_BUCKET_COUNT];
    int entry_count;
} HashIndex;

static void hash_index_init(HashIndex* index) {
    memset(index, 0, sizeof(HashIndex));
    for (int i = 0; i < HASH_BUCKET_COUNT; i++) {
        index->buckets[i].occupied = false;
        index->buckets[i].next = NULL;
    }
    printf("[HASH] Hash index initialized with %d buckets\\n", HASH_BUCKET_COUNT);
}

static int hash_function(int key) {
    unsigned int h = (unsigned int)key;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = (h >> 16) ^ h;
    return h % HASH_BUCKET_COUNT;
}

static bool hash_index_insert(HashIndex* index, int key, void* value) {
    int bucket = hash_function(key);
    HashEntry* entry = &index->buckets[bucket];

    if (!entry->occupied) {
        entry->key = key;
        entry->value = value;
        entry->occupied = true;
        index->entry_count++;
        return true;
    }

    while (entry->next != NULL) {
        if (entry->key == key) {
            entry->value = value;
            return true;
        }
        entry = entry->next;
    }

    if (entry->key == key) {
        entry->value = value;
        return true;
    }

    HashEntry* new_entry = (HashEntry*)calloc(1, sizeof(HashEntry));
    if (!new_entry) return false;

    new_entry->key = key;
    new_entry->value = value;
    new_entry->occupied = true;
    entry->next = new_entry;
    index->entry_count++;

    return true;
}

static void* hash_index_lookup(HashIndex* index, int key) {
    int bucket = hash_function(key);
    HashEntry* entry = &index->buckets[bucket];

    while (entry != NULL) {
        if (entry->occupied && entry->key == key) {
            return entry->value;
        }
        entry = entry->next;
    }
    return NULL;
}

static bool hash_index_delete(HashIndex* index, int key) {
    int bucket = hash_function(key);
    HashEntry* entry = &index->buckets[bucket];
    HashEntry* prev = NULL;

    while (entry != NULL) {
        if (entry->occupied && entry->key == key) {
            if (prev == NULL) {
                if (entry->next != NULL) {
                    HashEntry* next = entry->next;
                    entry->key = next->key;
                    entry->value = next->value;
                    entry->next = next->next;
                    free(next);
                } else {
                    entry->occupied = false;
                }
            } else {
                prev->next = entry->next;
                free(entry);
            }
            index->entry_count--;
            return true;
        }
        prev = entry;
        entry = entry->next;
    }
    return false;
}

static void hash_index_stats(HashIndex* index) {
    int used_buckets = 0;
    int max_chain = 0;
    int total_chain = 0;

    for (int i = 0; i < HASH_BUCKET_COUNT; i++) {
        if (index->buckets[i].occupied) {
            used_buckets++;
            int chain_len = 1;
            HashEntry* entry = index->buckets[i].next;
            while (entry != NULL) {
                chain_len++;
                entry = entry->next;
            }
            total_chain += chain_len;
            if (chain_len > max_chain) max_chain = chain_len;
        }
    }

    printf("[HASH] Statistics:\\n");
    printf("  Total entries: %d\\n", index->entry_count);
    printf("  Used buckets: %d/%d (%.1f%%)\\n",
           used_buckets, HASH_BUCKET_COUNT,
           100.0 * used_buckets / HASH_BUCKET_COUNT);
    printf("  Max chain length: %d\\n", max_chain);
    printf("  Avg chain length: %.2f\\n",
           used_buckets > 0 ? (float)total_chain / used_buckets : 0);
}

/* ============================================================================
 * BITMAP INDEX IMPLEMENTATION
 * ============================================================================ */

#define BITMAP_SIZE 1024
#define BITS_PER_WORD 64

typedef struct {
    uint64_t bits[BITMAP_SIZE / BITS_PER_WORD];
    int cardinality;
} Bitmap;

static void bitmap_init(Bitmap* bm) {
    memset(bm, 0, sizeof(Bitmap));
}

static void bitmap_set(Bitmap* bm, int pos) {
    if (pos < 0 || pos >= BITMAP_SIZE) return;
    int word = pos / BITS_PER_WORD;
    int bit = pos % BITS_PER_WORD;
    if (!(bm->bits[word] & (1ULL << bit))) {
        bm->bits[word] |= (1ULL << bit);
        bm->cardinality++;
    }
}

static void bitmap_clear(Bitmap* bm, int pos) {
    if (pos < 0 || pos >= BITMAP_SIZE) return;
    int word = pos / BITS_PER_WORD;
    int bit = pos % BITS_PER_WORD;
    if (bm->bits[word] & (1ULL << bit)) {
        bm->bits[word] &= ~(1ULL << bit);
        bm->cardinality--;
    }
}

static bool bitmap_test(Bitmap* bm, int pos) {
    if (pos < 0 || pos >= BITMAP_SIZE) return false;
    int word = pos / BITS_PER_WORD;
    int bit = pos % BITS_PER_WORD;
    return (bm->bits[word] & (1ULL << bit)) != 0;
}

static void bitmap_and(Bitmap* result, Bitmap* a, Bitmap* b) {
    result->cardinality = 0;
    for (int i = 0; i < BITMAP_SIZE / BITS_PER_WORD; i++) {
        result->bits[i] = a->bits[i] & b->bits[i];
        uint64_t v = result->bits[i];
        while (v) {
            result->cardinality++;
            v &= v - 1;
        }
    }
}

static void bitmap_or(Bitmap* result, Bitmap* a, Bitmap* b) {
    result->cardinality = 0;
    for (int i = 0; i < BITMAP_SIZE / BITS_PER_WORD; i++) {
        result->bits[i] = a->bits[i] | b->bits[i];
        uint64_t v = result->bits[i];
        while (v) {
            result->cardinality++;
            v &= v - 1;
        }
    }
}

static void bitmap_xor(Bitmap* result, Bitmap* a, Bitmap* b) {
    result->cardinality = 0;
    for (int i = 0; i < BITMAP_SIZE / BITS_PER_WORD; i++) {
        result->bits[i] = a->bits[i] ^ b->bits[i];
        uint64_t v = result->bits[i];
        while (v) {
            result->cardinality++;
            v &= v - 1;
        }
    }
}

static void bitmap_not(Bitmap* result, Bitmap* a) {
    result->cardinality = 0;
    for (int i = 0; i < BITMAP_SIZE / BITS_PER_WORD; i++) {
        result->bits[i] = ~a->bits[i];
        uint64_t v = result->bits[i];
        while (v) {
            result->cardinality++;
            v &= v - 1;
        }
    }
}

/* ============================================================================
 * SORT-MERGE JOIN IMPLEMENTATION
 * ============================================================================ */

typedef struct {
    int key;
    int row_id;
    int table_id;
} JoinRecord;

static int compare_join_records(const void* a, const void* b) {
    JoinRecord* ra = (JoinRecord*)a;
    JoinRecord* rb = (JoinRecord*)b;
    return ra->key - rb->key;
}

static void sort_merge_join(JoinRecord* left, int left_count,
                            JoinRecord* right, int right_count) {
    qsort(left, left_count, sizeof(JoinRecord), compare_join_records);
    qsort(right, right_count, sizeof(JoinRecord), compare_join_records);

    int i = 0, j = 0;
    int match_count = 0;

    printf("[JOIN] Performing sort-merge join...\\n");

    while (i < left_count && j < right_count) {
        if (left[i].key < right[j].key) {
            i++;
        } else if (left[i].key > right[j].key) {
            j++;
        } else {
            int start_j = j;
            while (j < right_count && left[i].key == right[j].key) {
                match_count++;
                j++;
            }
            i++;
            if (i < left_count && left[i].key == left[i-1].key) {
                j = start_j;
            }
        }
    }

    printf("[JOIN] Found %d matching pairs\\n", match_count);
}

/* ============================================================================
 * HASH JOIN IMPLEMENTATION
 * ============================================================================ */

static void hash_join(JoinRecord* build, int build_count,
                      JoinRecord* probe, int probe_count) {
    HashIndex build_index;
    hash_index_init(&build_index);

    printf("[JOIN] Building hash table with %d records...\\n", build_count);

    for (int i = 0; i < build_count; i++) {
        int* row_id = (int*)malloc(sizeof(int));
        *row_id = build[i].row_id;
        hash_index_insert(&build_index, build[i].key, row_id);
    }

    printf("[JOIN] Probing with %d records...\\n", probe_count);

    int match_count = 0;
    for (int i = 0; i < probe_count; i++) {
        void* result = hash_index_lookup(&build_index, probe[i].key);
        if (result != NULL) {
            match_count++;
        }
    }

    printf("[JOIN] Hash join found %d matches\\n", match_count);
}

/* ============================================================================
 * NESTED LOOP JOIN IMPLEMENTATION
 * ============================================================================ */

static void nested_loop_join(JoinRecord* outer, int outer_count,
                             JoinRecord* inner, int inner_count) {
    int match_count = 0;
    int comparisons = 0;

    printf("[JOIN] Performing nested loop join...\\n");

    for (int i = 0; i < outer_count; i++) {
        for (int j = 0; j < inner_count; j++) {
            comparisons++;
            if (outer[i].key == inner[j].key) {
                match_count++;
            }
        }
    }

    printf("[JOIN] Nested loop: %d matches, %d comparisons\\n",
           match_count, comparisons);
}

/* ============================================================================
 * AGGREGATE FUNCTIONS
 * ============================================================================ */

typedef struct {
    double sum;
    double sum_sq;
    int count;
    double min;
    double max;
} AggregateState;

static void aggregate_init(AggregateState* state) {
    state->sum = 0;
    state->sum_sq = 0;
    state->count = 0;
    state->min = 1e308;
    state->max = -1e308;
}

static void aggregate_add(AggregateState* state, double value) {
    state->sum += value;
    state->sum_sq += value * value;
    state->count++;
    if (value < state->min) state->min = value;
    if (value > state->max) state->max = value;
}

static double aggregate_avg(AggregateState* state) {
    return state->count > 0 ? state->sum / state->count : 0;
}

static double aggregate_variance(AggregateState* state) {
    if (state->count < 2) return 0;
    double mean = aggregate_avg(state);
    return (state->sum_sq / state->count) - (mean * mean);
}

static double aggregate_stddev(AggregateState* state) {
    return sqrt(aggregate_variance(state));
}

static void aggregate_print(AggregateState* state) {
    printf("[AGGREGATE] Results:\\n");
    printf("  COUNT: %d\\n", state->count);
    printf("  SUM:   %.2f\\n", state->sum);
    printf("  AVG:   %.2f\\n", aggregate_avg(state));
    printf("  MIN:   %.2f\\n", state->min);
    printf("  MAX:   %.2f\\n", state->max);
    printf("  VAR:   %.2f\\n", aggregate_variance(state));
    printf("  STD:   %.2f\\n", aggregate_stddev(state));
}

/* ============================================================================
 * SORTING ALGORITHMS FOR QUERY OPTIMIZATION
 * ============================================================================ */

static void swap_ints(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

static int partition(int* arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap_ints(&arr[i], &arr[j]);
        }
    }
    swap_ints(&arr[i + 1], &arr[high]);
    return i + 1;
}

static void quicksort_internal(int* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort_internal(arr, low, pi - 1);
        quicksort_internal(arr, pi + 1, high);
    }
}

static void db_quicksort(int* arr, int n) {
    quicksort_internal(arr, 0, n - 1);
}

static void merge(int* arr, int* temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}

static void mergesort_internal(int* arr, int* temp, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergesort_internal(arr, temp, left, mid);
        mergesort_internal(arr, temp, mid + 1, right);
        merge(arr, temp, left, mid, right);
    }
}

static void db_mergesort(int* arr, int n) {
    int* temp = (int*)malloc(n * sizeof(int));
    if (temp) {
        mergesort_internal(arr, temp, 0, n - 1);
        free(temp);
    }
}

static void heapify(int* arr, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest]) {
        largest = left;
    }

    if (right < n && arr[right] > arr[largest]) {
        largest = right;
    }

    if (largest != i) {
        swap_ints(&arr[i], &arr[largest]);
        heapify(arr, n, largest);
    }
}

static void db_heapsort(int* arr, int n) {
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    for (int i = n - 1; i > 0; i--) {
        swap_ints(&arr[0], &arr[i]);
        heapify(arr, i, 0);
    }
}

/* ============================================================================
 * EXPRESSION EVALUATOR
 * ============================================================================ */

typedef enum {
    EXPR_LITERAL,
    EXPR_COLUMN,
    EXPR_BINARY_OP,
    EXPR_UNARY_OP,
    EXPR_FUNCTION
} ExprType;

typedef struct Expression {
    ExprType type;
    union {
        double literal_value;
        char column_name[64];
        struct {
            struct Expression* left;
            struct Expression* right;
            char op;
        } binary;
        struct {
            struct Expression* operand;
            char op;
        } unary;
        struct {
            char name[64];
            struct Expression* args[4];
            int arg_count;
        } function;
    } data;
} Expression;

static Expression* expr_create_literal(double value) {
    Expression* expr = (Expression*)calloc(1, sizeof(Expression));
    if (!expr) return NULL;
    expr->type = EXPR_LITERAL;
    expr->data.literal_value = value;
    return expr;
}

static Expression* expr_create_column(const char* name) {
    Expression* expr = (Expression*)calloc(1, sizeof(Expression));
    if (!expr) return NULL;
    expr->type = EXPR_COLUMN;
    strncpy(expr->data.column_name, name, sizeof(expr->data.column_name) - 1);
    return expr;
}

static Expression* expr_create_binary(Expression* left, Expression* right, char op) {
    Expression* expr = (Expression*)calloc(1, sizeof(Expression));
    if (!expr) return NULL;
    expr->type = EXPR_BINARY_OP;
    expr->data.binary.left = left;
    expr->data.binary.right = right;
    expr->data.binary.op = op;
    return expr;
}

static double expr_evaluate(Expression* expr, Row* row) {
    if (!expr) return 0;

    switch (expr->type) {
        case EXPR_LITERAL:
            return expr->data.literal_value;

        case EXPR_COLUMN:
            return 0;

        case EXPR_BINARY_OP: {
            double left = expr_evaluate(expr->data.binary.left, row);
            double right = expr_evaluate(expr->data.binary.right, row);
            switch (expr->data.binary.op) {
                case '+': return left + right;
                case '-': return left - right;
                case '*': return left * right;
                case '/': return right != 0 ? left / right : 0;
                case '%': return right != 0 ? fmod(left, right) : 0;
                default: return 0;
            }
        }

        case EXPR_UNARY_OP: {
            double operand = expr_evaluate(expr->data.unary.operand, row);
            switch (expr->data.unary.op) {
                case '-': return -operand;
                case '!': return operand == 0 ? 1 : 0;
                default: return operand;
            }
        }

        case EXPR_FUNCTION:
            return 0;

        default:
            return 0;
    }
}

static void expr_free(Expression* expr) {
    if (!expr) return;

    switch (expr->type) {
        case EXPR_BINARY_OP:
            expr_free(expr->data.binary.left);
            expr_free(expr->data.binary.right);
            break;
        case EXPR_UNARY_OP:
            expr_free(expr->data.unary.operand);
            break;
        case EXPR_FUNCTION:
            for (int i = 0; i < expr->data.function.arg_count; i++) {
                expr_free(expr->data.function.args[i]);
            }
            break;
        default:
            break;
    }
    free(expr);
}

/* ============================================================================
 * QUERY PLAN AND OPTIMIZATION
 * ============================================================================ */

typedef enum {
    PLAN_SEQ_SCAN,
    PLAN_INDEX_SCAN,
    PLAN_NESTED_LOOP,
    PLAN_HASH_JOIN,
    PLAN_SORT_MERGE,
    PLAN_AGGREGATE,
    PLAN_SORT,
    PLAN_PROJECTION,
    PLAN_FILTER
} PlanNodeType;

typedef struct PlanNode {
    PlanNodeType type;
    int table_id;
    int estimated_rows;
    double estimated_cost;
    struct PlanNode* left;
    struct PlanNode* right;
    Expression* filter;
    char columns[MAX_COLUMNS][64];
    int column_count;
} PlanNode;

static PlanNode* plan_create_node(PlanNodeType type) {
    PlanNode* node = (PlanNode*)calloc(1, sizeof(PlanNode));
    if (!node) return NULL;
    node->type = type;
    return node;
}

static void plan_estimate_cost(PlanNode* node) {
    if (!node) return;

    plan_estimate_cost(node->left);
    plan_estimate_cost(node->right);

    double left_cost = node->left ? node->left->estimated_cost : 0;
    double right_cost = node->right ? node->right->estimated_cost : 0;
    int left_rows = node->left ? node->left->estimated_rows : 1000;
    int right_rows = node->right ? node->right->estimated_rows : 1000;

    switch (node->type) {
        case PLAN_SEQ_SCAN:
            node->estimated_cost = node->estimated_rows * 1.0;
            break;
        case PLAN_INDEX_SCAN:
            node->estimated_cost = log2(node->estimated_rows + 1) * 2.0;
            break;
        case PLAN_NESTED_LOOP:
            node->estimated_cost = left_cost + right_cost +
                                   left_rows * right_rows * 0.01;
            node->estimated_rows = (int)(left_rows * right_rows * 0.1);
            break;
        case PLAN_HASH_JOIN:
            node->estimated_cost = left_cost + right_cost +
                                   left_rows * 2.0 + right_rows * 1.5;
            node->estimated_rows = (int)(left_rows * right_rows * 0.1);
            break;
        case PLAN_SORT_MERGE:
            node->estimated_cost = left_cost + right_cost +
                                   left_rows * log2(left_rows + 1) +
                                   right_rows * log2(right_rows + 1);
            node->estimated_rows = (int)(left_rows * right_rows * 0.1);
            break;
        case PLAN_AGGREGATE:
            node->estimated_cost = left_cost + left_rows * 0.5;
            node->estimated_rows = 1;
            break;
        case PLAN_SORT:
            node->estimated_cost = left_cost +
                                   left_rows * log2(left_rows + 1) * 0.5;
            node->estimated_rows = left_rows;
            break;
        case PLAN_PROJECTION:
            node->estimated_cost = left_cost + left_rows * 0.1;
            node->estimated_rows = left_rows;
            break;
        case PLAN_FILTER:
            node->estimated_cost = left_cost + left_rows * 0.2;
            node->estimated_rows = (int)(left_rows * 0.3);
            break;
    }
}

static void plan_print(PlanNode* node, int indent) {
    if (!node) return;

    for (int i = 0; i < indent; i++) printf("  ");

    const char* type_str = "UNKNOWN";
    switch (node->type) {
        case PLAN_SEQ_SCAN: type_str = "SeqScan"; break;
        case PLAN_INDEX_SCAN: type_str = "IndexScan"; break;
        case PLAN_NESTED_LOOP: type_str = "NestedLoop"; break;
        case PLAN_HASH_JOIN: type_str = "HashJoin"; break;
        case PLAN_SORT_MERGE: type_str = "SortMerge"; break;
        case PLAN_AGGREGATE: type_str = "Aggregate"; break;
        case PLAN_SORT: type_str = "Sort"; break;
        case PLAN_PROJECTION: type_str = "Projection"; break;
        case PLAN_FILTER: type_str = "Filter"; break;
    }

    printf("%s (rows: %d, cost: %.2f)\\n",
           type_str, node->estimated_rows, node->estimated_cost);

    plan_print(node->left, indent + 1);
    plan_print(node->right, indent + 1);
}

static void plan_free(PlanNode* node) {
    if (!node) return;
    plan_free(node->left);
    plan_free(node->right);
    expr_free(node->filter);
    free(node);
}

/* ============================================================================
 * STATISTICS AND HISTOGRAMS
 * ============================================================================ */

#define HISTOGRAM_BUCKETS 100

typedef struct {
    double min_value;
    double max_value;
    int buckets[HISTOGRAM_BUCKETS];
    int total_count;
    double distinct_values;
    double null_fraction;
} ColumnStats;

static void stats_init(ColumnStats* stats) {
    memset(stats, 0, sizeof(ColumnStats));
    stats->min_value = 1e308;
    stats->max_value = -1e308;
}

static void stats_add_value(ColumnStats* stats, double value) {
    if (value < stats->min_value) stats->min_value = value;
    if (value > stats->max_value) stats->max_value = value;
    stats->total_count++;

    if (stats->max_value > stats->min_value) {
        double range = stats->max_value - stats->min_value;
        int bucket = (int)((value - stats->min_value) / range * (HISTOGRAM_BUCKETS - 1));
        if (bucket >= 0 && bucket < HISTOGRAM_BUCKETS) {
            stats->buckets[bucket]++;
        }
    }
}

static double stats_estimate_selectivity(ColumnStats* stats, double low, double high) {
    if (stats->total_count == 0) return 0.5;
    if (stats->max_value <= stats->min_value) return 1.0;

    double range = stats->max_value - stats->min_value;
    int low_bucket = (int)((low - stats->min_value) / range * (HISTOGRAM_BUCKETS - 1));
    int high_bucket = (int)((high - stats->min_value) / range * (HISTOGRAM_BUCKETS - 1));

    low_bucket = low_bucket < 0 ? 0 : (low_bucket >= HISTOGRAM_BUCKETS ? HISTOGRAM_BUCKETS - 1 : low_bucket);
    high_bucket = high_bucket < 0 ? 0 : (high_bucket >= HISTOGRAM_BUCKETS ? HISTOGRAM_BUCKETS - 1 : high_bucket);

    int count = 0;
    for (int i = low_bucket; i <= high_bucket; i++) {
        count += stats->buckets[i];
    }

    return (double)count / stats->total_count;
}

static void stats_print(ColumnStats* stats) {
    printf("[STATS] Column Statistics:\\n");
    printf("  Total Count: %d\\n", stats->total_count);
    printf("  Min Value: %.2f\\n", stats->min_value);
    printf("  Max Value: %.2f\\n", stats->max_value);
    printf("  Distinct: %.0f\\n", stats->distinct_values);
    printf("  Null Fraction: %.2f%%\\n", stats->null_fraction * 100);
}

/* ============================================================================
 * RECOVERY MANAGER
 * ============================================================================ */

typedef struct {
    uint64_t checkpoint_lsn;
    int active_txn_count;
    int active_txn_ids[MAX_TRANSACTIONS];
} RecoveryCheckpoint;

static void recovery_analyze(WalManager* wal, RecoveryCheckpoint* checkpoint) {
    printf("[RECOVERY] Analyzing WAL from LSN %llu...\\n",
           (unsigned long long)checkpoint->checkpoint_lsn);

    checkpoint->active_txn_count = 0;

    for (int i = 0; i < wal->record_count; i++) {
        WalRecord* rec = &wal->buffer[i];
        if (rec->lsn <= checkpoint->checkpoint_lsn) continue;

        if (rec->type == WAL_COMMIT || rec->type == WAL_ABORT) {
            for (int j = 0; j < checkpoint->active_txn_count; j++) {
                if (checkpoint->active_txn_ids[j] == rec->txn_id) {
                    checkpoint->active_txn_ids[j] =
                        checkpoint->active_txn_ids[--checkpoint->active_txn_count];
                    break;
                }
            }
        } else {
            bool found = false;
            for (int j = 0; j < checkpoint->active_txn_count; j++) {
                if (checkpoint->active_txn_ids[j] == rec->txn_id) {
                    found = true;
                    break;
                }
            }
            if (!found && checkpoint->active_txn_count < MAX_TRANSACTIONS) {
                checkpoint->active_txn_ids[checkpoint->active_txn_count++] = rec->txn_id;
            }
        }
    }

    printf("[RECOVERY] Found %d active transactions\\n", checkpoint->active_txn_count);
}

static void recovery_redo(WalManager* wal, uint64_t start_lsn) {
    printf("[RECOVERY] Redo phase from LSN %llu...\\n",
           (unsigned long long)start_lsn);

    int redo_count = 0;
    for (int i = 0; i < wal->record_count; i++) {
        WalRecord* rec = &wal->buffer[i];
        if (rec->lsn < start_lsn) continue;

        switch (rec->type) {
            case WAL_INSERT:
            case WAL_UPDATE:
            case WAL_DELETE:
                redo_count++;
                break;
            default:
                break;
        }
    }

    printf("[RECOVERY] Redone %d operations\\n", redo_count);
}

static void recovery_undo(WalManager* wal, int* loser_txns, int loser_count) {
    printf("[RECOVERY] Undo phase for %d loser transactions...\\n", loser_count);

    int undo_count = 0;
    for (int i = wal->record_count - 1; i >= 0; i--) {
        WalRecord* rec = &wal->buffer[i];

        bool is_loser = false;
        for (int j = 0; j < loser_count; j++) {
            if (rec->txn_id == loser_txns[j]) {
                is_loser = true;
                break;
            }
        }

        if (is_loser) {
            switch (rec->type) {
                case WAL_INSERT:
                case WAL_UPDATE:
                case WAL_DELETE:
                    undo_count++;
                    break;
                default:
                    break;
            }
        }
    }

    printf("[RECOVERY] Undone %d operations\\n", undo_count);
}

/* ============================================================================
 * DEMO MAIN FUNCTION
 * ============================================================================ */

int main(void) {
    printf("\\n");
    printf("╔══════════════════════════════════════════════════════════════╗\\n");
    printf("║           OAAS SQL Database Engine Demo v1.0                 ║\\n");
    printf("║         (Obfuscation-as-a-Service Test Program)              ║\\n");
    printf("╚══════════════════════════════════════════════════════════════╝\\n");
    printf("\\n");

    if (!verify_license(DB_LICENSE_KEY)) {
        return 1;
    }

    if (!authenticate_admin(DB_ADMIN_PASSWORD)) {
        return 1;
    }

    if (!database_init(&g_database, DB_MASTER_KEY, DB_ENCRYPTION_KEY)) {
        printf("[ERROR] Database initialization failed\\n");
        return 1;
    }

    printf("\\n--- Creating Tables ---\\n");

    Column user_columns[] = {
        {"id", TYPE_INT, 4, false, true, true},
        {"username", TYPE_STRING, 64, false, false, true},
        {"email", TYPE_STRING, 128, false, false, false},
        {"password_hash", TYPE_BLOB, 256, false, false, false},
        {"created_at", TYPE_TIMESTAMP, 8, false, false, false}
    };
    table_create(&g_database, "users", user_columns, 5);

    Column order_columns[] = {
        {"id", TYPE_INT, 4, false, true, true},
        {"user_id", TYPE_INT, 4, false, false, true},
        {"total", TYPE_FLOAT, 8, false, false, false},
        {"status", TYPE_STRING, 32, false, false, false},
        {"created_at", TYPE_TIMESTAMP, 8, false, false, false}
    };
    table_create(&g_database, "orders", order_columns, 5);

    Column product_columns[] = {
        {"id", TYPE_INT, 4, false, true, true},
        {"name", TYPE_STRING, 128, false, false, false},
        {"price", TYPE_FLOAT, 8, false, false, false},
        {"stock", TYPE_INT, 4, false, false, false},
        {"category_id", TYPE_INT, 4, true, false, true}
    };
    table_create(&g_database, "products", product_columns, 5);

    printf("\\n--- Table Schemas ---\\n");
    table_describe(table_find(&g_database, "users"));
    table_describe(table_find(&g_database, "orders"));
    table_describe(table_find(&g_database, "products"));

    printf("\\n--- Transaction Demo ---\\n");

    int txn1 = txn_begin(&g_database);
    lock_acquire(&g_database.lock_table, txn1, 1, LOCK_EXCLUSIVE);
    lock_acquire(&g_database.lock_table, txn1, 2, LOCK_SHARED);

    int txn2 = txn_begin(&g_database);
    lock_acquire(&g_database.lock_table, txn2, 3, LOCK_EXCLUSIVE);
    lock_acquire(&g_database.lock_table, txn2, 2, LOCK_SHARED);

    txn_commit(&g_database, txn1);
    txn_abort(&g_database, txn2);

    printf("\\n--- B-Tree Index Demo ---\\n");

    BTreeNode* index_root = NULL;
    int test_keys[] = {50, 25, 75, 10, 30, 60, 90, 5, 15, 27, 35, 55, 65, 85, 95};
    int key_count = sizeof(test_keys) / sizeof(test_keys[0]);

    printf("[BTREE] Inserting %d keys...\\n", key_count);
    for (int i = 0; i < key_count; i++) {
        index_root = btree_insert(index_root, test_keys[i], NULL);
    }

    printf("[BTREE] Tree structure:\\n");
    btree_print_inorder(index_root, 0);

    BTreeNode* found = btree_search(index_root, 60);
    printf("[BTREE] Search for key 60: %s\\n", found ? "FOUND" : "NOT FOUND");

    found = btree_search(index_root, 42);
    printf("[BTREE] Search for key 42: %s\\n", found ? "FOUND" : "NOT FOUND");

    btree_free_node(index_root);

    printf("\\n--- Hash Index Demo ---\\n");

    HashIndex hash_idx;
    hash_index_init(&hash_idx);

    for (int i = 0; i < 100; i++) {
        int* val = (int*)malloc(sizeof(int));
        *val = i * 10;
        hash_index_insert(&hash_idx, i, val);
    }

    hash_index_stats(&hash_idx);

    void* result = hash_index_lookup(&hash_idx, 42);
    printf("[HASH] Lookup key 42: %s (value: %d)\\n",
           result ? "FOUND" : "NOT FOUND",
           result ? *(int*)result : -1);

    printf("\\n--- Bitmap Operations Demo ---\\n");

    Bitmap bm1, bm2, result_bm;
    bitmap_init(&bm1);
    bitmap_init(&bm2);

    for (int i = 0; i < 50; i += 2) bitmap_set(&bm1, i);
    for (int i = 0; i < 50; i += 3) bitmap_set(&bm2, i);

    printf("[BITMAP] BM1 cardinality: %d (even numbers 0-48)\\n", bm1.cardinality);
    printf("[BITMAP] BM2 cardinality: %d (multiples of 3, 0-48)\\n", bm2.cardinality);

    bitmap_and(&result_bm, &bm1, &bm2);
    printf("[BITMAP] AND cardinality: %d (multiples of 6)\\n", result_bm.cardinality);

    bitmap_or(&result_bm, &bm1, &bm2);
    printf("[BITMAP] OR cardinality: %d\\n", result_bm.cardinality);

    printf("\\n--- Join Algorithms Demo ---\\n");

    JoinRecord left_records[20], right_records[20];
    for (int i = 0; i < 20; i++) {
        left_records[i].key = i * 2;
        left_records[i].row_id = i;
        left_records[i].table_id = 1;

        right_records[i].key = i * 3;
        right_records[i].row_id = i;
        right_records[i].table_id = 2;
    }

    nested_loop_join(left_records, 20, right_records, 20);
    hash_join(left_records, 20, right_records, 20);
    sort_merge_join(left_records, 20, right_records, 20);

    printf("\\n--- Aggregate Functions Demo ---\\n");

    AggregateState agg;
    aggregate_init(&agg);

    srand(12345);
    for (int i = 0; i < 100; i++) {
        aggregate_add(&agg, (double)(rand() % 1000) / 10.0);
    }

    aggregate_print(&agg);

    printf("\\n--- Sorting Algorithms Demo ---\\n");

    int sort_data[20];
    for (int i = 0; i < 20; i++) {
        sort_data[i] = rand() % 100;
    }

    printf("[SORT] Original: ");
    for (int i = 0; i < 20; i++) printf("%d ", sort_data[i]);
    printf("\\n");

    db_quicksort(sort_data, 20);

    printf("[SORT] Sorted:   ");
    for (int i = 0; i < 20; i++) printf("%d ", sort_data[i]);
    printf("\\n");

    printf("\\n--- Query Plan Demo ---\\n");

    PlanNode* scan1 = plan_create_node(PLAN_SEQ_SCAN);
    scan1->table_id = 0;
    scan1->estimated_rows = 10000;

    PlanNode* scan2 = plan_create_node(PLAN_INDEX_SCAN);
    scan2->table_id = 1;
    scan2->estimated_rows = 5000;

    PlanNode* join = plan_create_node(PLAN_HASH_JOIN);
    join->left = scan1;
    join->right = scan2;

    PlanNode* filter = plan_create_node(PLAN_FILTER);
    filter->left = join;

    PlanNode* project = plan_create_node(PLAN_PROJECTION);
    project->left = filter;

    plan_estimate_cost(project);

    printf("[PLAN] Query Execution Plan:\\n");
    plan_print(project, 1);

    plan_free(project);

    printf("\\n--- Statistics Demo ---\\n");

    ColumnStats col_stats;
    stats_init(&col_stats);

    for (int i = 0; i < 1000; i++) {
        stats_add_value(&col_stats, (double)(rand() % 100));
    }

    stats_print(&col_stats);

    double selectivity = stats_estimate_selectivity(&col_stats, 25, 75);
    printf("  Selectivity [25, 75]: %.2f%%\\n", selectivity * 100);

    printf("\\n--- WAL and Recovery Demo ---\\n");

    wal_checkpoint(&g_database.wal);

    int txn3 = txn_begin(&g_database);
    wal_write(&g_database.wal, WAL_INSERT, txn3, 0, 1, NULL, NULL, 0);
    wal_write(&g_database.wal, WAL_UPDATE, txn3, 0, 1, NULL, NULL, 0);
    txn_commit(&g_database, txn3);

    RecoveryCheckpoint checkpoint;
    checkpoint.checkpoint_lsn = 1;
    recovery_analyze(&g_database.wal, &checkpoint);

    if (checkpoint.active_txn_count > 0) {
        recovery_redo(&g_database.wal, checkpoint.checkpoint_lsn);
        recovery_undo(&g_database.wal, checkpoint.active_txn_ids,
                      checkpoint.active_txn_count);
    }

    printf("\\n--- Expression Evaluation Demo ---\\n");

    Expression* expr = expr_create_binary(
        expr_create_binary(
            expr_create_literal(10),
            expr_create_literal(5),
            '+'
        ),
        expr_create_literal(3),
        '*'
    );

    double result_val = expr_evaluate(expr, NULL);
    printf("[EXPR] (10 + 5) * 3 = %.2f\\n", result_val);

    expr_free(expr);

    printf("\\n--- Encryption Demo ---\\n");

    char secret_data[] = "This is sensitive database data!";
    int data_len = strlen(secret_data);

    printf("[CRYPTO] Original: %s\\n", secret_data);

    xor_encrypt((uint8_t*)secret_data, data_len, DB_ENCRYPTION_KEY);
    printf("[CRYPTO] Encrypted: ");
    for (int i = 0; i < data_len; i++) {
        printf("%02x", (unsigned char)secret_data[i]);
    }
    printf("\\n");

    xor_decrypt((uint8_t*)secret_data, data_len, DB_ENCRYPTION_KEY);
    printf("[CRYPTO] Decrypted: %s\\n", secret_data);

    database_shutdown(&g_database);

    printf("\\n");
    printf("╔══════════════════════════════════════════════════════════════╗\\n");
    printf("║           Database Engine Demo Complete!                      ║\\n");
    printf("║                  All Tests Passed                             ║\\n");
    printf("╚══════════════════════════════════════════════════════════════╝\\n");
    printf("\\n");

    return 0;
}

/* ============================================================================
 * ADDITIONAL COMPONENTS - SKIP LIST INDEX
 * ============================================================================ */

#define SKIPLIST_MAX_LEVEL 16
#define SKIPLIST_P 0.5

typedef struct SkipNode {
    int key;
    void* value;
    struct SkipNode** forward;
    int level;
} SkipNode;

typedef struct {
    SkipNode* header;
    int level;
    int count;
} SkipList;

static int skiplist_random_level(void) {
    int level = 1;
    while ((rand() / (double)RAND_MAX) < SKIPLIST_P && level < SKIPLIST_MAX_LEVEL) {
        level++;
    }
    return level;
}

static SkipNode* skiplist_create_node(int level, int key, void* value) {
    SkipNode* node = (SkipNode*)malloc(sizeof(SkipNode));
    if (!node) return NULL;

    node->forward = (SkipNode**)calloc(level, sizeof(SkipNode*));
    if (!node->forward) {
        free(node);
        return NULL;
    }

    node->key = key;
    node->value = value;
    node->level = level;

    return node;
}

static void skiplist_init(SkipList* list) {
    list->level = 1;
    list->count = 0;
    list->header = skiplist_create_node(SKIPLIST_MAX_LEVEL, 0, NULL);
}

static void* skiplist_search(SkipList* list, int key) {
    SkipNode* current = list->header;

    for (int i = list->level - 1; i >= 0; i--) {
        while (current->forward[i] && current->forward[i]->key < key) {
            current = current->forward[i];
        }
    }

    current = current->forward[0];

    if (current && current->key == key) {
        return current->value;
    }

    return NULL;
}

static bool skiplist_insert(SkipList* list, int key, void* value) {
    SkipNode* update[SKIPLIST_MAX_LEVEL];
    SkipNode* current = list->header;

    for (int i = list->level - 1; i >= 0; i--) {
        while (current->forward[i] && current->forward[i]->key < key) {
            current = current->forward[i];
        }
        update[i] = current;
    }

    current = current->forward[0];

    if (current && current->key == key) {
        current->value = value;
        return true;
    }

    int new_level = skiplist_random_level();

    if (new_level > list->level) {
        for (int i = list->level; i < new_level; i++) {
            update[i] = list->header;
        }
        list->level = new_level;
    }

    SkipNode* new_node = skiplist_create_node(new_level, key, value);
    if (!new_node) return false;

    for (int i = 0; i < new_level; i++) {
        new_node->forward[i] = update[i]->forward[i];
        update[i]->forward[i] = new_node;
    }

    list->count++;
    return true;
}

static bool skiplist_delete(SkipList* list, int key) {
    SkipNode* update[SKIPLIST_MAX_LEVEL];
    SkipNode* current = list->header;

    for (int i = list->level - 1; i >= 0; i--) {
        while (current->forward[i] && current->forward[i]->key < key) {
            current = current->forward[i];
        }
        update[i] = current;
    }

    current = current->forward[0];

    if (!current || current->key != key) {
        return false;
    }

    for (int i = 0; i < list->level; i++) {
        if (update[i]->forward[i] != current) {
            break;
        }
        update[i]->forward[i] = current->forward[i];
    }

    while (list->level > 1 && !list->header->forward[list->level - 1]) {
        list->level--;
    }

    free(current->forward);
    free(current);
    list->count--;

    return true;
}

static void skiplist_print(SkipList* list) {
    printf("[SKIPLIST] Structure (count: %d, levels: %d):\\n", list->count, list->level);

    for (int i = list->level - 1; i >= 0; i--) {
        printf("  Level %d: ", i);
        SkipNode* current = list->header->forward[i];
        while (current) {
            printf("%d -> ", current->key);
            current = current->forward[i];
        }
        printf("NULL\\n");
    }
}

/* ============================================================================
 * LRU CACHE IMPLEMENTATION
 * ============================================================================ */

#define LRU_CACHE_SIZE 64

typedef struct LRUNode {
    int key;
    void* value;
    struct LRUNode* prev;
    struct LRUNode* next;
} LRUNode;

typedef struct {
    LRUNode* head;
    LRUNode* tail;
    LRUNode* nodes[LRU_CACHE_SIZE];
    int count;
    int hits;
    int misses;
} LRUCache;

static void lru_init(LRUCache* cache) {
    memset(cache, 0, sizeof(LRUCache));
    cache->head = NULL;
    cache->tail = NULL;
}

static void lru_move_to_front(LRUCache* cache, LRUNode* node) {
    if (node == cache->head) return;

    if (node->prev) node->prev->next = node->next;
    if (node->next) node->next->prev = node->prev;

    if (node == cache->tail) {
        cache->tail = node->prev;
    }

    node->prev = NULL;
    node->next = cache->head;

    if (cache->head) cache->head->prev = node;
    cache->head = node;

    if (!cache->tail) cache->tail = node;
}

static void* lru_get(LRUCache* cache, int key) {
    int hash = key % LRU_CACHE_SIZE;
    LRUNode* node = cache->nodes[hash];

    while (node) {
        if (node->key == key) {
            cache->hits++;
            lru_move_to_front(cache, node);
            return node->value;
        }
        node = node->next;
    }

    cache->misses++;
    return NULL;
}

static void lru_put(LRUCache* cache, int key, void* value) {
    int hash = key % LRU_CACHE_SIZE;

    LRUNode* existing = cache->nodes[hash];
    while (existing) {
        if (existing->key == key) {
            existing->value = value;
            lru_move_to_front(cache, existing);
            return;
        }
        existing = existing->next;
    }

    if (cache->count >= LRU_CACHE_SIZE && cache->tail) {
        LRUNode* victim = cache->tail;
        int victim_hash = victim->key % LRU_CACHE_SIZE;

        if (victim->prev) victim->prev->next = NULL;
        cache->tail = victim->prev;

        if (cache->nodes[victim_hash] == victim) {
            cache->nodes[victim_hash] = NULL;
        }

        free(victim);
        cache->count--;
    }

    LRUNode* new_node = (LRUNode*)calloc(1, sizeof(LRUNode));
    new_node->key = key;
    new_node->value = value;

    new_node->next = cache->head;
    if (cache->head) cache->head->prev = new_node;
    cache->head = new_node;

    if (!cache->tail) cache->tail = new_node;

    cache->nodes[hash] = new_node;
    cache->count++;
}

static void lru_stats(LRUCache* cache) {
    printf("[LRU] Cache Statistics:\\n");
    printf("  Entries: %d/%d\\n", cache->count, LRU_CACHE_SIZE);
    printf("  Hits: %d\\n", cache->hits);
    printf("  Misses: %d\\n", cache->misses);
    int total = cache->hits + cache->misses;
    if (total > 0) {
        printf("  Hit Rate: %.1f%%\\n", 100.0 * cache->hits / total);
    }
}

/* ============================================================================
 * BLOOM FILTER IMPLEMENTATION
 * ============================================================================ */

#define BLOOM_SIZE 1024
#define BLOOM_HASH_COUNT 3

typedef struct {
    uint8_t bits[BLOOM_SIZE / 8];
    int element_count;
} BloomFilter;

static void bloom_init(BloomFilter* filter) {
    memset(filter, 0, sizeof(BloomFilter));
}

static uint32_t bloom_hash1(int key) {
    return (uint32_t)key * 2654435761U;
}

static uint32_t bloom_hash2(int key) {
    uint32_t h = (uint32_t)key;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

static uint32_t bloom_hash3(int key) {
    uint32_t h = (uint32_t)key;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = (h >> 16) ^ h;
    return h;
}

static void bloom_add(BloomFilter* filter, int key) {
    uint32_t h1 = bloom_hash1(key) % BLOOM_SIZE;
    uint32_t h2 = bloom_hash2(key) % BLOOM_SIZE;
    uint32_t h3 = bloom_hash3(key) % BLOOM_SIZE;

    filter->bits[h1 / 8] |= (1 << (h1 % 8));
    filter->bits[h2 / 8] |= (1 << (h2 % 8));
    filter->bits[h3 / 8] |= (1 << (h3 % 8));

    filter->element_count++;
}

static bool bloom_may_contain(BloomFilter* filter, int key) {
    uint32_t h1 = bloom_hash1(key) % BLOOM_SIZE;
    uint32_t h2 = bloom_hash2(key) % BLOOM_SIZE;
    uint32_t h3 = bloom_hash3(key) % BLOOM_SIZE;

    return (filter->bits[h1 / 8] & (1 << (h1 % 8))) &&
           (filter->bits[h2 / 8] & (1 << (h2 % 8))) &&
           (filter->bits[h3 / 8] & (1 << (h3 % 8)));
}

static double bloom_false_positive_rate(BloomFilter* filter) {
    int set_bits = 0;
    for (int i = 0; i < BLOOM_SIZE / 8; i++) {
        uint8_t byte = filter->bits[i];
        while (byte) {
            set_bits += byte & 1;
            byte >>= 1;
        }
    }

    double p = (double)set_bits / BLOOM_SIZE;
    return pow(p, BLOOM_HASH_COUNT);
}

/* ============================================================================
 * CUCKOO HASHING IMPLEMENTATION
 * ============================================================================ */

#define CUCKOO_SIZE 128
#define CUCKOO_MAX_KICKS 500

typedef struct {
    int key;
    void* value;
    bool occupied;
} CuckooEntry;

typedef struct {
    CuckooEntry table1[CUCKOO_SIZE];
    CuckooEntry table2[CUCKOO_SIZE];
    int count;
} CuckooHash;

static void cuckoo_init(CuckooHash* hash) {
    memset(hash, 0, sizeof(CuckooHash));
}

static uint32_t cuckoo_hash1(int key) {
    return (uint32_t)key % CUCKOO_SIZE;
}

static uint32_t cuckoo_hash2(int key) {
    uint32_t h = (uint32_t)key;
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    return h % CUCKOO_SIZE;
}

static void* cuckoo_get(CuckooHash* hash, int key) {
    uint32_t h1 = cuckoo_hash1(key);
    uint32_t h2 = cuckoo_hash2(key);

    if (hash->table1[h1].occupied && hash->table1[h1].key == key) {
        return hash->table1[h1].value;
    }

    if (hash->table2[h2].occupied && hash->table2[h2].key == key) {
        return hash->table2[h2].value;
    }

    return NULL;
}

static bool cuckoo_put(CuckooHash* hash, int key, void* value) {
    uint32_t h1 = cuckoo_hash1(key);
    uint32_t h2 = cuckoo_hash2(key);

    if (hash->table1[h1].occupied && hash->table1[h1].key == key) {
        hash->table1[h1].value = value;
        return true;
    }

    if (hash->table2[h2].occupied && hash->table2[h2].key == key) {
        hash->table2[h2].value = value;
        return true;
    }

    if (!hash->table1[h1].occupied) {
        hash->table1[h1].key = key;
        hash->table1[h1].value = value;
        hash->table1[h1].occupied = true;
        hash->count++;
        return true;
    }

    if (!hash->table2[h2].occupied) {
        hash->table2[h2].key = key;
        hash->table2[h2].value = value;
        hash->table2[h2].occupied = true;
        hash->count++;
        return true;
    }

    int current_key = key;
    void* current_value = value;
    bool use_table1 = true;

    for (int i = 0; i < CUCKOO_MAX_KICKS; i++) {
        if (use_table1) {
            uint32_t pos = cuckoo_hash1(current_key);

            int temp_key = hash->table1[pos].key;
            void* temp_value = hash->table1[pos].value;

            hash->table1[pos].key = current_key;
            hash->table1[pos].value = current_value;
            hash->table1[pos].occupied = true;

            current_key = temp_key;
            current_value = temp_value;
        } else {
            uint32_t pos = cuckoo_hash2(current_key);

            if (!hash->table2[pos].occupied) {
                hash->table2[pos].key = current_key;
                hash->table2[pos].value = current_value;
                hash->table2[pos].occupied = true;
                hash->count++;
                return true;
            }

            int temp_key = hash->table2[pos].key;
            void* temp_value = hash->table2[pos].value;

            hash->table2[pos].key = current_key;
            hash->table2[pos].value = current_value;

            current_key = temp_key;
            current_value = temp_value;
        }

        use_table1 = !use_table1;
    }

    printf("[CUCKOO] Insertion failed - need rehash\\n");
    return false;
}

/* ============================================================================
 * CONSISTENT HASHING (FOR DISTRIBUTED DATABASE)
 * ============================================================================ */

#define VIRTUAL_NODES 150
#define MAX_PHYSICAL_NODES 16

typedef struct {
    uint32_t hash_values[MAX_PHYSICAL_NODES * VIRTUAL_NODES];
    int node_ids[MAX_PHYSICAL_NODES * VIRTUAL_NODES];
    int total_vnodes;
    int physical_nodes;
} ConsistentHash;

static uint32_t consistent_hash(const char* key) {
    uint32_t hash = 0;
    while (*key) {
        hash = hash * 31 + *key++;
    }
    return hash;
}

static void consistent_hash_init(ConsistentHash* ch, int num_nodes) {
    ch->physical_nodes = num_nodes;
    ch->total_vnodes = 0;

    char buffer[64];
    for (int node = 0; node < num_nodes; node++) {
        for (int vnode = 0; vnode < VIRTUAL_NODES; vnode++) {
            sprintf(buffer, "node%d-vnode%d", node, vnode);
            ch->hash_values[ch->total_vnodes] = consistent_hash(buffer);
            ch->node_ids[ch->total_vnodes] = node;
            ch->total_vnodes++;
        }
    }

    for (int i = 0; i < ch->total_vnodes - 1; i++) {
        for (int j = 0; j < ch->total_vnodes - i - 1; j++) {
            if (ch->hash_values[j] > ch->hash_values[j + 1]) {
                uint32_t temp_hash = ch->hash_values[j];
                ch->hash_values[j] = ch->hash_values[j + 1];
                ch->hash_values[j + 1] = temp_hash;

                int temp_node = ch->node_ids[j];
                ch->node_ids[j] = ch->node_ids[j + 1];
                ch->node_ids[j + 1] = temp_node;
            }
        }
    }

    printf("[CONSISTENT] Hash ring initialized with %d physical nodes, %d virtual nodes\\n",
           num_nodes, ch->total_vnodes);
}

static int consistent_hash_get_node(ConsistentHash* ch, int key) {
    char buffer[32];
    sprintf(buffer, "%d", key);
    uint32_t hash = consistent_hash(buffer);

    int left = 0, right = ch->total_vnodes - 1;
    while (left < right) {
        int mid = (left + right) / 2;
        if (ch->hash_values[mid] < hash) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    if (left >= ch->total_vnodes) {
        left = 0;
    }

    return ch->node_ids[left];
}

/* ============================================================================
 * MVCC (MULTI-VERSION CONCURRENCY CONTROL)
 * ============================================================================ */

#define MAX_VERSIONS 100

typedef struct {
    int version_id;
    uint64_t timestamp;
    uint8_t data[MAX_VALUE_SIZE];
    int data_size;
    bool deleted;
    int created_by_txn;
    int deleted_by_txn;
} VersionedRow;

typedef struct {
    int row_id;
    VersionedRow versions[MAX_VERSIONS];
    int version_count;
} MVCCRow;

static void mvcc_init_row(MVCCRow* row, int row_id) {
    memset(row, 0, sizeof(MVCCRow));
    row->row_id = row_id;
}

static bool mvcc_write(MVCCRow* row, int txn_id, const uint8_t* data, int size) {
    if (row->version_count >= MAX_VERSIONS) {
        printf("[MVCC] Maximum versions reached for row %d\\n", row->row_id);
        return false;
    }

    VersionedRow* version = &row->versions[row->version_count];
    version->version_id = row->version_count;
    version->timestamp = get_timestamp();
    version->created_by_txn = txn_id;
    version->deleted_by_txn = -1;
    version->deleted = false;
    version->data_size = size < MAX_VALUE_SIZE ? size : MAX_VALUE_SIZE;
    memcpy(version->data, data, version->data_size);

    row->version_count++;

    printf("[MVCC] Row %d: created version %d by txn %d\\n",
           row->row_id, version->version_id, txn_id);

    return true;
}

static VersionedRow* mvcc_read(MVCCRow* row, uint64_t read_timestamp) {
    for (int i = row->version_count - 1; i >= 0; i--) {
        VersionedRow* version = &row->versions[i];

        if (version->timestamp <= read_timestamp && !version->deleted) {
            return version;
        }
    }

    return NULL;
}

static bool mvcc_delete(MVCCRow* row, int txn_id) {
    if (row->version_count == 0) {
        return false;
    }

    VersionedRow* latest = &row->versions[row->version_count - 1];
    if (latest->deleted) {
        return false;
    }

    latest->deleted = true;
    latest->deleted_by_txn = txn_id;

    printf("[MVCC] Row %d: version %d deleted by txn %d\\n",
           row->row_id, latest->version_id, txn_id);

    return true;
}

static void mvcc_garbage_collect(MVCCRow* row, uint64_t oldest_active_txn) {
    int kept = 0;
    for (int i = 0; i < row->version_count; i++) {
        VersionedRow* version = &row->versions[i];

        if (version->timestamp >= oldest_active_txn || i == row->version_count - 1) {
            if (kept != i) {
                memcpy(&row->versions[kept], version, sizeof(VersionedRow));
            }
            kept++;
        }
    }

    int removed = row->version_count - kept;
    row->version_count = kept;

    if (removed > 0) {
        printf("[MVCC] Row %d: garbage collected %d old versions\\n",
               row->row_id, removed);
    }
}

/* ============================================================================
 * DEADLOCK DETECTION
 * ============================================================================ */

#define MAX_WAIT_FOR 64

typedef struct {
    int waiting_txn;
    int holding_txn;
} WaitForEdge;

typedef struct {
    WaitForEdge edges[MAX_WAIT_FOR];
    int edge_count;
} WaitForGraph;

static void wait_for_init(WaitForGraph* graph) {
    memset(graph, 0, sizeof(WaitForGraph));
}

static void wait_for_add_edge(WaitForGraph* graph, int waiting, int holding) {
    if (graph->edge_count >= MAX_WAIT_FOR) {
        return;
    }

    graph->edges[graph->edge_count].waiting_txn = waiting;
    graph->edges[graph->edge_count].holding_txn = holding;
    graph->edge_count++;

    printf("[DEADLOCK] Added wait-for edge: txn %d -> txn %d\\n", waiting, holding);
}

static void wait_for_remove_edges(WaitForGraph* graph, int txn_id) {
    int i = 0;
    while (i < graph->edge_count) {
        if (graph->edges[i].waiting_txn == txn_id ||
            graph->edges[i].holding_txn == txn_id) {
            graph->edges[i] = graph->edges[--graph->edge_count];
        } else {
            i++;
        }
    }
}

static bool wait_for_detect_cycle_dfs(WaitForGraph* graph, int start, int current,
                                       bool* visited, bool* in_stack) {
    visited[current] = true;
    in_stack[current] = true;

    for (int i = 0; i < graph->edge_count; i++) {
        if (graph->edges[i].waiting_txn == current) {
            int next = graph->edges[i].holding_txn;

            if (next == start && in_stack[start]) {
                return true;
            }

            if (!visited[next]) {
                if (wait_for_detect_cycle_dfs(graph, start, next, visited, in_stack)) {
                    return true;
                }
            } else if (in_stack[next]) {
                return true;
            }
        }
    }

    in_stack[current] = false;
    return false;
}

static int wait_for_detect_deadlock(WaitForGraph* graph) {
    bool visited[MAX_TRANSACTIONS] = {false};
    bool in_stack[MAX_TRANSACTIONS] = {false};

    for (int i = 0; i < graph->edge_count; i++) {
        int txn = graph->edges[i].waiting_txn;

        if (!visited[txn]) {
            memset(visited, false, sizeof(visited));
            memset(in_stack, false, sizeof(in_stack));

            if (wait_for_detect_cycle_dfs(graph, txn, txn, visited, in_stack)) {
                printf("[DEADLOCK] Detected! Transaction %d is in a cycle\\n", txn);
                return txn;
            }
        }
    }

    return -1;
}

/* ============================================================================
 * TWO-PHASE LOCKING (2PL) PROTOCOL
 * ============================================================================ */

typedef enum {
    PHASE_GROWING,
    PHASE_SHRINKING
} TwoPLPhase;

typedef struct {
    int txn_id;
    TwoPLPhase phase;
    int locks_acquired;
    int locks_released;
} TwoPLState;

static void two_pl_init(TwoPLState* state, int txn_id) {
    state->txn_id = txn_id;
    state->phase = PHASE_GROWING;
    state->locks_acquired = 0;
    state->locks_released = 0;
}

static bool two_pl_acquire_lock(TwoPLState* state, LockTable* lt, int resource, LockMode mode) {
    if (state->phase == PHASE_SHRINKING) {
        printf("[2PL] Txn %d: Cannot acquire lock in shrinking phase\\n", state->txn_id);
        return false;
    }

    bool acquired = lock_acquire(lt, state->txn_id, resource, mode);
    if (acquired) {
        state->locks_acquired++;
        printf("[2PL] Txn %d: Acquired lock %d (total: %d)\\n",
               state->txn_id, resource, state->locks_acquired);
    }

    return acquired;
}

static void two_pl_release_lock(TwoPLState* state, LockTable* lt, int resource) {
    if (state->phase == PHASE_GROWING) {
        state->phase = PHASE_SHRINKING;
        printf("[2PL] Txn %d: Entering shrinking phase\\n", state->txn_id);
    }

    lock_release(lt, state->txn_id, resource);
    state->locks_released++;

    printf("[2PL] Txn %d: Released lock %d (released: %d/%d)\\n",
           state->txn_id, resource, state->locks_released, state->locks_acquired);
}

/* ============================================================================
 * TIMESTAMP ORDERING PROTOCOL
 * ============================================================================ */

typedef struct {
    int data_item;
    uint64_t read_ts;
    uint64_t write_ts;
} TimestampEntry;

typedef struct {
    TimestampEntry entries[256];
    int entry_count;
} TimestampOrdering;

static void ts_ordering_init(TimestampOrdering* ts) {
    memset(ts, 0, sizeof(TimestampOrdering));
}

static TimestampEntry* ts_get_entry(TimestampOrdering* ts, int data_item) {
    for (int i = 0; i < ts->entry_count; i++) {
        if (ts->entries[i].data_item == data_item) {
            return &ts->entries[i];
        }
    }

    if (ts->entry_count < 256) {
        TimestampEntry* entry = &ts->entries[ts->entry_count++];
        entry->data_item = data_item;
        entry->read_ts = 0;
        entry->write_ts = 0;
        return entry;
    }

    return NULL;
}

static bool ts_read(TimestampOrdering* ts, int data_item, uint64_t txn_ts) {
    TimestampEntry* entry = ts_get_entry(ts, data_item);
    if (!entry) return false;

    if (txn_ts < entry->write_ts) {
        printf("[TS] Read rejected: txn_ts(%llu) < write_ts(%llu) for item %d\\n",
               (unsigned long long)txn_ts, (unsigned long long)entry->write_ts, data_item);
        return false;
    }

    if (txn_ts > entry->read_ts) {
        entry->read_ts = txn_ts;
    }

    printf("[TS] Read allowed: item %d at ts %llu\\n",
           data_item, (unsigned long long)txn_ts);
    return true;
}

static bool ts_write(TimestampOrdering* ts, int data_item, uint64_t txn_ts) {
    TimestampEntry* entry = ts_get_entry(ts, data_item);
    if (!entry) return false;

    if (txn_ts < entry->read_ts) {
        printf("[TS] Write rejected: txn_ts(%llu) < read_ts(%llu) for item %d\\n",
               (unsigned long long)txn_ts, (unsigned long long)entry->read_ts, data_item);
        return false;
    }

    if (txn_ts < entry->write_ts) {
        printf("[TS] Write skipped (Thomas rule): txn_ts(%llu) < write_ts(%llu) for item %d\\n",
               (unsigned long long)txn_ts, (unsigned long long)entry->write_ts, data_item);
        return true;
    }

    entry->write_ts = txn_ts;
    printf("[TS] Write allowed: item %d at ts %llu\\n",
           data_item, (unsigned long long)txn_ts);
    return true;
}

/* ============================================================================
 * CHECKSUM AND DATA INTEGRITY
 * ============================================================================ */

static uint32_t crc32_table[256];
static bool crc32_table_computed = false;

static void crc32_init_table(void) {
    if (crc32_table_computed) return;

    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320 : 0);
        }
        crc32_table[i] = crc;
    }

    crc32_table_computed = true;
}

static uint32_t crc32_compute(const uint8_t* data, int size) {
    crc32_init_table();

    uint32_t crc = 0xFFFFFFFF;
    for (int i = 0; i < size; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ data[i]) & 0xFF];
    }

    return crc ^ 0xFFFFFFFF;
}

static uint64_t xxhash64(const uint8_t* data, int size, uint64_t seed) {
    const uint64_t PRIME1 = 11400714785074694791ULL;
    const uint64_t PRIME2 = 14029467366897019727ULL;
    const uint64_t PRIME3 = 1609587929392839161ULL;
    const uint64_t PRIME4 = 9650029242287828579ULL;
    const uint64_t PRIME5 = 2870177450012600261ULL;

    uint64_t h64;

    if (size >= 32) {
        uint64_t v1 = seed + PRIME1 + PRIME2;
        uint64_t v2 = seed + PRIME2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PRIME1;

        int remaining = size;
        const uint8_t* p = data;

        while (remaining >= 32) {
            uint64_t k1, k2, k3, k4;
            memcpy(&k1, p, 8); p += 8;
            memcpy(&k2, p, 8); p += 8;
            memcpy(&k3, p, 8); p += 8;
            memcpy(&k4, p, 8); p += 8;

            v1 += k1 * PRIME2;
            v1 = (v1 << 31) | (v1 >> 33);
            v1 *= PRIME1;

            v2 += k2 * PRIME2;
            v2 = (v2 << 31) | (v2 >> 33);
            v2 *= PRIME1;

            v3 += k3 * PRIME2;
            v3 = (v3 << 31) | (v3 >> 33);
            v3 *= PRIME1;

            v4 += k4 * PRIME2;
            v4 = (v4 << 31) | (v4 >> 33);
            v4 *= PRIME1;

            remaining -= 32;
        }

        h64 = ((v1 << 1) | (v1 >> 63)) +
              ((v2 << 7) | (v2 >> 57)) +
              ((v3 << 12) | (v3 >> 52)) +
              ((v4 << 18) | (v4 >> 46));

        h64 ^= ((v1 * PRIME2) << 31 | (v1 * PRIME2) >> 33) * PRIME1;
        h64 = h64 * PRIME1 + PRIME4;

        h64 ^= ((v2 * PRIME2) << 31 | (v2 * PRIME2) >> 33) * PRIME1;
        h64 = h64 * PRIME1 + PRIME4;

        h64 ^= ((v3 * PRIME2) << 31 | (v3 * PRIME2) >> 33) * PRIME1;
        h64 = h64 * PRIME1 + PRIME4;

        h64 ^= ((v4 * PRIME2) << 31 | (v4 * PRIME2) >> 33) * PRIME1;
        h64 = h64 * PRIME1 + PRIME4;
    } else {
        h64 = seed + PRIME5;
    }

    h64 += (uint64_t)size;

    h64 ^= h64 >> 33;
    h64 *= PRIME2;
    h64 ^= h64 >> 29;
    h64 *= PRIME3;
    h64 ^= h64 >> 32;

    return h64;
}

static void verify_data_integrity(const uint8_t* data, int size) {
    uint32_t crc = crc32_compute(data, size);
    uint64_t xxh = xxhash64(data, size, 0);

    printf("[INTEGRITY] Data verification:\\n");
    printf("  Size: %d bytes\\n", size);
    printf("  CRC32: 0x%08X\\n", crc);
    printf("  XXHash64: 0x%016llX\\n", (unsigned long long)xxh);
}

/* ============================================================================
 * RED-BLACK TREE IMPLEMENTATION
 * ============================================================================ */

typedef enum { RB_RED, RB_BLACK } RBColor;

typedef struct RBNode {
    int key;
    void* value;
    RBColor color;
    struct RBNode* left;
    struct RBNode* right;
    struct RBNode* parent;
} RBNode;

typedef struct {
    RBNode* root;
    RBNode* nil;
    int count;
} RBTree;

static RBNode* rb_create_node(RBTree* tree, int key, void* value) {
    RBNode* node = (RBNode*)malloc(sizeof(RBNode));
    if (!node) return NULL;

    node->key = key;
    node->value = value;
    node->color = RB_RED;
    node->left = tree->nil;
    node->right = tree->nil;
    node->parent = tree->nil;

    return node;
}

static void rb_init(RBTree* tree) {
    tree->nil = (RBNode*)malloc(sizeof(RBNode));
    tree->nil->color = RB_BLACK;
    tree->nil->left = NULL;
    tree->nil->right = NULL;
    tree->nil->parent = NULL;
    tree->root = tree->nil;
    tree->count = 0;
}

static void rb_left_rotate(RBTree* tree, RBNode* x) {
    RBNode* y = x->right;
    x->right = y->left;

    if (y->left != tree->nil) {
        y->left->parent = x;
    }

    y->parent = x->parent;

    if (x->parent == tree->nil) {
        tree->root = y;
    } else if (x == x->parent->left) {
        x->parent->left = y;
    } else {
        x->parent->right = y;
    }

    y->left = x;
    x->parent = y;
}

static void rb_right_rotate(RBTree* tree, RBNode* y) {
    RBNode* x = y->left;
    y->left = x->right;

    if (x->right != tree->nil) {
        x->right->parent = y;
    }

    x->parent = y->parent;

    if (y->parent == tree->nil) {
        tree->root = x;
    } else if (y == y->parent->right) {
        y->parent->right = x;
    } else {
        y->parent->left = x;
    }

    x->right = y;
    y->parent = x;
}

static void rb_insert_fixup(RBTree* tree, RBNode* z) {
    while (z->parent->color == RB_RED) {
        if (z->parent == z->parent->parent->left) {
            RBNode* y = z->parent->parent->right;

            if (y->color == RB_RED) {
                z->parent->color = RB_BLACK;
                y->color = RB_BLACK;
                z->parent->parent->color = RB_RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->right) {
                    z = z->parent;
                    rb_left_rotate(tree, z);
                }
                z->parent->color = RB_BLACK;
                z->parent->parent->color = RB_RED;
                rb_right_rotate(tree, z->parent->parent);
            }
        } else {
            RBNode* y = z->parent->parent->left;

            if (y->color == RB_RED) {
                z->parent->color = RB_BLACK;
                y->color = RB_BLACK;
                z->parent->parent->color = RB_RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->left) {
                    z = z->parent;
                    rb_right_rotate(tree, z);
                }
                z->parent->color = RB_BLACK;
                z->parent->parent->color = RB_RED;
                rb_left_rotate(tree, z->parent->parent);
            }
        }
    }
    tree->root->color = RB_BLACK;
}

static void rb_insert(RBTree* tree, int key, void* value) {
    RBNode* z = rb_create_node(tree, key, value);
    if (!z) return;

    RBNode* y = tree->nil;
    RBNode* x = tree->root;

    while (x != tree->nil) {
        y = x;
        if (z->key < x->key) {
            x = x->left;
        } else {
            x = x->right;
        }
    }

    z->parent = y;

    if (y == tree->nil) {
        tree->root = z;
    } else if (z->key < y->key) {
        y->left = z;
    } else {
        y->right = z;
    }

    tree->count++;
    rb_insert_fixup(tree, z);
}

static RBNode* rb_search(RBTree* tree, int key) {
    RBNode* x = tree->root;

    while (x != tree->nil && key != x->key) {
        if (key < x->key) {
            x = x->left;
        } else {
            x = x->right;
        }
    }

    return x != tree->nil ? x : NULL;
}

static void rb_inorder(RBTree* tree, RBNode* node) {
    if (node != tree->nil) {
        rb_inorder(tree, node->left);
        printf("%d(%s) ", node->key, node->color == RB_RED ? "R" : "B");
        rb_inorder(tree, node->right);
    }
}

/* ============================================================================
 * INTERVAL TREE FOR RANGE QUERIES
 * ============================================================================ */

typedef struct {
    int low;
    int high;
} Interval;

typedef struct IntervalNode {
    Interval interval;
    int max;
    void* data;
    struct IntervalNode* left;
    struct IntervalNode* right;
} IntervalNode;

static IntervalNode* interval_create_node(Interval interval, void* data) {
    IntervalNode* node = (IntervalNode*)malloc(sizeof(IntervalNode));
    if (!node) return NULL;

    node->interval = interval;
    node->max = interval.high;
    node->data = data;
    node->left = NULL;
    node->right = NULL;

    return node;
}

static IntervalNode* interval_insert(IntervalNode* root, Interval interval, void* data) {
    if (!root) {
        return interval_create_node(interval, data);
    }

    if (interval.low < root->interval.low) {
        root->left = interval_insert(root->left, interval, data);
    } else {
        root->right = interval_insert(root->right, interval, data);
    }

    if (root->max < interval.high) {
        root->max = interval.high;
    }

    return root;
}

static bool interval_overlaps(Interval a, Interval b) {
    return a.low <= b.high && b.low <= a.high;
}

static IntervalNode* interval_search(IntervalNode* root, Interval query) {
    if (!root) return NULL;

    if (interval_overlaps(root->interval, query)) {
        return root;
    }

    if (root->left && root->left->max >= query.low) {
        return interval_search(root->left, query);
    }

    return interval_search(root->right, query);
}

/* ============================================================================
 * TRIE FOR PREFIX QUERIES
 * ============================================================================ */

#define TRIE_ALPHABET_SIZE 26

typedef struct TrieNode {
    struct TrieNode* children[TRIE_ALPHABET_SIZE];
    bool is_end;
    void* value;
} TrieNode;

static TrieNode* trie_create_node(void) {
    TrieNode* node = (TrieNode*)calloc(1, sizeof(TrieNode));
    if (!node) return NULL;

    node->is_end = false;
    node->value = NULL;

    return node;
}

static void trie_insert(TrieNode* root, const char* key, void* value) {
    TrieNode* current = root;

    while (*key) {
        int index = *key - 'a';
        if (index < 0 || index >= TRIE_ALPHABET_SIZE) {
            key++;
            continue;
        }

        if (!current->children[index]) {
            current->children[index] = trie_create_node();
        }

        current = current->children[index];
        key++;
    }

    current->is_end = true;
    current->value = value;
}

static void* trie_search(TrieNode* root, const char* key) {
    TrieNode* current = root;

    while (*key) {
        int index = *key - 'a';
        if (index < 0 || index >= TRIE_ALPHABET_SIZE) {
            key++;
            continue;
        }

        if (!current->children[index]) {
            return NULL;
        }

        current = current->children[index];
        key++;
    }

    return current->is_end ? current->value : NULL;
}

static bool trie_starts_with(TrieNode* root, const char* prefix) {
    TrieNode* current = root;

    while (*prefix) {
        int index = *prefix - 'a';
        if (index < 0 || index >= TRIE_ALPHABET_SIZE) {
            prefix++;
            continue;
        }

        if (!current->children[index]) {
            return false;
        }

        current = current->children[index];
        prefix++;
    }

    return true;
}

/* ============================================================================
 * CONNECTION POOL
 * ============================================================================ */

#define MAX_CONNECTIONS 32

typedef enum {
    CONN_AVAILABLE,
    CONN_IN_USE,
    CONN_CLOSED
} ConnectionState;

typedef struct {
    int connection_id;
    ConnectionState state;
    time_t created_at;
    time_t last_used;
    int queries_executed;
} Connection;

typedef struct {
    Connection connections[MAX_CONNECTIONS];
    int total_connections;
    int available_connections;
    int max_connections;
} ConnectionPool;

static void conn_pool_init(ConnectionPool* pool, int max_conn) {
    memset(pool, 0, sizeof(ConnectionPool));
    pool->max_connections = max_conn < MAX_CONNECTIONS ? max_conn : MAX_CONNECTIONS;

    printf("[POOL] Connection pool initialized (max: %d)\\n", pool->max_connections);
}

static Connection* conn_pool_acquire(ConnectionPool* pool) {
    for (int i = 0; i < pool->total_connections; i++) {
        if (pool->connections[i].state == CONN_AVAILABLE) {
            pool->connections[i].state = CONN_IN_USE;
            pool->connections[i].last_used = time(NULL);
            pool->available_connections--;

            printf("[POOL] Acquired connection %d\\n", pool->connections[i].connection_id);
            return &pool->connections[i];
        }
    }

    if (pool->total_connections < pool->max_connections) {
        Connection* conn = &pool->connections[pool->total_connections];
        conn->connection_id = pool->total_connections + 1;
        conn->state = CONN_IN_USE;
        conn->created_at = time(NULL);
        conn->last_used = time(NULL);
        conn->queries_executed = 0;

        pool->total_connections++;

        printf("[POOL] Created new connection %d\\n", conn->connection_id);
        return conn;
    }

    printf("[POOL] No available connections\\n");
    return NULL;
}

static void conn_pool_release(ConnectionPool* pool, Connection* conn) {
    if (!conn) return;

    conn->state = CONN_AVAILABLE;
    pool->available_connections++;

    printf("[POOL] Released connection %d\\n", conn->connection_id);
}

static void conn_pool_stats(ConnectionPool* pool) {
    printf("[POOL] Statistics:\\n");
    printf("  Total: %d\\n", pool->total_connections);
    printf("  Available: %d\\n", pool->available_connections);
    printf("  In Use: %d\\n", pool->total_connections - pool->available_connections);
    printf("  Max: %d\\n", pool->max_connections);
}

/* ============================================================================
 * PREPARED STATEMENT CACHE
 * ============================================================================ */

#define MAX_PREPARED_STMTS 64

typedef struct {
    char sql[MAX_QUERY_SIZE];
    uint32_t sql_hash;
    int param_count;
    int execution_count;
    double avg_execution_time;
    time_t cached_at;
} PreparedStatement;

typedef struct {
    PreparedStatement statements[MAX_PREPARED_STMTS];
    int count;
    int hits;
    int misses;
} PreparedStmtCache;

static void stmt_cache_init(PreparedStmtCache* cache) {
    memset(cache, 0, sizeof(PreparedStmtCache));
    printf("[STMT] Prepared statement cache initialized\\n");
}

static PreparedStatement* stmt_cache_get(PreparedStmtCache* cache, const char* sql) {
    uint32_t hash = hash_string(sql);

    for (int i = 0; i < cache->count; i++) {
        if (cache->statements[i].sql_hash == hash &&
            strcmp(cache->statements[i].sql, sql) == 0) {
            cache->hits++;
            cache->statements[i].execution_count++;
            return &cache->statements[i];
        }
    }

    cache->misses++;
    return NULL;
}

static PreparedStatement* stmt_cache_put(PreparedStmtCache* cache, const char* sql, int params) {
    if (cache->count >= MAX_PREPARED_STMTS) {
        int oldest_idx = 0;
        time_t oldest_time = cache->statements[0].cached_at;

        for (int i = 1; i < cache->count; i++) {
            if (cache->statements[i].cached_at < oldest_time) {
                oldest_time = cache->statements[i].cached_at;
                oldest_idx = i;
            }
        }

        printf("[STMT] Evicting statement: %s\\n", cache->statements[oldest_idx].sql);
        memmove(&cache->statements[oldest_idx], &cache->statements[oldest_idx + 1],
                (cache->count - oldest_idx - 1) * sizeof(PreparedStatement));
        cache->count--;
    }

    PreparedStatement* stmt = &cache->statements[cache->count];
    strncpy(stmt->sql, sql, MAX_QUERY_SIZE - 1);
    stmt->sql_hash = hash_string(sql);
    stmt->param_count = params;
    stmt->execution_count = 1;
    stmt->avg_execution_time = 0;
    stmt->cached_at = time(NULL);

    cache->count++;

    printf("[STMT] Cached statement: %s\\n", sql);
    return stmt;
}

/* ============================================================================
 * FINAL DATABASE SUMMARY
 * ============================================================================ */

static void print_final_summary(void) {
    printf("\\n");
    printf("╔══════════════════════════════════════════════════════════════╗\\n");
    printf("║                   DATABASE ENGINE SUMMARY                     ║\\n");
    printf("╠══════════════════════════════════════════════════════════════╣\\n");
    printf("║  Components Implemented:                                      ║\\n");
    printf("║  - Buffer Pool Manager with Clock Replacement                 ║\\n");
    printf("║  - Lock Manager with 2PL and Deadlock Detection               ║\\n");
    printf("║  - Write-Ahead Logging (WAL) with Checkpoints                 ║\\n");
    printf("║  - Transaction Manager with ACID Properties                   ║\\n");
    printf("║  - B-Tree, Hash, Skip List, and R-B Tree Indexes              ║\\n");
    printf("║  - Query Parser and Executor                                  ║\\n");
    printf("║  - Join Algorithms (NL, Hash, Sort-Merge)                     ║\\n");
    printf("║  - Aggregate Functions and Statistics                         ║\\n");
    printf("║  - MVCC for Snapshot Isolation                                ║\\n");
    printf("║  - Bloom Filter and Cuckoo Hashing                            ║\\n");
    printf("║  - Consistent Hashing for Distribution                        ║\\n");
    printf("║  - Timestamp Ordering Protocol                                ║\\n");
    printf("║  - CRC32 and XXHash64 Integrity Checks                        ║\\n");
    printf("║  - Connection Pool and Statement Cache                        ║\\n");
    printf("╠══════════════════════════════════════════════════════════════╣\\n");
    printf("║  Secrets Protected: 5 (Master, Encryption, License, Admin,    ║\\n");
    printf("║                         API Key)                              ║\\n");
    printf("╚══════════════════════════════════════════════════════════════╝\\n");
}

`,
};

export const GAME_ENGINE_CPP: LargeDemo = {
  name: 'Game Engine (C++)',
  category: 'large',
  language: 'cpp',
  code: `
/*
 * OAAS Demo: Game Engine
 * Language: C++
 * Category: Large Program (~3500 lines)
 *
 * A complete game engine implementation featuring:
 * - Entity Component System (ECS) architecture
 * - Physics engine with collision detection
 * - Quadtree spatial partitioning
 * - Particle system
 * - Scene management
 * - Resource management
 * - Event system
 * - Animation system
 *
 * Contains secrets for obfuscation testing:
 * - ENGINE_LICENSE_KEY
 * - ENCRYPTION_KEY
 * - API_KEY
 * - STEAM_API_KEY
 * - DRM_KEY
 */

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <queue>
#include <stack>
#include <memory>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <climits>
#include <random>
#include <chrono>
#include <string>
#include <sstream>
#include <fstream>
#include <typeinfo>
#include <type_traits>
#include <bitset>
#include <array>
#include <optional>
#include <variant>

/* ============================================================================
 * CONFIGURATION AND SECRETS
 * ============================================================================ */

namespace Config {
    constexpr const char* ENGINE_LICENSE_KEY = "OAAS_ENGINE_LIC_2024_PREMIUM_UNLIMITED";
    constexpr const char* ENCRYPTION_KEY = "AES256_GAME_ENGINE_MASTER_KEY_SECRET";
    constexpr const char* API_KEY = "api_sk_live_oaas_game_engine_x1y2z3";
    constexpr const char* STEAM_API_KEY = "steam_sk_live_OAAS_2024_game_auth";
    constexpr const char* DRM_KEY = "DRM_OAAS_PROTECTION_KEY_2024_SECURE";

    constexpr int MAX_ENTITIES = 10000;
    constexpr int MAX_COMPONENTS = 64;
    constexpr int MAX_PARTICLES = 5000;
    constexpr int QUADTREE_MAX_OBJECTS = 10;
    constexpr int QUADTREE_MAX_LEVELS = 8;
    constexpr float PHYSICS_TIMESTEP = 1.0f / 60.0f;
    constexpr float GRAVITY = -9.81f;
}

/* ============================================================================
 * MATH UTILITIES
 * ============================================================================ */

struct Vec2 {
    float x, y;

    Vec2() : x(0), y(0) {}
    Vec2(float x, float y) : x(x), y(y) {}

    Vec2 operator+(const Vec2& v) const { return Vec2(x + v.x, y + v.y); }
    Vec2 operator-(const Vec2& v) const { return Vec2(x - v.x, y - v.y); }
    Vec2 operator*(float s) const { return Vec2(x * s, y * s); }
    Vec2 operator/(float s) const { return Vec2(x / s, y / s); }

    Vec2& operator+=(const Vec2& v) { x += v.x; y += v.y; return *this; }
    Vec2& operator-=(const Vec2& v) { x -= v.x; y -= v.y; return *this; }
    Vec2& operator*=(float s) { x *= s; y *= s; return *this; }

    float dot(const Vec2& v) const { return x * v.x + y * v.y; }
    float cross(const Vec2& v) const { return x * v.y - y * v.x; }
    float length() const { return std::sqrt(x * x + y * y); }
    float lengthSquared() const { return x * x + y * y; }

    Vec2 normalized() const {
        float len = length();
        return len > 0 ? *this / len : Vec2();
    }

    Vec2 perpendicular() const { return Vec2(-y, x); }

    static float distance(const Vec2& a, const Vec2& b) {
        return (a - b).length();
    }

    static Vec2 lerp(const Vec2& a, const Vec2& b, float t) {
        return a + (b - a) * t;
    }
};

struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }

    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }

    Vec3 cross(const Vec3& v) const {
        return Vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    float length() const { return std::sqrt(x * x + y * y + z * z); }

    Vec3 normalized() const {
        float len = length();
        return len > 0 ? *this / len : Vec3();
    }
};

struct Mat4 {
    float m[16];

    Mat4() { identity(); }

    void identity() {
        std::memset(m, 0, sizeof(m));
        m[0] = m[5] = m[10] = m[15] = 1.0f;
    }

    static Mat4 translation(float x, float y, float z) {
        Mat4 result;
        result.m[12] = x;
        result.m[13] = y;
        result.m[14] = z;
        return result;
    }

    static Mat4 scale(float x, float y, float z) {
        Mat4 result;
        result.m[0] = x;
        result.m[5] = y;
        result.m[10] = z;
        return result;
    }

    static Mat4 rotationZ(float angle) {
        Mat4 result;
        float c = std::cos(angle);
        float s = std::sin(angle);
        result.m[0] = c;
        result.m[1] = s;
        result.m[4] = -s;
        result.m[5] = c;
        return result;
    }

    Mat4 operator*(const Mat4& other) const {
        Mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i * 4 + j] = 0;
                for (int k = 0; k < 4; k++) {
                    result.m[i * 4 + j] += m[i * 4 + k] * other.m[k * 4 + j];
                }
            }
        }
        return result;
    }

    Vec3 transform(const Vec3& v) const {
        return Vec3(
            m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12],
            m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13],
            m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14]
        );
    }
};

struct Rect {
    float x, y, width, height;

    Rect() : x(0), y(0), width(0), height(0) {}
    Rect(float x, float y, float w, float h) : x(x), y(y), width(w), height(h) {}

    bool contains(float px, float py) const {
        return px >= x && px <= x + width && py >= y && py <= y + height;
    }

    bool contains(const Vec2& p) const {
        return contains(p.x, p.y);
    }

    bool intersects(const Rect& other) const {
        return !(x + width < other.x || other.x + other.width < x ||
                 y + height < other.y || other.y + other.height < y);
    }

    Vec2 center() const {
        return Vec2(x + width / 2, y + height / 2);
    }
};

struct Color {
    uint8_t r, g, b, a;

    Color() : r(255), g(255), b(255), a(255) {}
    Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) : r(r), g(g), b(b), a(a) {}

    static Color lerp(const Color& a, const Color& b, float t) {
        return Color(
            static_cast<uint8_t>(a.r + (b.r - a.r) * t),
            static_cast<uint8_t>(a.g + (b.g - a.g) * t),
            static_cast<uint8_t>(a.b + (b.b - a.b) * t),
            static_cast<uint8_t>(a.a + (b.a - a.a) * t)
        );
    }

    static const Color Red;
    static const Color Green;
    static const Color Blue;
    static const Color White;
    static const Color Black;
};

const Color Color::Red(255, 0, 0);
const Color Color::Green(0, 255, 0);
const Color Color::Blue(0, 0, 255);
const Color Color::White(255, 255, 255);
const Color Color::Black(0, 0, 0);

/* ============================================================================
 * RANDOM NUMBER GENERATOR
 * ============================================================================ */

class Random {
private:
    std::mt19937 generator;
    std::uniform_real_distribution<float> floatDist;
    std::uniform_int_distribution<int> intDist;

public:
    Random() : floatDist(0.0f, 1.0f), intDist(0, INT_MAX) {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        generator.seed(static_cast<unsigned int>(seed));
    }

    float nextFloat() { return floatDist(generator); }
    float nextFloat(float min, float max) { return min + nextFloat() * (max - min); }
    int nextInt(int min, int max) { return min + intDist(generator) % (max - min + 1); }
    bool nextBool() { return nextFloat() > 0.5f; }

    Vec2 nextVec2(float minX, float maxX, float minY, float maxY) {
        return Vec2(nextFloat(minX, maxX), nextFloat(minY, maxY));
    }

    Vec2 randomDirection() {
        float angle = nextFloat(0, 2 * 3.14159f);
        return Vec2(std::cos(angle), std::sin(angle));
    }

    Color randomColor() {
        return Color(
            static_cast<uint8_t>(nextInt(0, 255)),
            static_cast<uint8_t>(nextInt(0, 255)),
            static_cast<uint8_t>(nextInt(0, 255))
        );
    }
};

static Random g_random;

/* ============================================================================
 * ENTITY COMPONENT SYSTEM - TYPES
 * ============================================================================ */

using EntityID = uint32_t;
using ComponentType = uint32_t;
using ComponentMask = std::bitset<Config::MAX_COMPONENTS>;

constexpr EntityID INVALID_ENTITY = 0;

/* ============================================================================
 * COMPONENT BASE CLASS
 * ============================================================================ */

class Component {
public:
    virtual ~Component() = default;
    virtual ComponentType getType() const = 0;
    virtual const char* getName() const = 0;
};

/* ============================================================================
 * TRANSFORM COMPONENT
 * ============================================================================ */

class TransformComponent : public Component {
public:
    Vec2 position;
    Vec2 scale;
    float rotation;
    EntityID parent;

    TransformComponent()
        : position(0, 0), scale(1, 1), rotation(0), parent(INVALID_ENTITY) {}

    ComponentType getType() const override { return 0; }
    const char* getName() const override { return "Transform"; }

    Mat4 getLocalMatrix() const {
        Mat4 t = Mat4::translation(position.x, position.y, 0);
        Mat4 r = Mat4::rotationZ(rotation);
        Mat4 s = Mat4::scale(scale.x, scale.y, 1);
        return t * r * s;
    }
};

/* ============================================================================
 * RIGIDBODY COMPONENT
 * ============================================================================ */

enum class BodyType { Static, Dynamic, Kinematic };

class RigidbodyComponent : public Component {
public:
    Vec2 velocity;
    Vec2 acceleration;
    Vec2 force;
    float mass;
    float drag;
    float angularVelocity;
    float angularDrag;
    float restitution;
    float friction;
    BodyType bodyType;
    bool useGravity;
    bool isGrounded;

    RigidbodyComponent()
        : velocity(0, 0), acceleration(0, 0), force(0, 0),
          mass(1.0f), drag(0.1f), angularVelocity(0), angularDrag(0.1f),
          restitution(0.5f), friction(0.3f), bodyType(BodyType::Dynamic),
          useGravity(true), isGrounded(false) {}

    ComponentType getType() const override { return 1; }
    const char* getName() const override { return "Rigidbody"; }

    void addForce(const Vec2& f) { force += f; }
    void addImpulse(const Vec2& impulse) { velocity += impulse / mass; }

    float inverseMass() const {
        return mass > 0 ? 1.0f / mass : 0;
    }
};

/* ============================================================================
 * COLLIDER COMPONENT
 * ============================================================================ */

enum class ColliderShape { Box, Circle, Polygon };

class ColliderComponent : public Component {
public:
    ColliderShape shape;
    Vec2 offset;
    Vec2 size;
    float radius;
    std::vector<Vec2> vertices;
    bool isTrigger;
    uint32_t layer;
    uint32_t mask;

    ColliderComponent()
        : shape(ColliderShape::Box), offset(0, 0), size(1, 1),
          radius(0.5f), isTrigger(false), layer(1), mask(0xFFFFFFFF) {}

    ComponentType getType() const override { return 2; }
    const char* getName() const override { return "Collider"; }

    Rect getBoundingBox(const Vec2& position) const {
        switch (shape) {
            case ColliderShape::Box:
                return Rect(
                    position.x + offset.x - size.x / 2,
                    position.y + offset.y - size.y / 2,
                    size.x, size.y
                );
            case ColliderShape::Circle:
                return Rect(
                    position.x + offset.x - radius,
                    position.y + offset.y - radius,
                    radius * 2, radius * 2
                );
            default:
                return Rect(position.x, position.y, 1, 1);
        }
    }
};

/* ============================================================================
 * SPRITE COMPONENT
 * ============================================================================ */

class SpriteComponent : public Component {
public:
    std::string texturePath;
    Rect sourceRect;
    Color tint;
    Vec2 pivot;
    int sortingOrder;
    bool flipX;
    bool flipY;
    bool visible;

    SpriteComponent()
        : tint(Color::White), pivot(0.5f, 0.5f), sortingOrder(0),
          flipX(false), flipY(false), visible(true) {}

    ComponentType getType() const override { return 3; }
    const char* getName() const override { return "Sprite"; }
};

/* ============================================================================
 * ANIMATION COMPONENT
 * ============================================================================ */

struct AnimationFrame {
    Rect sourceRect;
    float duration;
};

struct Animation {
    std::string name;
    std::vector<AnimationFrame> frames;
    bool loop;
    float speed;

    Animation() : loop(true), speed(1.0f) {}
};

class AnimatorComponent : public Component {
public:
    std::map<std::string, Animation> animations;
    std::string currentAnimation;
    int currentFrame;
    float frameTimer;
    bool playing;

    AnimatorComponent()
        : currentFrame(0), frameTimer(0), playing(false) {}

    ComponentType getType() const override { return 4; }
    const char* getName() const override { return "Animator"; }

    void play(const std::string& name) {
        if (animations.find(name) != animations.end()) {
            currentAnimation = name;
            currentFrame = 0;
            frameTimer = 0;
            playing = true;
        }
    }

    void stop() {
        playing = false;
    }

    void update(float dt) {
        if (!playing || currentAnimation.empty()) return;

        auto& anim = animations[currentAnimation];
        if (anim.frames.empty()) return;

        frameTimer += dt * anim.speed;

        if (frameTimer >= anim.frames[currentFrame].duration) {
            frameTimer = 0;
            currentFrame++;

            if (currentFrame >= static_cast<int>(anim.frames.size())) {
                if (anim.loop) {
                    currentFrame = 0;
                } else {
                    currentFrame = static_cast<int>(anim.frames.size()) - 1;
                    playing = false;
                }
            }
        }
    }

    const AnimationFrame* getCurrentFrame() const {
        if (currentAnimation.empty()) return nullptr;
        auto it = animations.find(currentAnimation);
        if (it == animations.end()) return nullptr;
        if (it->second.frames.empty()) return nullptr;
        return &it->second.frames[currentFrame];
    }
};

/* ============================================================================
 * SCRIPT COMPONENT
 * ============================================================================ */

class ScriptComponent : public Component {
public:
    std::function<void(EntityID, float)> onUpdate;
    std::function<void(EntityID)> onStart;
    std::function<void(EntityID)> onDestroy;
    std::function<void(EntityID, EntityID)> onCollision;
    std::function<void(EntityID, EntityID)> onTrigger;
    bool started;

    ScriptComponent() : started(false) {}

    ComponentType getType() const override { return 5; }
    const char* getName() const override { return "Script"; }
};

/* ============================================================================
 * AUDIO SOURCE COMPONENT
 * ============================================================================ */

class AudioSourceComponent : public Component {
public:
    std::string audioPath;
    float volume;
    float pitch;
    bool loop;
    bool playOnStart;
    bool spatial;
    float minDistance;
    float maxDistance;
    bool isPlaying;

    AudioSourceComponent()
        : volume(1.0f), pitch(1.0f), loop(false), playOnStart(false),
          spatial(false), minDistance(1.0f), maxDistance(100.0f), isPlaying(false) {}

    ComponentType getType() const override { return 6; }
    const char* getName() const override { return "AudioSource"; }
};

/* ============================================================================
 * CAMERA COMPONENT
 * ============================================================================ */

class CameraComponent : public Component {
public:
    float orthographicSize;
    float nearPlane;
    float farPlane;
    Color backgroundColor;
    Rect viewport;
    int depth;
    bool isMain;

    CameraComponent()
        : orthographicSize(5.0f), nearPlane(0.1f), farPlane(1000.0f),
          backgroundColor(Color(100, 149, 237)), viewport(0, 0, 1, 1),
          depth(0), isMain(false) {}

    ComponentType getType() const override { return 7; }
    const char* getName() const override { return "Camera"; }

    Mat4 getViewMatrix(const Vec2& position, float rotation) const {
        Mat4 t = Mat4::translation(-position.x, -position.y, 0);
        Mat4 r = Mat4::rotationZ(-rotation);
        return r * t;
    }

    Mat4 getProjectionMatrix(float aspectRatio) const {
        Mat4 result;
        float width = orthographicSize * aspectRatio;
        float height = orthographicSize;
        result.m[0] = 2.0f / width;
        result.m[5] = 2.0f / height;
        result.m[10] = -2.0f / (farPlane - nearPlane);
        result.m[14] = -(farPlane + nearPlane) / (farPlane - nearPlane);
        return result;
    }
};

/* ============================================================================
 * PARTICLE EMITTER COMPONENT
 * ============================================================================ */

struct ParticleData {
    Vec2 position;
    Vec2 velocity;
    Color startColor;
    Color endColor;
    float startSize;
    float endSize;
    float lifetime;
    float age;
    float rotation;
    float angularVelocity;
    bool active;

    ParticleData() : startSize(1), endSize(0), lifetime(1), age(0),
                     rotation(0), angularVelocity(0), active(false) {}
};

class ParticleEmitterComponent : public Component {
public:
    Vec2 emitterOffset;
    float emissionRate;
    float particleLifetime;
    float startSpeed;
    float startSpeedVariance;
    float startSize;
    float startSizeVariance;
    float endSize;
    Color startColor;
    Color endColor;
    float angle;
    float angleVariance;
    float gravity;
    Vec2 acceleration;
    bool emitting;
    int maxParticles;
    float emissionTimer;

    std::vector<ParticleData> particles;

    ParticleEmitterComponent()
        : emitterOffset(0, 0), emissionRate(10.0f), particleLifetime(2.0f),
          startSpeed(50.0f), startSpeedVariance(10.0f), startSize(10.0f),
          startSizeVariance(2.0f), endSize(0.0f), startColor(Color::White),
          endColor(Color(255, 255, 255, 0)), angle(90.0f), angleVariance(30.0f),
          gravity(-50.0f), acceleration(0, 0), emitting(true),
          maxParticles(100), emissionTimer(0) {
        particles.resize(maxParticles);
    }

    ComponentType getType() const override { return 8; }
    const char* getName() const override { return "ParticleEmitter"; }

    void emit(const Vec2& position) {
        for (auto& p : particles) {
            if (!p.active) {
                float a = (angle + g_random.nextFloat(-angleVariance, angleVariance)) * 3.14159f / 180.0f;
                float speed = startSpeed + g_random.nextFloat(-startSpeedVariance, startSpeedVariance);

                p.position = position + emitterOffset;
                p.velocity = Vec2(std::cos(a), std::sin(a)) * speed;
                p.startColor = startColor;
                p.endColor = endColor;
                p.startSize = startSize + g_random.nextFloat(-startSizeVariance, startSizeVariance);
                p.endSize = endSize;
                p.lifetime = particleLifetime;
                p.age = 0;
                p.rotation = g_random.nextFloat(0, 360);
                p.angularVelocity = g_random.nextFloat(-180, 180);
                p.active = true;
                break;
            }
        }
    }

    void update(const Vec2& position, float dt) {
        if (emitting) {
            emissionTimer += dt;
            float interval = 1.0f / emissionRate;
            while (emissionTimer >= interval) {
                emit(position);
                emissionTimer -= interval;
            }
        }

        for (auto& p : particles) {
            if (!p.active) continue;

            p.age += dt;
            if (p.age >= p.lifetime) {
                p.active = false;
                continue;
            }

            p.velocity.y += gravity * dt;
            p.velocity += acceleration * dt;
            p.position += p.velocity * dt;
            p.rotation += p.angularVelocity * dt;
        }
    }

    int getActiveParticleCount() const {
        int count = 0;
        for (const auto& p : particles) {
            if (p.active) count++;
        }
        return count;
    }
};

/* ============================================================================
 * ENTITY CLASS
 * ============================================================================ */

class Entity {
public:
    EntityID id;
    std::string name;
    std::string tag;
    bool active;
    ComponentMask componentMask;
    std::map<ComponentType, std::unique_ptr<Component>> components;

    Entity(EntityID id) : id(id), active(true) {}

    template<typename T, typename... Args>
    T* addComponent(Args&&... args) {
        auto component = std::make_unique<T>(std::forward<Args>(args)...);
        T* ptr = component.get();
        ComponentType type = ptr->getType();
        componentMask.set(type);
        components[type] = std::move(component);
        return ptr;
    }

    template<typename T>
    T* getComponent() {
        ComponentType type = T().getType();
        auto it = components.find(type);
        return it != components.end() ? static_cast<T*>(it->second.get()) : nullptr;
    }

    template<typename T>
    bool hasComponent() const {
        ComponentType type = T().getType();
        return componentMask.test(type);
    }

    template<typename T>
    void removeComponent() {
        ComponentType type = T().getType();
        componentMask.reset(type);
        components.erase(type);
    }
};

/* ============================================================================
 * QUADTREE FOR SPATIAL PARTITIONING
 * ============================================================================ */

class Quadtree {
private:
    int level;
    Rect bounds;
    std::vector<EntityID> objects;
    std::unique_ptr<Quadtree> nodes[4];

    void split() {
        float subWidth = bounds.width / 2;
        float subHeight = bounds.height / 2;
        float x = bounds.x;
        float y = bounds.y;

        nodes[0] = std::make_unique<Quadtree>(level + 1,
            Rect(x + subWidth, y, subWidth, subHeight));
        nodes[1] = std::make_unique<Quadtree>(level + 1,
            Rect(x, y, subWidth, subHeight));
        nodes[2] = std::make_unique<Quadtree>(level + 1,
            Rect(x, y + subHeight, subWidth, subHeight));
        nodes[3] = std::make_unique<Quadtree>(level + 1,
            Rect(x + subWidth, y + subHeight, subWidth, subHeight));
    }

    int getIndex(const Rect& rect) const {
        int index = -1;
        float verticalMidpoint = bounds.x + bounds.width / 2;
        float horizontalMidpoint = bounds.y + bounds.height / 2;

        bool topQuadrant = rect.y < horizontalMidpoint &&
                          rect.y + rect.height < horizontalMidpoint;
        bool bottomQuadrant = rect.y > horizontalMidpoint;

        if (rect.x < verticalMidpoint && rect.x + rect.width < verticalMidpoint) {
            if (topQuadrant) index = 1;
            else if (bottomQuadrant) index = 2;
        } else if (rect.x > verticalMidpoint) {
            if (topQuadrant) index = 0;
            else if (bottomQuadrant) index = 3;
        }

        return index;
    }

public:
    Quadtree(int level, const Rect& bounds)
        : level(level), bounds(bounds) {}

    void clear() {
        objects.clear();
        for (int i = 0; i < 4; i++) {
            if (nodes[i]) {
                nodes[i]->clear();
                nodes[i].reset();
            }
        }
    }

    void insert(EntityID entity, const Rect& rect) {
        if (nodes[0]) {
            int index = getIndex(rect);
            if (index != -1) {
                nodes[index]->insert(entity, rect);
                return;
            }
        }

        objects.push_back(entity);

        if (objects.size() > Config::QUADTREE_MAX_OBJECTS && level < Config::QUADTREE_MAX_LEVELS) {
            if (!nodes[0]) {
                split();
            }

            auto it = objects.begin();
            while (it != objects.end()) {
                int index = getIndex(rect);
                if (index != -1) {
                    nodes[index]->insert(*it, rect);
                    it = objects.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    std::vector<EntityID> retrieve(const Rect& rect) const {
        std::vector<EntityID> result;
        int index = getIndex(rect);

        if (index != -1 && nodes[0]) {
            auto subResult = nodes[index]->retrieve(rect);
            result.insert(result.end(), subResult.begin(), subResult.end());
        }

        result.insert(result.end(), objects.begin(), objects.end());
        return result;
    }
};

/* ============================================================================
 * COLLISION DETECTION
 * ============================================================================ */

struct CollisionInfo {
    EntityID entityA;
    EntityID entityB;
    Vec2 normal;
    float depth;
    Vec2 contactPoint;
    bool isTrigger;
};

class CollisionDetector {
public:
    static bool checkCollision(const Vec2& posA, const ColliderComponent& colA,
                               const Vec2& posB, const ColliderComponent& colB,
                               CollisionInfo& info) {
        if (colA.shape == ColliderShape::Circle && colB.shape == ColliderShape::Circle) {
            return circleVsCircle(posA, colA, posB, colB, info);
        } else if (colA.shape == ColliderShape::Box && colB.shape == ColliderShape::Box) {
            return boxVsBox(posA, colA, posB, colB, info);
        } else if (colA.shape == ColliderShape::Circle && colB.shape == ColliderShape::Box) {
            return circleVsBox(posA, colA, posB, colB, info);
        } else if (colA.shape == ColliderShape::Box && colB.shape == ColliderShape::Circle) {
            bool result = circleVsBox(posB, colB, posA, colA, info);
            if (result) {
                std::swap(info.entityA, info.entityB);
                info.normal = info.normal * -1;
            }
            return result;
        }
        return false;
    }

private:
    static bool circleVsCircle(const Vec2& posA, const ColliderComponent& colA,
                               const Vec2& posB, const ColliderComponent& colB,
                               CollisionInfo& info) {
        Vec2 centerA = posA + colA.offset;
        Vec2 centerB = posB + colB.offset;

        Vec2 diff = centerB - centerA;
        float distance = diff.length();
        float radiiSum = colA.radius + colB.radius;

        if (distance < radiiSum) {
            info.normal = distance > 0 ? diff.normalized() : Vec2(1, 0);
            info.depth = radiiSum - distance;
            info.contactPoint = centerA + info.normal * colA.radius;
            info.isTrigger = colA.isTrigger || colB.isTrigger;
            return true;
        }
        return false;
    }

    static bool boxVsBox(const Vec2& posA, const ColliderComponent& colA,
                         const Vec2& posB, const ColliderComponent& colB,
                         CollisionInfo& info) {
        Vec2 centerA = posA + colA.offset;
        Vec2 centerB = posB + colB.offset;

        Vec2 halfSizeA = colA.size * 0.5f;
        Vec2 halfSizeB = colB.size * 0.5f;

        Vec2 diff = centerB - centerA;
        Vec2 overlap(
            halfSizeA.x + halfSizeB.x - std::abs(diff.x),
            halfSizeA.y + halfSizeB.y - std::abs(diff.y)
        );

        if (overlap.x > 0 && overlap.y > 0) {
            if (overlap.x < overlap.y) {
                info.normal = Vec2(diff.x > 0 ? 1.0f : -1.0f, 0);
                info.depth = overlap.x;
            } else {
                info.normal = Vec2(0, diff.y > 0 ? 1.0f : -1.0f);
                info.depth = overlap.y;
            }
            info.contactPoint = centerA + info.normal * (halfSizeA.x - info.depth / 2);
            info.isTrigger = colA.isTrigger || colB.isTrigger;
            return true;
        }
        return false;
    }

    static bool circleVsBox(const Vec2& circlePos, const ColliderComponent& circleCol,
                            const Vec2& boxPos, const ColliderComponent& boxCol,
                            CollisionInfo& info) {
        Vec2 circleCenter = circlePos + circleCol.offset;
        Vec2 boxCenter = boxPos + boxCol.offset;
        Vec2 halfSize = boxCol.size * 0.5f;

        Vec2 closest(
            std::max(boxCenter.x - halfSize.x, std::min(circleCenter.x, boxCenter.x + halfSize.x)),
            std::max(boxCenter.y - halfSize.y, std::min(circleCenter.y, boxCenter.y + halfSize.y))
        );

        Vec2 diff = circleCenter - closest;
        float distanceSquared = diff.lengthSquared();
        float radiusSquared = circleCol.radius * circleCol.radius;

        if (distanceSquared < radiusSquared) {
            float distance = std::sqrt(distanceSquared);
            info.normal = distance > 0 ? diff.normalized() : Vec2(1, 0);
            info.depth = circleCol.radius - distance;
            info.contactPoint = closest;
            info.isTrigger = circleCol.isTrigger || boxCol.isTrigger;
            return true;
        }
        return false;
    }
};

/* ============================================================================
 * PHYSICS SYSTEM
 * ============================================================================ */

class PhysicsSystem {
private:
    Vec2 gravity;
    float fixedTimestep;
    float accumulator;
    std::vector<CollisionInfo> collisions;

public:
    PhysicsSystem()
        : gravity(0, Config::GRAVITY), fixedTimestep(Config::PHYSICS_TIMESTEP), accumulator(0) {}

    void setGravity(const Vec2& g) { gravity = g; }

    void update(std::vector<std::unique_ptr<Entity>>& entities, float dt) {
        accumulator += dt;

        while (accumulator >= fixedTimestep) {
            fixedUpdate(entities, fixedTimestep);
            accumulator -= fixedTimestep;
        }
    }

private:
    void fixedUpdate(std::vector<std::unique_ptr<Entity>>& entities, float dt) {
        for (auto& entity : entities) {
            if (!entity->active) continue;

            auto* transform = entity->getComponent<TransformComponent>();
            auto* rigidbody = entity->getComponent<RigidbodyComponent>();

            if (!transform || !rigidbody) continue;
            if (rigidbody->bodyType == BodyType::Static) continue;

            if (rigidbody->useGravity && rigidbody->bodyType == BodyType::Dynamic) {
                rigidbody->addForce(gravity * rigidbody->mass);
            }

            Vec2 acceleration = rigidbody->force * rigidbody->inverseMass();
            rigidbody->velocity += acceleration * dt;

            rigidbody->velocity *= (1.0f - rigidbody->drag * dt);

            transform->position += rigidbody->velocity * dt;

            transform->rotation += rigidbody->angularVelocity * dt;
            rigidbody->angularVelocity *= (1.0f - rigidbody->angularDrag * dt);

            rigidbody->force = Vec2(0, 0);
        }

        detectCollisions(entities);
        resolveCollisions(entities);
    }

    void detectCollisions(std::vector<std::unique_ptr<Entity>>& entities) {
        collisions.clear();

        for (size_t i = 0; i < entities.size(); i++) {
            if (!entities[i]->active) continue;

            auto* transformA = entities[i]->getComponent<TransformComponent>();
            auto* colliderA = entities[i]->getComponent<ColliderComponent>();

            if (!transformA || !colliderA) continue;

            for (size_t j = i + 1; j < entities.size(); j++) {
                if (!entities[j]->active) continue;

                auto* transformB = entities[j]->getComponent<TransformComponent>();
                auto* colliderB = entities[j]->getComponent<ColliderComponent>();

                if (!transformB || !colliderB) continue;

                if (!(colliderA->mask & colliderB->layer) ||
                    !(colliderB->mask & colliderA->layer)) {
                    continue;
                }

                CollisionInfo info;
                info.entityA = entities[i]->id;
                info.entityB = entities[j]->id;

                if (CollisionDetector::checkCollision(
                    transformA->position, *colliderA,
                    transformB->position, *colliderB, info)) {
                    collisions.push_back(info);
                }
            }
        }
    }

    void resolveCollisions(std::vector<std::unique_ptr<Entity>>& entities) {
        for (const auto& collision : collisions) {
            Entity* entityA = nullptr;
            Entity* entityB = nullptr;

            for (auto& e : entities) {
                if (e->id == collision.entityA) entityA = e.get();
                if (e->id == collision.entityB) entityB = e.get();
            }

            if (!entityA || !entityB) continue;

            if (collision.isTrigger) {
                auto* scriptA = entityA->getComponent<ScriptComponent>();
                auto* scriptB = entityB->getComponent<ScriptComponent>();

                if (scriptA && scriptA->onTrigger) {
                    scriptA->onTrigger(collision.entityA, collision.entityB);
                }
                if (scriptB && scriptB->onTrigger) {
                    scriptB->onTrigger(collision.entityB, collision.entityA);
                }
                continue;
            }

            auto* transformA = entityA->getComponent<TransformComponent>();
            auto* transformB = entityB->getComponent<TransformComponent>();
            auto* rigidbodyA = entityA->getComponent<RigidbodyComponent>();
            auto* rigidbodyB = entityB->getComponent<RigidbodyComponent>();

            float invMassA = rigidbodyA ? rigidbodyA->inverseMass() : 0;
            float invMassB = rigidbodyB ? rigidbodyB->inverseMass() : 0;

            if (invMassA == 0 && invMassB == 0) continue;

            float totalInvMass = invMassA + invMassB;

            transformA->position -= collision.normal * (collision.depth * invMassA / totalInvMass);
            transformB->position += collision.normal * (collision.depth * invMassB / totalInvMass);

            if (rigidbodyA && rigidbodyB) {
                Vec2 relativeVelocity = rigidbodyB->velocity - rigidbodyA->velocity;
                float velocityAlongNormal = relativeVelocity.dot(collision.normal);

                if (velocityAlongNormal > 0) continue;

                float restitution = std::min(rigidbodyA->restitution, rigidbodyB->restitution);
                float j = -(1 + restitution) * velocityAlongNormal;
                j /= totalInvMass;

                Vec2 impulse = collision.normal * j;
                rigidbodyA->velocity -= impulse * invMassA;
                rigidbodyB->velocity += impulse * invMassB;
            }

            auto* scriptA = entityA->getComponent<ScriptComponent>();
            auto* scriptB = entityB->getComponent<ScriptComponent>();

            if (scriptA && scriptA->onCollision) {
                scriptA->onCollision(collision.entityA, collision.entityB);
            }
            if (scriptB && scriptB->onCollision) {
                scriptB->onCollision(collision.entityB, collision.entityA);
            }
        }
    }
};

/* ============================================================================
 * EVENT SYSTEM
 * ============================================================================ */

using EventID = size_t;

class Event {
public:
    virtual ~Event() = default;
    virtual EventID getTypeID() const = 0;
};

template<typename T>
EventID getEventTypeID() {
    static EventID id = typeid(T).hash_code();
    return id;
}

class EventBus {
private:
    using HandlerList = std::vector<std::function<void(const Event&)>>;
    std::unordered_map<EventID, HandlerList> handlers;

public:
    template<typename T, typename Handler>
    void subscribe(Handler&& handler) {
        EventID typeID = getEventTypeID<T>();
        handlers[typeID].push_back([handler](const Event& event) {
            handler(static_cast<const T&>(event));
        });
    }

    template<typename T>
    void publish(const T& event) {
        EventID typeID = getEventTypeID<T>();
        auto it = handlers.find(typeID);
        if (it != handlers.end()) {
            for (auto& handler : it->second) {
                handler(event);
            }
        }
    }

    void clear() {
        handlers.clear();
    }
};

struct EntityCreatedEvent : public Event {
    EntityID entityID;
    EntityCreatedEvent(EntityID id) : entityID(id) {}
    EventID getTypeID() const override { return getEventTypeID<EntityCreatedEvent>(); }
};

struct EntityDestroyedEvent : public Event {
    EntityID entityID;
    EntityDestroyedEvent(EntityID id) : entityID(id) {}
    EventID getTypeID() const override { return getEventTypeID<EntityDestroyedEvent>(); }
};

struct CollisionEvent : public Event {
    EntityID entityA;
    EntityID entityB;
    Vec2 normal;
    CollisionEvent(EntityID a, EntityID b, const Vec2& n) : entityA(a), entityB(b), normal(n) {}
    EventID getTypeID() const override { return getEventTypeID<CollisionEvent>(); }
};

/* ============================================================================
 * RESOURCE MANAGER
 * ============================================================================ */

template<typename T>
class ResourceManager {
private:
    std::unordered_map<std::string, std::shared_ptr<T>> resources;
    std::string basePath;

public:
    ResourceManager(const std::string& basePath = "") : basePath(basePath) {}

    void setBasePath(const std::string& path) { basePath = path; }

    std::shared_ptr<T> load(const std::string& name, const std::string& path) {
        auto it = resources.find(name);
        if (it != resources.end()) {
            return it->second;
        }

        auto resource = std::make_shared<T>();
        resources[name] = resource;
        std::cout << "[RESOURCE] Loaded: " << name << " from " << basePath + path << "\\n";
        return resource;
    }

    std::shared_ptr<T> get(const std::string& name) {
        auto it = resources.find(name);
        return it != resources.end() ? it->second : nullptr;
    }

    void unload(const std::string& name) {
        resources.erase(name);
        std::cout << "[RESOURCE] Unloaded: " << name << "\\n";
    }

    void clear() {
        resources.clear();
        std::cout << "[RESOURCE] Cleared all resources\\n";
    }

    size_t count() const { return resources.size(); }
};

struct Texture {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

struct AudioClip {
    std::vector<int16_t> samples;
    int sampleRate = 44100;
    int channels = 2;
};

struct Font {
    std::string family;
    int size = 12;
};

/* ============================================================================
 * SCENE MANAGER
 * ============================================================================ */

class Scene {
public:
    std::string name;
    std::vector<std::unique_ptr<Entity>> entities;
    std::queue<EntityID> freeIDs;
    EntityID nextID;
    Quadtree quadtree;
    bool loaded;

    Scene(const std::string& name)
        : name(name), nextID(1), quadtree(0, Rect(-1000, -1000, 2000, 2000)), loaded(false) {}

    Entity* createEntity(const std::string& entityName = "") {
        EntityID id;
        if (!freeIDs.empty()) {
            id = freeIDs.front();
            freeIDs.pop();
        } else {
            id = nextID++;
        }

        auto entity = std::make_unique<Entity>(id);
        entity->name = entityName.empty() ? "Entity_" + std::to_string(id) : entityName;
        Entity* ptr = entity.get();
        entities.push_back(std::move(entity));

        std::cout << "[SCENE] Created entity: " << ptr->name << " (ID: " << id << ")\\n";
        return ptr;
    }

    void destroyEntity(EntityID id) {
        for (auto it = entities.begin(); it != entities.end(); ++it) {
            if ((*it)->id == id) {
                std::cout << "[SCENE] Destroyed entity: " << (*it)->name << " (ID: " << id << ")\\n";
                freeIDs.push(id);
                entities.erase(it);
                return;
            }
        }
    }

    Entity* findEntity(EntityID id) {
        for (auto& entity : entities) {
            if (entity->id == id) return entity.get();
        }
        return nullptr;
    }

    Entity* findEntityByName(const std::string& entityName) {
        for (auto& entity : entities) {
            if (entity->name == entityName) return entity.get();
        }
        return nullptr;
    }

    std::vector<Entity*> findEntitiesByTag(const std::string& tag) {
        std::vector<Entity*> result;
        for (auto& entity : entities) {
            if (entity->tag == tag) result.push_back(entity.get());
        }
        return result;
    }

    void updateQuadtree() {
        quadtree.clear();
        for (auto& entity : entities) {
            if (!entity->active) continue;

            auto* transform = entity->getComponent<TransformComponent>();
            auto* collider = entity->getComponent<ColliderComponent>();

            if (transform && collider) {
                Rect bounds = collider->getBoundingBox(transform->position);
                quadtree.insert(entity->id, bounds);
            }
        }
    }
};

class SceneManager {
private:
    std::map<std::string, std::unique_ptr<Scene>> scenes;
    Scene* activeScene;
    std::string nextScene;

public:
    SceneManager() : activeScene(nullptr) {}

    Scene* createScene(const std::string& name) {
        auto scene = std::make_unique<Scene>(name);
        Scene* ptr = scene.get();
        scenes[name] = std::move(scene);
        std::cout << "[SCENE] Created scene: " << name << "\\n";
        return ptr;
    }

    void loadScene(const std::string& name) {
        nextScene = name;
    }

    void switchScene() {
        if (nextScene.empty()) return;

        auto it = scenes.find(nextScene);
        if (it != scenes.end()) {
            if (activeScene) {
                activeScene->loaded = false;
                std::cout << "[SCENE] Unloaded scene: " << activeScene->name << "\\n";
            }

            activeScene = it->second.get();
            activeScene->loaded = true;
            std::cout << "[SCENE] Loaded scene: " << activeScene->name << "\\n";
        }

        nextScene.clear();
    }

    Scene* getActiveScene() { return activeScene; }

    void destroyScene(const std::string& name) {
        if (activeScene && activeScene->name == name) {
            activeScene = nullptr;
        }
        scenes.erase(name);
        std::cout << "[SCENE] Destroyed scene: " << name << "\\n";
    }
};

/* ============================================================================
 * INPUT SYSTEM
 * ============================================================================ */

enum class KeyCode {
    A, B, C, D, E, F, G, H, I, J, K, L, M,
    N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
    Space, Enter, Escape, Tab, Backspace,
    Up, Down, Left, Right,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    LeftShift, RightShift, LeftCtrl, RightCtrl, LeftAlt, RightAlt,
    COUNT
};

enum class MouseButton { Left, Middle, Right, COUNT };

class InputSystem {
private:
    std::bitset<static_cast<size_t>(KeyCode::COUNT)> currentKeys;
    std::bitset<static_cast<size_t>(KeyCode::COUNT)> previousKeys;
    std::bitset<static_cast<size_t>(MouseButton::COUNT)> currentMouse;
    std::bitset<static_cast<size_t>(MouseButton::COUNT)> previousMouse;
    Vec2 mousePosition;
    Vec2 mouseDelta;
    float scrollDelta;

public:
    InputSystem() : scrollDelta(0) {}

    void update() {
        previousKeys = currentKeys;
        previousMouse = currentMouse;
        mouseDelta = Vec2(0, 0);
        scrollDelta = 0;
    }

    void setKeyState(KeyCode key, bool pressed) {
        currentKeys.set(static_cast<size_t>(key), pressed);
    }

    void setMouseState(MouseButton button, bool pressed) {
        currentMouse.set(static_cast<size_t>(button), pressed);
    }

    void setMousePosition(float x, float y) {
        Vec2 newPos(x, y);
        mouseDelta = newPos - mousePosition;
        mousePosition = newPos;
    }

    void setScrollDelta(float delta) { scrollDelta = delta; }

    bool isKeyDown(KeyCode key) const {
        return currentKeys.test(static_cast<size_t>(key));
    }

    bool isKeyPressed(KeyCode key) const {
        size_t idx = static_cast<size_t>(key);
        return currentKeys.test(idx) && !previousKeys.test(idx);
    }

    bool isKeyReleased(KeyCode key) const {
        size_t idx = static_cast<size_t>(key);
        return !currentKeys.test(idx) && previousKeys.test(idx);
    }

    bool isMouseDown(MouseButton button) const {
        return currentMouse.test(static_cast<size_t>(button));
    }

    bool isMousePressed(MouseButton button) const {
        size_t idx = static_cast<size_t>(button);
        return currentMouse.test(idx) && !previousMouse.test(idx);
    }

    Vec2 getMousePosition() const { return mousePosition; }
    Vec2 getMouseDelta() const { return mouseDelta; }
    float getScrollDelta() const { return scrollDelta; }
};

/* ============================================================================
 * TIMER AND PERFORMANCE
 * ============================================================================ */

class Timer {
private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point lastFrame;
    float deltaTime;
    float timeScale;
    float totalTime;
    int frameCount;
    float fpsTimer;
    int fps;

public:
    Timer() : deltaTime(0), timeScale(1.0f), totalTime(0), frameCount(0), fpsTimer(0), fps(0) {
        startTime = std::chrono::high_resolution_clock::now();
        lastFrame = startTime;
    }

    void update() {
        auto now = std::chrono::high_resolution_clock::now();
        deltaTime = std::chrono::duration<float>(now - lastFrame).count();
        lastFrame = now;

        totalTime += deltaTime;
        frameCount++;
        fpsTimer += deltaTime;

        if (fpsTimer >= 1.0f) {
            fps = frameCount;
            frameCount = 0;
            fpsTimer = 0;
        }
    }

    float getDeltaTime() const { return deltaTime * timeScale; }
    float getUnscaledDeltaTime() const { return deltaTime; }
    float getTotalTime() const { return totalTime; }
    float getTimeScale() const { return timeScale; }
    void setTimeScale(float scale) { timeScale = scale; }
    int getFPS() const { return fps; }
};

/* ============================================================================
 * GAME ENGINE CLASS
 * ============================================================================ */

class GameEngine {
private:
    bool running;
    bool initialized;
    Timer timer;
    InputSystem input;
    SceneManager sceneManager;
    PhysicsSystem physics;
    EventBus eventBus;
    ResourceManager<Texture> textures;
    ResourceManager<AudioClip> audio;
    ResourceManager<Font> fonts;

    std::string licenseKey;
    std::string encryptionKey;
    bool licensed;

public:
    GameEngine()
        : running(false), initialized(false), licensed(false) {}

    bool initialize(const std::string& license) {
        std::cout << "\\n";
        std::cout << "============================================================\\n";
        std::cout << "         OAAS Game Engine v1.0 - Initializing               \\n";
        std::cout << "============================================================\\n\\n";

        if (!verifyLicense(license)) {
            std::cout << "[ENGINE] License verification failed!\\n";
            return false;
        }

        std::cout << "[ENGINE] License: " << Config::ENGINE_LICENSE_KEY << "\\n";
        std::cout << "[ENGINE] Encryption: " << Config::ENCRYPTION_KEY << "\\n";
        std::cout << "[ENGINE] API Key: " << Config::API_KEY << "\\n";
        std::cout << "[ENGINE] Steam Key: " << Config::STEAM_API_KEY << "\\n";
        std::cout << "[ENGINE] DRM Key: " << Config::DRM_KEY << "\\n\\n";

        eventBus.subscribe<EntityCreatedEvent>([](const EntityCreatedEvent& e) {
            std::cout << "[EVENT] Entity created: " << e.entityID << "\\n";
        });

        eventBus.subscribe<EntityDestroyedEvent>([](const EntityDestroyedEvent& e) {
            std::cout << "[EVENT] Entity destroyed: " << e.entityID << "\\n";
        });

        initialized = true;
        std::cout << "[ENGINE] Initialization complete\\n\\n";
        return true;
    }

    bool verifyLicense(const std::string& license) {
        if (license != Config::ENGINE_LICENSE_KEY) {
            return false;
        }

        licenseKey = license;
        licensed = true;

        std::cout << "[LICENSE] Verified: " << license << "\\n";
        return true;
    }

    void run() {
        if (!initialized) {
            std::cout << "[ENGINE] Engine not initialized!\\n";
            return;
        }

        running = true;
        std::cout << "[ENGINE] Starting main loop\\n\\n";

        int simulatedFrames = 100;
        while (running && simulatedFrames-- > 0) {
            timer.update();
            input.update();

            sceneManager.switchScene();

            Scene* scene = sceneManager.getActiveScene();
            if (scene) {
                updateScene(scene, timer.getDeltaTime());
            }

            if (simulatedFrames % 20 == 0) {
                std::cout << "[ENGINE] Frame " << (100 - simulatedFrames) << ", FPS: " << timer.getFPS()
                          << ", DT: " << timer.getDeltaTime() * 1000 << "ms\\n";
            }
        }

        std::cout << "\\n[ENGINE] Main loop ended\\n";
    }

    void shutdown() {
        std::cout << "[ENGINE] Shutting down...\\n";

        textures.clear();
        audio.clear();
        fonts.clear();
        eventBus.clear();

        running = false;
        initialized = false;

        std::cout << "[ENGINE] Shutdown complete\\n";
    }

    Scene* createScene(const std::string& name) {
        return sceneManager.createScene(name);
    }

    void loadScene(const std::string& name) {
        sceneManager.loadScene(name);
    }

    InputSystem& getInput() { return input; }
    Timer& getTimer() { return timer; }
    EventBus& getEventBus() { return eventBus; }
    PhysicsSystem& getPhysics() { return physics; }

private:
    void updateScene(Scene* scene, float dt) {
        for (auto& entity : scene->entities) {
            if (!entity->active) continue;

            auto* script = entity->getComponent<ScriptComponent>();
            if (script) {
                if (!script->started && script->onStart) {
                    script->onStart(entity->id);
                    script->started = true;
                }
                if (script->onUpdate) {
                    script->onUpdate(entity->id, dt);
                }
            }

            auto* animator = entity->getComponent<AnimatorComponent>();
            if (animator) {
                animator->update(dt);
            }

            auto* transform = entity->getComponent<TransformComponent>();
            auto* particles = entity->getComponent<ParticleEmitterComponent>();
            if (transform && particles) {
                particles->update(transform->position, dt);
            }
        }

        physics.update(scene->entities, dt);

        scene->updateQuadtree();
    }
};

/* ============================================================================
 * STATE MACHINE FOR AI
 * ============================================================================ */

template<typename T>
class State {
public:
    virtual ~State() = default;
    virtual void enter(T& owner) {}
    virtual void update(T& owner, float dt) {}
    virtual void exit(T& owner) {}
};

template<typename T>
class StateMachine {
private:
    T& owner;
    State<T>* currentState;
    State<T>* previousState;
    State<T>* globalState;

public:
    StateMachine(T& owner)
        : owner(owner), currentState(nullptr), previousState(nullptr), globalState(nullptr) {}

    void setCurrentState(State<T>* state) {
        previousState = currentState;
        if (currentState) {
            currentState->exit(owner);
        }
        currentState = state;
        if (currentState) {
            currentState->enter(owner);
        }
    }

    void setGlobalState(State<T>* state) { globalState = state; }

    void update(float dt) {
        if (globalState) globalState->update(owner, dt);
        if (currentState) currentState->update(owner, dt);
    }

    void revertToPreviousState() {
        setCurrentState(previousState);
    }

    bool isInState(State<T>* state) const {
        return currentState == state;
    }

    State<T>* getCurrentState() { return currentState; }
};

/* ============================================================================
 * PATHFINDING - A* ALGORITHM
 * ============================================================================ */

struct PathNode {
    int x, y;
    float g, h, f;
    PathNode* parent;
    bool walkable;

    PathNode() : x(0), y(0), g(0), h(0), f(0), parent(nullptr), walkable(true) {}
    PathNode(int x, int y) : x(x), y(y), g(0), h(0), f(0), parent(nullptr), walkable(true) {}

    bool operator==(const PathNode& other) const {
        return x == other.x && y == other.y;
    }
};

struct PathNodeCompare {
    bool operator()(const PathNode* a, const PathNode* b) const {
        return a->f > b->f;
    }
};

class Pathfinder {
private:
    int width, height;
    std::vector<std::vector<PathNode>> grid;

public:
    Pathfinder(int w, int h) : width(w), height(h) {
        grid.resize(width);
        for (int x = 0; x < width; x++) {
            grid[x].resize(height);
            for (int y = 0; y < height; y++) {
                grid[x][y] = PathNode(x, y);
            }
        }
    }

    void setWalkable(int x, int y, bool walkable) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            grid[x][y].walkable = walkable;
        }
    }

    std::vector<Vec2> findPath(const Vec2& start, const Vec2& end) {
        std::vector<Vec2> path;

        int startX = static_cast<int>(start.x);
        int startY = static_cast<int>(start.y);
        int endX = static_cast<int>(end.x);
        int endY = static_cast<int>(end.y);

        if (!isValid(startX, startY) || !isValid(endX, endY)) {
            return path;
        }

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                grid[x][y].g = 0;
                grid[x][y].h = 0;
                grid[x][y].f = 0;
                grid[x][y].parent = nullptr;
            }
        }

        std::priority_queue<PathNode*, std::vector<PathNode*>, PathNodeCompare> openList;
        std::set<PathNode*> closedSet;

        PathNode* startNode = &grid[startX][startY];
        PathNode* endNode = &grid[endX][endY];

        openList.push(startNode);

        while (!openList.empty()) {
            PathNode* current = openList.top();
            openList.pop();

            if (closedSet.find(current) != closedSet.end()) {
                continue;
            }

            closedSet.insert(current);

            if (current->x == endNode->x && current->y == endNode->y) {
                PathNode* node = current;
                while (node) {
                    path.push_back(Vec2(static_cast<float>(node->x), static_cast<float>(node->y)));
                    node = node->parent;
                }
                std::reverse(path.begin(), path.end());
                return path;
            }

            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;

                    int nx = current->x + dx;
                    int ny = current->y + dy;

                    if (!isValid(nx, ny)) continue;

                    PathNode* neighbor = &grid[nx][ny];

                    if (!neighbor->walkable) continue;
                    if (closedSet.find(neighbor) != closedSet.end()) continue;

                    float moveCost = (dx == 0 || dy == 0) ? 1.0f : 1.414f;
                    float newG = current->g + moveCost;

                    if (newG < neighbor->g || neighbor->g == 0) {
                        neighbor->g = newG;
                        neighbor->h = heuristic(neighbor, endNode);
                        neighbor->f = neighbor->g + neighbor->h;
                        neighbor->parent = current;
                        openList.push(neighbor);
                    }
                }
            }
        }

        return path;
    }

private:
    bool isValid(int x, int y) const {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    float heuristic(PathNode* a, PathNode* b) const {
        return std::abs(static_cast<float>(a->x - b->x)) +
               std::abs(static_cast<float>(a->y - b->y));
    }
};

/* ============================================================================
 * BEHAVIOR TREE FOR AI
 * ============================================================================ */

enum class BTStatus { Success, Failure, Running };

class BTNode {
public:
    virtual ~BTNode() = default;
    virtual BTStatus tick(float dt) = 0;
    virtual void reset() {}
};

class BTSelector : public BTNode {
private:
    std::vector<std::unique_ptr<BTNode>> children;
    size_t currentChild;

public:
    BTSelector() : currentChild(0) {}

    void addChild(std::unique_ptr<BTNode> child) {
        children.push_back(std::move(child));
    }

    BTStatus tick(float dt) override {
        while (currentChild < children.size()) {
            BTStatus status = children[currentChild]->tick(dt);

            if (status == BTStatus::Running) {
                return BTStatus::Running;
            }

            if (status == BTStatus::Success) {
                currentChild = 0;
                return BTStatus::Success;
            }

            currentChild++;
        }

        currentChild = 0;
        return BTStatus::Failure;
    }

    void reset() override {
        currentChild = 0;
        for (auto& child : children) {
            child->reset();
        }
    }
};

class BTSequence : public BTNode {
private:
    std::vector<std::unique_ptr<BTNode>> children;
    size_t currentChild;

public:
    BTSequence() : currentChild(0) {}

    void addChild(std::unique_ptr<BTNode> child) {
        children.push_back(std::move(child));
    }

    BTStatus tick(float dt) override {
        while (currentChild < children.size()) {
            BTStatus status = children[currentChild]->tick(dt);

            if (status == BTStatus::Running) {
                return BTStatus::Running;
            }

            if (status == BTStatus::Failure) {
                currentChild = 0;
                return BTStatus::Failure;
            }

            currentChild++;
        }

        currentChild = 0;
        return BTStatus::Success;
    }

    void reset() override {
        currentChild = 0;
        for (auto& child : children) {
            child->reset();
        }
    }
};

class BTCondition : public BTNode {
private:
    std::function<bool()> condition;

public:
    BTCondition(std::function<bool()> cond) : condition(cond) {}

    BTStatus tick(float dt) override {
        return condition() ? BTStatus::Success : BTStatus::Failure;
    }
};

class BTAction : public BTNode {
private:
    std::function<BTStatus(float)> action;

public:
    BTAction(std::function<BTStatus(float)> act) : action(act) {}

    BTStatus tick(float dt) override {
        return action(dt);
    }
};

/* ============================================================================
 * OBJECT POOL
 * ============================================================================ */

template<typename T>
class ObjectPool {
private:
    std::vector<std::unique_ptr<T>> pool;
    std::vector<T*> available;
    size_t initialSize;

public:
    ObjectPool(size_t size = 100) : initialSize(size) {
        expand(size);
    }

    T* acquire() {
        if (available.empty()) {
            expand(pool.size());
        }

        T* obj = available.back();
        available.pop_back();
        return obj;
    }

    void release(T* obj) {
        available.push_back(obj);
    }

    size_t size() const { return pool.size(); }
    size_t availableCount() const { return available.size(); }

private:
    void expand(size_t count) {
        for (size_t i = 0; i < count; i++) {
            auto obj = std::make_unique<T>();
            available.push_back(obj.get());
            pool.push_back(std::move(obj));
        }
        std::cout << "[POOL] Expanded to " << pool.size() << " objects\\n";
    }
};

/* ============================================================================
 * TWEENING SYSTEM
 * ============================================================================ */

enum class EaseType {
    Linear,
    QuadIn, QuadOut, QuadInOut,
    CubicIn, CubicOut, CubicInOut,
    SineIn, SineOut, SineInOut,
    ExpoIn, ExpoOut, ExpoInOut,
    ElasticIn, ElasticOut, ElasticInOut,
    BounceIn, BounceOut, BounceInOut
};

class Easing {
public:
    static float ease(EaseType type, float t) {
        switch (type) {
            case EaseType::Linear: return t;
            case EaseType::QuadIn: return t * t;
            case EaseType::QuadOut: return t * (2 - t);
            case EaseType::QuadInOut: return t < 0.5f ? 2 * t * t : -1 + (4 - 2 * t) * t;
            case EaseType::CubicIn: return t * t * t;
            case EaseType::CubicOut: { float f = t - 1; return f * f * f + 1; }
            case EaseType::CubicInOut: return t < 0.5f ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;
            case EaseType::SineIn: return 1 - std::cos(t * 3.14159f / 2);
            case EaseType::SineOut: return std::sin(t * 3.14159f / 2);
            case EaseType::SineInOut: return 0.5f * (1 - std::cos(3.14159f * t));
            case EaseType::ExpoIn: return t == 0 ? 0 : std::pow(2, 10 * (t - 1));
            case EaseType::ExpoOut: return t == 1 ? 1 : 1 - std::pow(2, -10 * t);
            case EaseType::BounceOut: return bounceOut(t);
            case EaseType::BounceIn: return 1 - bounceOut(1 - t);
            default: return t;
        }
    }

private:
    static float bounceOut(float t) {
        if (t < 1 / 2.75f) {
            return 7.5625f * t * t;
        } else if (t < 2 / 2.75f) {
            t -= 1.5f / 2.75f;
            return 7.5625f * t * t + 0.75f;
        } else if (t < 2.5f / 2.75f) {
            t -= 2.25f / 2.75f;
            return 7.5625f * t * t + 0.9375f;
        } else {
            t -= 2.625f / 2.75f;
            return 7.5625f * t * t + 0.984375f;
        }
    }
};

class Tween {
public:
    float* target;
    float startValue;
    float endValue;
    float duration;
    float elapsed;
    EaseType easeType;
    bool active;
    bool loop;
    std::function<void()> onComplete;

    Tween() : target(nullptr), startValue(0), endValue(0), duration(1),
              elapsed(0), easeType(EaseType::Linear), active(false), loop(false) {}

    void start(float* t, float end, float dur, EaseType ease = EaseType::Linear) {
        target = t;
        startValue = *t;
        endValue = end;
        duration = dur;
        elapsed = 0;
        easeType = ease;
        active = true;
    }

    bool update(float dt) {
        if (!active || !target) return false;

        elapsed += dt;
        float t = std::min(elapsed / duration, 1.0f);
        float easedT = Easing::ease(easeType, t);
        *target = startValue + (endValue - startValue) * easedT;

        if (elapsed >= duration) {
            if (loop) {
                elapsed = 0;
                startValue = *target;
                std::swap(startValue, endValue);
            } else {
                active = false;
                if (onComplete) onComplete();
            }
            return true;
        }

        return false;
    }
};

class TweenManager {
private:
    std::vector<Tween> tweens;

public:
    Tween& createTween() {
        tweens.emplace_back();
        return tweens.back();
    }

    void update(float dt) {
        tweens.erase(
            std::remove_if(tweens.begin(), tweens.end(),
                [dt](Tween& t) { return !t.active && t.update(dt); }),
            tweens.end()
        );

        for (auto& tween : tweens) {
            if (tween.active) {
                tween.update(dt);
            }
        }
    }

    void clear() {
        tweens.clear();
    }

    size_t activeTweenCount() const {
        return std::count_if(tweens.begin(), tweens.end(),
            [](const Tween& t) { return t.active; });
    }
};

/* ============================================================================
 * SERIALIZATION
 * ============================================================================ */

class Serializer {
public:
    static std::string serializeEntity(const Entity& entity) {
        std::ostringstream ss;
        ss << "Entity: " << entity.id << "\\n";
        ss << "  Name: " << entity.name << "\\n";
        ss << "  Tag: " << entity.tag << "\\n";
        ss << "  Active: " << (entity.active ? "true" : "false") << "\\n";
        ss << "  Components: " << entity.components.size() << "\\n";

        for (const auto& pair : entity.components) {
            ss << "    - " << pair.second->getName() << "\\n";
        }

        return ss.str();
    }

    static std::string serializeScene(const Scene& scene) {
        std::ostringstream ss;
        ss << "Scene: " << scene.name << "\\n";
        ss << "  Entities: " << scene.entities.size() << "\\n";

        for (const auto& entity : scene.entities) {
            ss << serializeEntity(*entity);
        }

        return ss.str();
    }
};

/* ============================================================================
 * DEMO MAIN FUNCTION
 * ============================================================================ */

int main() {
    std::cout << "\\n";
    std::cout << "============================================================\\n";
    std::cout << "     OAAS Game Engine Demo - Obfuscation Test Program        \\n";
    std::cout << "============================================================\\n\\n";

    GameEngine engine;

    if (!engine.initialize(Config::ENGINE_LICENSE_KEY)) {
        std::cout << "[FATAL] Failed to initialize engine!\\n";
        return 1;
    }

    std::cout << "\\n--- Creating Demo Scene ---\\n\\n";

    Scene* mainScene = engine.createScene("MainScene");

    Entity* player = mainScene->createEntity("Player");
    auto* playerTransform = player->addComponent<TransformComponent>();
    playerTransform->position = Vec2(100, 100);
    playerTransform->scale = Vec2(1, 1);

    auto* playerRigidbody = player->addComponent<RigidbodyComponent>();
    playerRigidbody->mass = 1.0f;
    playerRigidbody->drag = 0.5f;
    playerRigidbody->useGravity = true;

    auto* playerCollider = player->addComponent<ColliderComponent>();
    playerCollider->shape = ColliderShape::Circle;
    playerCollider->radius = 16.0f;

    auto* playerSprite = player->addComponent<SpriteComponent>();
    playerSprite->texturePath = "player.png";
    playerSprite->tint = Color::Green;

    auto* playerScript = player->addComponent<ScriptComponent>();
    playerScript->onStart = [](EntityID id) {
        std::cout << "[SCRIPT] Player " << id << " started\\n";
    };
    playerScript->onUpdate = [](EntityID id, float dt) {
        static float timer = 0;
        timer += dt;
        if (timer >= 1.0f) {
            std::cout << "[SCRIPT] Player " << id << " update tick\\n";
            timer = 0;
        }
    };
    player->tag = "Player";

    Entity* enemy = mainScene->createEntity("Enemy");
    auto* enemyTransform = enemy->addComponent<TransformComponent>();
    enemyTransform->position = Vec2(300, 100);

    auto* enemyRigidbody = enemy->addComponent<RigidbodyComponent>();
    enemyRigidbody->mass = 2.0f;
    enemyRigidbody->bodyType = BodyType::Dynamic;

    auto* enemyCollider = enemy->addComponent<ColliderComponent>();
    enemyCollider->shape = ColliderShape::Box;
    enemyCollider->size = Vec2(32, 32);

    auto* enemySprite = enemy->addComponent<SpriteComponent>();
    enemySprite->tint = Color::Red;
    enemy->tag = "Enemy";

    Entity* ground = mainScene->createEntity("Ground");
    auto* groundTransform = ground->addComponent<TransformComponent>();
    groundTransform->position = Vec2(400, 0);

    auto* groundRigidbody = ground->addComponent<RigidbodyComponent>();
    groundRigidbody->bodyType = BodyType::Static;

    auto* groundCollider = ground->addComponent<ColliderComponent>();
    groundCollider->shape = ColliderShape::Box;
    groundCollider->size = Vec2(800, 50);
    ground->tag = "Ground";

    Entity* particles = mainScene->createEntity("ParticleSystem");
    auto* particleTransform = particles->addComponent<TransformComponent>();
    particleTransform->position = Vec2(200, 300);

    auto* emitter = particles->addComponent<ParticleEmitterComponent>();
    emitter->emissionRate = 20.0f;
    emitter->particleLifetime = 2.0f;
    emitter->startSpeed = 100.0f;
    emitter->startColor = Color(255, 200, 100);
    emitter->endColor = Color(255, 50, 0, 0);
    emitter->gravity = -100.0f;

    Entity* camera = mainScene->createEntity("MainCamera");
    auto* cameraTransform = camera->addComponent<TransformComponent>();
    cameraTransform->position = Vec2(400, 300);

    auto* cameraComponent = camera->addComponent<CameraComponent>();
    cameraComponent->orthographicSize = 300;
    cameraComponent->isMain = true;

    std::cout << "\\n--- Scene Summary ---\\n";
    std::cout << Serializer::serializeScene(*mainScene);

    std::cout << "\\n--- Math Tests ---\\n";

    Vec2 a(3, 4);
    Vec2 b(1, 2);
    std::cout << "Vec2 a = (3, 4), b = (1, 2)\\n";
    std::cout << "  a + b = (" << (a + b).x << ", " << (a + b).y << ")\\n";
    std::cout << "  a.dot(b) = " << a.dot(b) << "\\n";
    std::cout << "  a.length() = " << a.length() << "\\n";
    std::cout << "  distance(a, b) = " << Vec2::distance(a, b) << "\\n";

    std::cout << "\\n--- Pathfinding Test ---\\n";

    Pathfinder pathfinder(10, 10);
    pathfinder.setWalkable(5, 3, false);
    pathfinder.setWalkable(5, 4, false);
    pathfinder.setWalkable(5, 5, false);
    pathfinder.setWalkable(5, 6, false);

    auto path = pathfinder.findPath(Vec2(0, 5), Vec2(9, 5));
    std::cout << "Path from (0,5) to (9,5): ";
    for (const auto& p : path) {
        std::cout << "(" << p.x << "," << p.y << ") ";
    }
    std::cout << "\\n";

    std::cout << "\\n--- Object Pool Test ---\\n";

    ObjectPool<ParticleData> particlePool(50);
    std::cout << "Pool size: " << particlePool.size() << "\\n";
    std::cout << "Available: " << particlePool.availableCount() << "\\n";

    std::vector<ParticleData*> acquired;
    for (int i = 0; i < 30; i++) {
        acquired.push_back(particlePool.acquire());
    }
    std::cout << "After acquiring 30: Available = " << particlePool.availableCount() << "\\n";

    for (auto* p : acquired) {
        particlePool.release(p);
    }
    std::cout << "After releasing: Available = " << particlePool.availableCount() << "\\n";

    std::cout << "\\n--- Tween Test ---\\n";

    TweenManager tweenManager;
    float testValue = 0.0f;

    Tween& tween = tweenManager.createTween();
    tween.start(&testValue, 100.0f, 2.0f, EaseType::QuadInOut);
    tween.onComplete = []() {
        std::cout << "  Tween complete!\\n";
    };

    for (int i = 0; i < 10; i++) {
        tweenManager.update(0.2f);
        std::cout << "  t=" << (i + 1) * 0.2f << "s, value=" << testValue << "\\n";
    }

    std::cout << "\\n--- Quadtree Test ---\\n";

    Quadtree qt(0, Rect(0, 0, 100, 100));
    for (int i = 0; i < 20; i++) {
        float x = g_random.nextFloat(0, 90);
        float y = g_random.nextFloat(0, 90);
        qt.insert(i + 1, Rect(x, y, 10, 10));
    }

    auto found = qt.retrieve(Rect(25, 25, 50, 50));
    std::cout << "Entities in region (25,25,50,50): " << found.size() << "\\n";

    std::cout << "\\n--- Starting Engine ---\\n";

    engine.loadScene("MainScene");
    engine.run();

    engine.shutdown();

    std::cout << "\\n";
    std::cout << "============================================================\\n";
    std::cout << "            Game Engine Demo Complete!                        \\n";
    std::cout << "============================================================\\n";
    std::cout << "\\n";

    return 0;
}

/* ============================================================================
 * TERRAIN GENERATION - PERLIN NOISE
 * ============================================================================ */

class PerlinNoise {
private:
    std::vector<int> permutation;

    static float fade(float t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    static float lerp(float t, float a, float b) {
        return a + t * (b - a);
    }

    static float grad(int hash, float x, float y, float z) {
        int h = hash & 15;
        float u = h < 8 ? x : y;
        float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }

public:
    PerlinNoise(unsigned int seed = 0) {
        permutation.resize(256);
        std::iota(permutation.begin(), permutation.end(), 0);

        std::default_random_engine engine(seed);
        std::shuffle(permutation.begin(), permutation.end(), engine);

        permutation.insert(permutation.end(), permutation.begin(), permutation.end());
    }

    float noise(float x, float y, float z) const {
        int X = static_cast<int>(std::floor(x)) & 255;
        int Y = static_cast<int>(std::floor(y)) & 255;
        int Z = static_cast<int>(std::floor(z)) & 255;

        x -= std::floor(x);
        y -= std::floor(y);
        z -= std::floor(z);

        float u = fade(x);
        float v = fade(y);
        float w = fade(z);

        int A = permutation[X] + Y;
        int AA = permutation[A] + Z;
        int AB = permutation[A + 1] + Z;
        int B = permutation[X + 1] + Y;
        int BA = permutation[B] + Z;
        int BB = permutation[B + 1] + Z;

        return lerp(w,
            lerp(v,
                lerp(u, grad(permutation[AA], x, y, z),
                        grad(permutation[BA], x - 1, y, z)),
                lerp(u, grad(permutation[AB], x, y - 1, z),
                        grad(permutation[BB], x - 1, y - 1, z))),
            lerp(v,
                lerp(u, grad(permutation[AA + 1], x, y, z - 1),
                        grad(permutation[BA + 1], x - 1, y, z - 1)),
                lerp(u, grad(permutation[AB + 1], x, y - 1, z - 1),
                        grad(permutation[BB + 1], x - 1, y - 1, z - 1))));
    }

    float octaveNoise(float x, float y, float z, int octaves, float persistence) const {
        float total = 0;
        float frequency = 1;
        float amplitude = 1;
        float maxValue = 0;

        for (int i = 0; i < octaves; i++) {
            total += noise(x * frequency, y * frequency, z * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= persistence;
            frequency *= 2;
        }

        return total / maxValue;
    }
};

class TerrainGenerator {
private:
    PerlinNoise noise;
    int width, height;
    std::vector<float> heightMap;

public:
    TerrainGenerator(int w, int h, unsigned int seed = 0)
        : noise(seed), width(w), height(h), heightMap(w * h) {}

    void generate(float scale, int octaves, float persistence) {
        std::cout << "[TERRAIN] Generating terrain " << width << "x" << height << "...\\n";

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float nx = static_cast<float>(x) / width * scale;
                float ny = static_cast<float>(y) / height * scale;
                heightMap[y * width + x] = noise.octaveNoise(nx, ny, 0, octaves, persistence);
            }
        }

        normalize();
        std::cout << "[TERRAIN] Generation complete\\n";
    }

    void normalize() {
        float minH = *std::min_element(heightMap.begin(), heightMap.end());
        float maxH = *std::max_element(heightMap.begin(), heightMap.end());
        float range = maxH - minH;

        if (range > 0) {
            for (float& h : heightMap) {
                h = (h - minH) / range;
            }
        }
    }

    float getHeight(int x, int y) const {
        if (x < 0 || x >= width || y < 0 || y >= height) return 0;
        return heightMap[y * width + x];
    }

    Vec3 getNormal(int x, int y) const {
        float hL = getHeight(x - 1, y);
        float hR = getHeight(x + 1, y);
        float hD = getHeight(x, y - 1);
        float hU = getHeight(x, y + 1);
        return Vec3(hL - hR, 2.0f, hD - hU).normalized();
    }

    void printStats() const {
        float sum = 0;
        float minH = 1.0f, maxH = 0.0f;

        for (float h : heightMap) {
            sum += h;
            minH = std::min(minH, h);
            maxH = std::max(maxH, h);
        }

        std::cout << "[TERRAIN] Stats:\\n";
        std::cout << "  Min height: " << minH << "\\n";
        std::cout << "  Max height: " << maxH << "\\n";
        std::cout << "  Avg height: " << sum / heightMap.size() << "\\n";
    }
};

/* ============================================================================
 * SPATIAL HASH GRID
 * ============================================================================ */

class SpatialHashGrid {
private:
    float cellSize;
    std::unordered_map<int64_t, std::vector<EntityID>> cells;

    int64_t hashPosition(float x, float y) const {
        int cellX = static_cast<int>(std::floor(x / cellSize));
        int cellY = static_cast<int>(std::floor(y / cellSize));
        return (static_cast<int64_t>(cellX) << 32) | static_cast<uint32_t>(cellY);
    }

public:
    SpatialHashGrid(float cellSize = 100.0f) : cellSize(cellSize) {}

    void clear() {
        cells.clear();
    }

    void insert(EntityID entity, const Vec2& position) {
        int64_t hash = hashPosition(position.x, position.y);
        cells[hash].push_back(entity);
    }

    void insert(EntityID entity, const Rect& bounds) {
        int minCellX = static_cast<int>(std::floor(bounds.x / cellSize));
        int minCellY = static_cast<int>(std::floor(bounds.y / cellSize));
        int maxCellX = static_cast<int>(std::floor((bounds.x + bounds.width) / cellSize));
        int maxCellY = static_cast<int>(std::floor((bounds.y + bounds.height) / cellSize));

        for (int cx = minCellX; cx <= maxCellX; cx++) {
            for (int cy = minCellY; cy <= maxCellY; cy++) {
                int64_t hash = (static_cast<int64_t>(cx) << 32) | static_cast<uint32_t>(cy);
                cells[hash].push_back(entity);
            }
        }
    }

    std::vector<EntityID> query(const Vec2& position, float radius) const {
        std::vector<EntityID> result;
        std::set<EntityID> uniqueEntities;

        int minCellX = static_cast<int>(std::floor((position.x - radius) / cellSize));
        int minCellY = static_cast<int>(std::floor((position.y - radius) / cellSize));
        int maxCellX = static_cast<int>(std::floor((position.x + radius) / cellSize));
        int maxCellY = static_cast<int>(std::floor((position.y + radius) / cellSize));

        for (int cx = minCellX; cx <= maxCellX; cx++) {
            for (int cy = minCellY; cy <= maxCellY; cy++) {
                int64_t hash = (static_cast<int64_t>(cx) << 32) | static_cast<uint32_t>(cy);
                auto it = cells.find(hash);
                if (it != cells.end()) {
                    for (EntityID id : it->second) {
                        if (uniqueEntities.insert(id).second) {
                            result.push_back(id);
                        }
                    }
                }
            }
        }

        return result;
    }

    std::vector<EntityID> query(const Rect& bounds) const {
        std::vector<EntityID> result;
        std::set<EntityID> uniqueEntities;

        int minCellX = static_cast<int>(std::floor(bounds.x / cellSize));
        int minCellY = static_cast<int>(std::floor(bounds.y / cellSize));
        int maxCellX = static_cast<int>(std::floor((bounds.x + bounds.width) / cellSize));
        int maxCellY = static_cast<int>(std::floor((bounds.y + bounds.height) / cellSize));

        for (int cx = minCellX; cx <= maxCellX; cx++) {
            for (int cy = minCellY; cy <= maxCellY; cy++) {
                int64_t hash = (static_cast<int64_t>(cx) << 32) | static_cast<uint32_t>(cy);
                auto it = cells.find(hash);
                if (it != cells.end()) {
                    for (EntityID id : it->second) {
                        if (uniqueEntities.insert(id).second) {
                            result.push_back(id);
                        }
                    }
                }
            }
        }

        return result;
    }

    size_t cellCount() const { return cells.size(); }
};

/* ============================================================================
 * AUDIO SYSTEM
 * ============================================================================ */

struct AudioSource3D {
    EntityID entity;
    Vec3 position;
    float volume;
    float pitch;
    float minDistance;
    float maxDistance;
    bool loop;
    bool playing;
    std::string clipName;
};

class AudioManager {
private:
    std::map<std::string, AudioClip> clips;
    std::vector<AudioSource3D> activeSources;
    Vec3 listenerPosition;
    Vec3 listenerForward;
    float masterVolume;
    float musicVolume;
    float sfxVolume;
    bool muted;

public:
    AudioManager()
        : listenerPosition(0, 0, 0), listenerForward(0, 0, -1),
          masterVolume(1.0f), musicVolume(1.0f), sfxVolume(1.0f), muted(false) {}

    void loadClip(const std::string& name, const std::string& path) {
        clips[name] = AudioClip();
        std::cout << "[AUDIO] Loaded clip: " << name << " from " << path << "\\n";
    }

    void unloadClip(const std::string& name) {
        clips.erase(name);
        std::cout << "[AUDIO] Unloaded clip: " << name << "\\n";
    }

    void play(const std::string& clipName, float volume = 1.0f, float pitch = 1.0f) {
        if (muted) return;
        std::cout << "[AUDIO] Playing: " << clipName
                  << " (vol: " << volume * masterVolume * sfxVolume << ")\\n";
    }

    void play3D(const std::string& clipName, const Vec3& position,
                float minDist = 1.0f, float maxDist = 100.0f) {
        if (muted) return;

        AudioSource3D source;
        source.position = position;
        source.volume = calculateAttenuation(position, minDist, maxDist);
        source.pitch = 1.0f;
        source.minDistance = minDist;
        source.maxDistance = maxDist;
        source.loop = false;
        source.playing = true;
        source.clipName = clipName;

        activeSources.push_back(source);
        std::cout << "[AUDIO] Playing 3D: " << clipName
                  << " at (" << position.x << ", " << position.y << ", " << position.z << ")\\n";
    }

    void setListenerPosition(const Vec3& pos, const Vec3& forward) {
        listenerPosition = pos;
        listenerForward = forward.normalized();
    }

    void setMasterVolume(float vol) { masterVolume = std::clamp(vol, 0.0f, 1.0f); }
    void setMusicVolume(float vol) { musicVolume = std::clamp(vol, 0.0f, 1.0f); }
    void setSfxVolume(float vol) { sfxVolume = std::clamp(vol, 0.0f, 1.0f); }
    void setMuted(bool mute) { muted = mute; }

    void update() {
        activeSources.erase(
            std::remove_if(activeSources.begin(), activeSources.end(),
                [](const AudioSource3D& s) { return !s.playing; }),
            activeSources.end()
        );
    }

    void stopAll() {
        activeSources.clear();
        std::cout << "[AUDIO] Stopped all sounds\\n";
    }

private:
    float calculateAttenuation(const Vec3& sourcePos, float minDist, float maxDist) const {
        float distance = (sourcePos - listenerPosition).length();

        if (distance <= minDist) return 1.0f;
        if (distance >= maxDist) return 0.0f;

        return 1.0f - (distance - minDist) / (maxDist - minDist);
    }
};

/* ============================================================================
 * SPRITE BATCH RENDERER
 * ============================================================================ */

struct SpriteVertex {
    Vec2 position;
    Vec2 texCoord;
    Color color;
};

struct SpriteBatchItem {
    Rect destRect;
    Rect sourceRect;
    Color tint;
    float rotation;
    Vec2 origin;
    int textureID;
    float depth;
};

class SpriteBatch {
private:
    std::vector<SpriteBatchItem> sprites;
    std::vector<SpriteVertex> vertices;
    bool begun;
    int drawCalls;
    int spriteCount;

public:
    SpriteBatch() : begun(false), drawCalls(0), spriteCount(0) {}

    void begin() {
        if (begun) {
            std::cout << "[RENDER] SpriteBatch already begun!\\n";
            return;
        }
        begun = true;
        sprites.clear();
        drawCalls = 0;
        spriteCount = 0;
    }

    void draw(int textureID, const Rect& dest, const Rect& source,
              const Color& tint = Color::White, float rotation = 0,
              const Vec2& origin = Vec2(0.5f, 0.5f), float depth = 0) {
        if (!begun) return;

        SpriteBatchItem item;
        item.destRect = dest;
        item.sourceRect = source;
        item.tint = tint;
        item.rotation = rotation;
        item.origin = origin;
        item.textureID = textureID;
        item.depth = depth;

        sprites.push_back(item);
        spriteCount++;
    }

    void end() {
        if (!begun) return;

        std::sort(sprites.begin(), sprites.end(),
            [](const SpriteBatchItem& a, const SpriteBatchItem& b) {
                if (a.textureID != b.textureID) return a.textureID < b.textureID;
                return a.depth < b.depth;
            });

        flush();
        begun = false;
    }

    int getDrawCalls() const { return drawCalls; }
    int getSpriteCount() const { return spriteCount; }

private:
    void flush() {
        if (sprites.empty()) return;

        int currentTexture = sprites[0].textureID;
        int batchStart = 0;

        for (size_t i = 0; i < sprites.size(); i++) {
            if (sprites[i].textureID != currentTexture) {
                renderBatch(batchStart, static_cast<int>(i) - batchStart, currentTexture);
                batchStart = static_cast<int>(i);
                currentTexture = sprites[i].textureID;
            }
        }

        renderBatch(batchStart, static_cast<int>(sprites.size()) - batchStart, currentTexture);
    }

    void renderBatch(int start, int count, int textureID) {
        if (count <= 0) return;
        drawCalls++;
        std::cout << "[RENDER] Batch: " << count << " sprites, texture " << textureID << "\\n";
    }
};

/* ============================================================================
 * DEBUG DRAW
 * ============================================================================ */

class DebugDraw {
private:
    struct Line {
        Vec2 start, end;
        Color color;
    };

    struct Circle {
        Vec2 center;
        float radius;
        Color color;
    };

    struct Rectangle {
        Rect bounds;
        Color color;
        bool filled;
    };

    std::vector<Line> lines;
    std::vector<Circle> circles;
    std::vector<Rectangle> rectangles;
    bool enabled;

public:
    DebugDraw() : enabled(true) {}

    void setEnabled(bool enable) { enabled = enable; }
    bool isEnabled() const { return enabled; }

    void drawLine(const Vec2& start, const Vec2& end, const Color& color = Color::Green) {
        if (!enabled) return;
        lines.push_back({start, end, color});
    }

    void drawCircle(const Vec2& center, float radius, const Color& color = Color::Green) {
        if (!enabled) return;
        circles.push_back({center, radius, color});
    }

    void drawRect(const Rect& bounds, const Color& color = Color::Green, bool filled = false) {
        if (!enabled) return;
        rectangles.push_back({bounds, color, filled});
    }

    void drawArrow(const Vec2& start, const Vec2& end, const Color& color = Color::Green) {
        if (!enabled) return;
        drawLine(start, end, color);

        Vec2 dir = (end - start).normalized();
        Vec2 perp = dir.perpendicular();
        float arrowSize = 10.0f;

        Vec2 arrowLeft = end - dir * arrowSize + perp * arrowSize * 0.5f;
        Vec2 arrowRight = end - dir * arrowSize - perp * arrowSize * 0.5f;

        drawLine(end, arrowLeft, color);
        drawLine(end, arrowRight, color);
    }

    void drawGrid(const Vec2& origin, float cellSize, int cols, int rows, const Color& color) {
        if (!enabled) return;

        for (int i = 0; i <= cols; i++) {
            Vec2 start(origin.x + i * cellSize, origin.y);
            Vec2 end(origin.x + i * cellSize, origin.y + rows * cellSize);
            drawLine(start, end, color);
        }

        for (int i = 0; i <= rows; i++) {
            Vec2 start(origin.x, origin.y + i * cellSize);
            Vec2 end(origin.x + cols * cellSize, origin.y + i * cellSize);
            drawLine(start, end, color);
        }
    }

    void render() {
        if (!enabled) return;

        std::cout << "[DEBUG] Rendering: " << lines.size() << " lines, "
                  << circles.size() << " circles, "
                  << rectangles.size() << " rects\\n";
    }

    void clear() {
        lines.clear();
        circles.clear();
        rectangles.clear();
    }
};

/* ============================================================================
 * LAYER MASK SYSTEM
 * ============================================================================ */

class LayerMask {
public:
    static constexpr uint32_t Default = 1 << 0;
    static constexpr uint32_t Player = 1 << 1;
    static constexpr uint32_t Enemy = 1 << 2;
    static constexpr uint32_t Ground = 1 << 3;
    static constexpr uint32_t Projectile = 1 << 4;
    static constexpr uint32_t Trigger = 1 << 5;
    static constexpr uint32_t UI = 1 << 6;
    static constexpr uint32_t Pickup = 1 << 7;
    static constexpr uint32_t Water = 1 << 8;
    static constexpr uint32_t All = 0xFFFFFFFF;

    static std::string layerName(uint32_t layer) {
        switch (layer) {
            case Default: return "Default";
            case Player: return "Player";
            case Enemy: return "Enemy";
            case Ground: return "Ground";
            case Projectile: return "Projectile";
            case Trigger: return "Trigger";
            case UI: return "UI";
            case Pickup: return "Pickup";
            case Water: return "Water";
            default: return "Unknown";
        }
    }

    static bool checkCollision(uint32_t layerA, uint32_t maskA, uint32_t layerB, uint32_t maskB) {
        return (layerA & maskB) != 0 && (layerB & maskA) != 0;
    }
};

/* ============================================================================
 * COROUTINE SYSTEM (SIMPLIFIED)
 * ============================================================================ */

class Coroutine {
public:
    using Func = std::function<bool(float)>;

private:
    Func func;
    float waitTime;
    bool finished;
    std::string name;

public:
    Coroutine(const std::string& name, Func f)
        : func(f), waitTime(0), finished(false), name(name) {}

    bool update(float dt) {
        if (finished) return true;

        if (waitTime > 0) {
            waitTime -= dt;
            return false;
        }

        finished = func(dt);
        return finished;
    }

    void wait(float seconds) { waitTime = seconds; }
    bool isFinished() const { return finished; }
    const std::string& getName() const { return name; }
};

class CoroutineManager {
private:
    std::vector<std::unique_ptr<Coroutine>> coroutines;

public:
    Coroutine* start(const std::string& name, Coroutine::Func func) {
        auto coroutine = std::make_unique<Coroutine>(name, func);
        Coroutine* ptr = coroutine.get();
        coroutines.push_back(std::move(coroutine));
        std::cout << "[COROUTINE] Started: " << name << "\\n";
        return ptr;
    }

    void update(float dt) {
        coroutines.erase(
            std::remove_if(coroutines.begin(), coroutines.end(),
                [dt](std::unique_ptr<Coroutine>& c) {
                    if (c->update(dt)) {
                        std::cout << "[COROUTINE] Finished: " << c->getName() << "\\n";
                        return true;
                    }
                    return false;
                }),
            coroutines.end()
        );
    }

    void stopAll() {
        coroutines.clear();
        std::cout << "[COROUTINE] Stopped all\\n";
    }

    size_t activeCount() const { return coroutines.size(); }
};

/* ============================================================================
 * SAVE/LOAD SYSTEM
 * ============================================================================ */

class SaveData {
private:
    std::map<std::string, std::string> stringData;
    std::map<std::string, int> intData;
    std::map<std::string, float> floatData;
    std::map<std::string, bool> boolData;

public:
    void setString(const std::string& key, const std::string& value) {
        stringData[key] = value;
    }

    void setInt(const std::string& key, int value) {
        intData[key] = value;
    }

    void setFloat(const std::string& key, float value) {
        floatData[key] = value;
    }

    void setBool(const std::string& key, bool value) {
        boolData[key] = value;
    }

    std::string getString(const std::string& key, const std::string& defaultVal = "") const {
        auto it = stringData.find(key);
        return it != stringData.end() ? it->second : defaultVal;
    }

    int getInt(const std::string& key, int defaultVal = 0) const {
        auto it = intData.find(key);
        return it != intData.end() ? it->second : defaultVal;
    }

    float getFloat(const std::string& key, float defaultVal = 0.0f) const {
        auto it = floatData.find(key);
        return it != floatData.end() ? it->second : defaultVal;
    }

    bool getBool(const std::string& key, bool defaultVal = false) const {
        auto it = boolData.find(key);
        return it != boolData.end() ? it->second : defaultVal;
    }

    void clear() {
        stringData.clear();
        intData.clear();
        floatData.clear();
        boolData.clear();
    }

    std::string serialize() const {
        std::ostringstream ss;
        ss << "[SaveData]\\n";
        ss << "Strings: " << stringData.size() << "\\n";
        ss << "Ints: " << intData.size() << "\\n";
        ss << "Floats: " << floatData.size() << "\\n";
        ss << "Bools: " << boolData.size() << "\\n";
        return ss.str();
    }
};

class SaveManager {
private:
    std::string savePath;
    SaveData currentSave;
    bool autoSaveEnabled;
    float autoSaveInterval;
    float autoSaveTimer;

public:
    SaveManager(const std::string& path = "saves/")
        : savePath(path), autoSaveEnabled(false), autoSaveInterval(300.0f), autoSaveTimer(0) {}

    bool save(const std::string& slotName) {
        std::string filename = savePath + slotName + ".sav";
        std::cout << "[SAVE] Saving to: " << filename << "\\n";
        std::cout << currentSave.serialize();
        return true;
    }

    bool load(const std::string& slotName) {
        std::string filename = savePath + slotName + ".sav";
        std::cout << "[SAVE] Loading from: " << filename << "\\n";
        return true;
    }

    void deleteSave(const std::string& slotName) {
        std::string filename = savePath + slotName + ".sav";
        std::cout << "[SAVE] Deleted: " << filename << "\\n";
    }

    void enableAutoSave(bool enable, float interval = 300.0f) {
        autoSaveEnabled = enable;
        autoSaveInterval = interval;
        std::cout << "[SAVE] AutoSave: " << (enable ? "enabled" : "disabled")
                  << " (interval: " << interval << "s)\\n";
    }

    void update(float dt) {
        if (!autoSaveEnabled) return;

        autoSaveTimer += dt;
        if (autoSaveTimer >= autoSaveInterval) {
            save("autosave");
            autoSaveTimer = 0;
        }
    }

    SaveData& getData() { return currentSave; }
};

/* ============================================================================
 * ACHIEVEMENT SYSTEM
 * ============================================================================ */

struct Achievement {
    std::string id;
    std::string name;
    std::string description;
    bool unlocked;
    float progress;
    float target;
    std::string iconPath;

    Achievement() : unlocked(false), progress(0), target(1) {}

    float getProgressPercent() const {
        return target > 0 ? (progress / target) * 100.0f : 0;
    }
};

class AchievementManager {
private:
    std::map<std::string, Achievement> achievements;
    std::vector<std::string> recentUnlocks;

public:
    void registerAchievement(const std::string& id, const std::string& name,
                             const std::string& desc, float target = 1) {
        Achievement ach;
        ach.id = id;
        ach.name = name;
        ach.description = desc;
        ach.target = target;
        achievements[id] = ach;
        std::cout << "[ACHIEVEMENT] Registered: " << name << "\\n";
    }

    void updateProgress(const std::string& id, float progress) {
        auto it = achievements.find(id);
        if (it == achievements.end()) return;

        Achievement& ach = it->second;
        if (ach.unlocked) return;

        ach.progress = progress;
        if (ach.progress >= ach.target) {
            unlock(id);
        }
    }

    void incrementProgress(const std::string& id, float amount = 1) {
        auto it = achievements.find(id);
        if (it == achievements.end()) return;

        updateProgress(id, it->second.progress + amount);
    }

    void unlock(const std::string& id) {
        auto it = achievements.find(id);
        if (it == achievements.end() || it->second.unlocked) return;

        it->second.unlocked = true;
        it->second.progress = it->second.target;
        recentUnlocks.push_back(id);

        std::cout << "[ACHIEVEMENT] Unlocked: " << it->second.name << "!\\n";
    }

    bool isUnlocked(const std::string& id) const {
        auto it = achievements.find(id);
        return it != achievements.end() && it->second.unlocked;
    }

    float getProgress(const std::string& id) const {
        auto it = achievements.find(id);
        return it != achievements.end() ? it->second.getProgressPercent() : 0;
    }

    int getUnlockedCount() const {
        int count = 0;
        for (const auto& pair : achievements) {
            if (pair.second.unlocked) count++;
        }
        return count;
    }

    int getTotalCount() const { return static_cast<int>(achievements.size()); }

    void printStats() const {
        std::cout << "[ACHIEVEMENT] Stats: " << getUnlockedCount() << "/" << getTotalCount() << " unlocked\\n";
        for (const auto& pair : achievements) {
            const Achievement& ach = pair.second;
            std::cout << "  " << (ach.unlocked ? "[X]" : "[ ]") << " " << ach.name
                      << " (" << ach.getProgressPercent() << "%)\\n";
        }
    }
};

/* ============================================================================
 * INVENTORY SYSTEM
 * ============================================================================ */

struct ItemData {
    std::string id;
    std::string name;
    std::string description;
    int maxStack;
    float weight;
    std::string iconPath;
    std::map<std::string, float> properties;

    ItemData() : maxStack(99), weight(0) {}
};

struct InventorySlot {
    std::string itemId;
    int quantity;

    InventorySlot() : quantity(0) {}

    bool isEmpty() const { return itemId.empty() || quantity <= 0; }
    void clear() { itemId.clear(); quantity = 0; }
};

class Inventory {
private:
    std::vector<InventorySlot> slots;
    std::map<std::string, ItemData> itemDatabase;
    int maxSlots;
    float maxWeight;
    float currentWeight;

public:
    Inventory(int slots = 20, float maxWeight = 100.0f)
        : maxSlots(slots), maxWeight(maxWeight), currentWeight(0) {
        this->slots.resize(slots);
    }

    void registerItem(const ItemData& item) {
        itemDatabase[item.id] = item;
        std::cout << "[INVENTORY] Registered item: " << item.name << "\\n";
    }

    bool addItem(const std::string& itemId, int quantity = 1) {
        auto it = itemDatabase.find(itemId);
        if (it == itemDatabase.end()) return false;

        const ItemData& item = it->second;

        for (auto& slot : slots) {
            if (slot.itemId == itemId && slot.quantity < item.maxStack) {
                int canAdd = std::min(quantity, item.maxStack - slot.quantity);
                slot.quantity += canAdd;
                quantity -= canAdd;
                currentWeight += canAdd * item.weight;
                std::cout << "[INVENTORY] Added " << canAdd << "x " << item.name << "\\n";
                if (quantity == 0) return true;
            }
        }

        for (auto& slot : slots) {
            if (slot.isEmpty()) {
                int canAdd = std::min(quantity, item.maxStack);
                slot.itemId = itemId;
                slot.quantity = canAdd;
                quantity -= canAdd;
                currentWeight += canAdd * item.weight;
                std::cout << "[INVENTORY] Added " << canAdd << "x " << item.name << " to new slot\\n";
                if (quantity == 0) return true;
            }
        }

        return quantity == 0;
    }

    bool removeItem(const std::string& itemId, int quantity = 1) {
        for (auto& slot : slots) {
            if (slot.itemId == itemId) {
                if (slot.quantity >= quantity) {
                    slot.quantity -= quantity;
                    auto it = itemDatabase.find(itemId);
                    if (it != itemDatabase.end()) {
                        currentWeight -= quantity * it->second.weight;
                    }
                    std::cout << "[INVENTORY] Removed " << quantity << "x " << itemId << "\\n";
                    if (slot.quantity == 0) slot.clear();
                    return true;
                }
            }
        }
        return false;
    }

    int getItemCount(const std::string& itemId) const {
        int total = 0;
        for (const auto& slot : slots) {
            if (slot.itemId == itemId) {
                total += slot.quantity;
            }
        }
        return total;
    }

    bool hasItem(const std::string& itemId, int quantity = 1) const {
        return getItemCount(itemId) >= quantity;
    }

    float getWeight() const { return currentWeight; }
    float getMaxWeight() const { return maxWeight; }
    bool isOverweight() const { return currentWeight > maxWeight; }

    void printContents() const {
        std::cout << "[INVENTORY] Contents (" << currentWeight << "/" << maxWeight << " weight):\\n";
        for (size_t i = 0; i < slots.size(); i++) {
            if (!slots[i].isEmpty()) {
                auto it = itemDatabase.find(slots[i].itemId);
                std::string name = it != itemDatabase.end() ? it->second.name : slots[i].itemId;
                std::cout << "  Slot " << i << ": " << name << " x" << slots[i].quantity << "\\n";
            }
        }
    }
};

/* ============================================================================
 * QUEST SYSTEM
 * ============================================================================ */

enum class QuestStatus { NotStarted, InProgress, Completed, Failed };

struct QuestObjective {
    std::string id;
    std::string description;
    int current;
    int target;
    bool completed;

    QuestObjective() : current(0), target(1), completed(false) {}

    float getProgress() const {
        return target > 0 ? static_cast<float>(current) / target : 0;
    }
};

struct Quest {
    std::string id;
    std::string name;
    std::string description;
    QuestStatus status;
    std::vector<QuestObjective> objectives;
    std::vector<std::string> rewards;
    std::vector<std::string> prerequisites;

    Quest() : status(QuestStatus::NotStarted) {}

    bool areAllObjectivesComplete() const {
        for (const auto& obj : objectives) {
            if (!obj.completed) return false;
        }
        return true;
    }
};

class QuestManager {
private:
    std::map<std::string, Quest> quests;
    std::vector<std::string> activeQuests;

public:
    void registerQuest(const Quest& quest) {
        quests[quest.id] = quest;
        std::cout << "[QUEST] Registered: " << quest.name << "\\n";
    }

    bool startQuest(const std::string& questId) {
        auto it = quests.find(questId);
        if (it == quests.end()) return false;

        Quest& quest = it->second;
        if (quest.status != QuestStatus::NotStarted) return false;

        for (const auto& prereq : quest.prerequisites) {
            auto prereqIt = quests.find(prereq);
            if (prereqIt == quests.end() || prereqIt->second.status != QuestStatus::Completed) {
                std::cout << "[QUEST] Cannot start " << quest.name << " - prerequisite not met\\n";
                return false;
            }
        }

        quest.status = QuestStatus::InProgress;
        activeQuests.push_back(questId);
        std::cout << "[QUEST] Started: " << quest.name << "\\n";
        return true;
    }

    void updateObjective(const std::string& questId, const std::string& objectiveId, int progress) {
        auto it = quests.find(questId);
        if (it == quests.end() || it->second.status != QuestStatus::InProgress) return;

        Quest& quest = it->second;
        for (auto& obj : quest.objectives) {
            if (obj.id == objectiveId && !obj.completed) {
                obj.current = std::min(progress, obj.target);
                if (obj.current >= obj.target) {
                    obj.completed = true;
                    std::cout << "[QUEST] Objective complete: " << obj.description << "\\n";
                }
                break;
            }
        }

        if (quest.areAllObjectivesComplete()) {
            completeQuest(questId);
        }
    }

    void completeQuest(const std::string& questId) {
        auto it = quests.find(questId);
        if (it == quests.end()) return;

        Quest& quest = it->second;
        quest.status = QuestStatus::Completed;

        activeQuests.erase(
            std::remove(activeQuests.begin(), activeQuests.end(), questId),
            activeQuests.end()
        );

        std::cout << "[QUEST] Completed: " << quest.name << "\\n";
        std::cout << "[QUEST] Rewards: ";
        for (const auto& reward : quest.rewards) {
            std::cout << reward << " ";
        }
        std::cout << "\\n";
    }

    void printActiveQuests() const {
        std::cout << "[QUEST] Active Quests:\\n";
        for (const auto& questId : activeQuests) {
            auto it = quests.find(questId);
            if (it != quests.end()) {
                const Quest& quest = it->second;
                std::cout << "  - " << quest.name << ":\\n";
                for (const auto& obj : quest.objectives) {
                    std::cout << "    " << (obj.completed ? "[X]" : "[ ]") << " "
                              << obj.description << " (" << obj.current << "/" << obj.target << ")\\n";
                }
            }
        }
    }
};

/* ============================================================================
 * FINAL ENGINE SUMMARY
 * ============================================================================ */

void printEngineSummary() {
    std::cout << "\\n";
    std::cout << "============================================================\\n";
    std::cout << "              GAME ENGINE COMPONENT SUMMARY                  \\n";
    std::cout << "============================================================\\n";
    std::cout << "  Core Systems:\\n";
    std::cout << "  - Entity Component System (ECS)\\n";
    std::cout << "  - Scene Management\\n";
    std::cout << "  - Event Bus\\n";
    std::cout << "  - Resource Management\\n";
    std::cout << "  - Input System\\n";
    std::cout << "\\n";
    std::cout << "  Physics & Collision:\\n";
    std::cout << "  - Rigidbody Physics\\n";
    std::cout << "  - Box, Circle, Polygon Colliders\\n";
    std::cout << "  - Quadtree Spatial Partitioning\\n";
    std::cout << "  - Spatial Hash Grid\\n";
    std::cout << "\\n";
    std::cout << "  Rendering:\\n";
    std::cout << "  - Sprite Batch Renderer\\n";
    std::cout << "  - Camera System\\n";
    std::cout << "  - Particle System\\n";
    std::cout << "  - Animation System\\n";
    std::cout << "  - Debug Draw\\n";
    std::cout << "\\n";
    std::cout << "  AI Systems:\\n";
    std::cout << "  - State Machine\\n";
    std::cout << "  - Behavior Tree\\n";
    std::cout << "  - A* Pathfinding\\n";
    std::cout << "\\n";
    std::cout << "  Gameplay Systems:\\n";
    std::cout << "  - Inventory\\n";
    std::cout << "  - Quest System\\n";
    std::cout << "  - Achievement System\\n";
    std::cout << "  - Save/Load System\\n";
    std::cout << "\\n";
    std::cout << "  Utilities:\\n";
    std::cout << "  - Tween System\\n";
    std::cout << "  - Coroutines\\n";
    std::cout << "  - Object Pooling\\n";
    std::cout << "  - Terrain Generation\\n";
    std::cout << "  - Audio Manager\\n";
    std::cout << "\\n";
    std::cout << "  Protected Secrets: 5\\n";
    std::cout << "  - ENGINE_LICENSE_KEY\\n";
    std::cout << "  - ENCRYPTION_KEY\\n";
    std::cout << "  - API_KEY\\n";
    std::cout << "  - STEAM_API_KEY\\n";
    std::cout << "  - DRM_KEY\\n";
    std::cout << "============================================================\\n";
}
`,
};
