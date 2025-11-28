/**
 * Test Program 10: Struct Operations
 * Purpose: Test struct handling and member access
 * Expected output: Struct member operations
 */
#include <stdio.h>
#include <string.h>

struct Person {
    char name[50];
    int age;
    float height;
    char email[100];
};

int main() {
    struct Person people[3];
    int i;

    // Initialize people
    strcpy(people[0].name, "Alice Johnson");
    people[0].age = 28;
    people[0].height = 5.6;
    strcpy(people[0].email, "alice@example.com");

    strcpy(people[1].name, "Bob Smith");
    people[1].age = 35;
    people[1].height = 6.1;
    strcpy(people[1].email, "bob@example.com");

    strcpy(people[2].name, "Charlie Brown");
    people[2].age = 22;
    people[2].height = 5.9;
    strcpy(people[2].email, "charlie@example.com");

    // Process and output
    for (i = 0; i < 3; i++) {
        printf("Name: %s\n", people[i].name);
        printf("Age: %d\n", people[i].age);
        printf("Height: %.2f\n", people[i].height);
        printf("Email: %s\n", people[i].email);
        printf("---\n");
    }

    return 0;
}
