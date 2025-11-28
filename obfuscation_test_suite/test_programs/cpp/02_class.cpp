/**
 * Test Program 2: Class Definition and Methods
 * Purpose: Test class obfuscation and method calls
 * Expected output: Class method results
 */
#include <iostream>
#include <string>

class Student {
private:
    std::string name;
    int id;
    double gpa;

public:
    Student(std::string n, int i, double g) : name(n), id(i), gpa(g) {}

    void display() {
        std::cout << "Name: " << name << ", ID: " << id << ", GPA: " << gpa << std::endl;
    }

    double getGPA() const {
        return gpa;
    }

    void updateGPA(double newGPA) {
        gpa = newGPA;
    }
};

int main() {
    Student s1("Alice", 101, 3.8);
    Student s2("Bob", 102, 3.5);
    Student s3("Charlie", 103, 3.9);

    s1.display();
    s2.display();
    s3.display();

    s2.updateGPA(3.7);
    s2.display();

    return 0;
}
