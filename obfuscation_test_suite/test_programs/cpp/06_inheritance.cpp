/**
 * Test Program 6: Inheritance and Polymorphism
 * Purpose: Test OOP inheritance and virtual methods
 * Expected output: Polymorphic behavior results
 */
#include <iostream>
#include <string>

class Animal {
protected:
    std::string name;

public:
    Animal(std::string n) : name(n) {}

    virtual void makeSound() {
        std::cout << name << " makes a sound" << std::endl;
    }

    virtual ~Animal() {}
};

class Dog : public Animal {
public:
    Dog(std::string n) : Animal(n) {}

    void makeSound() override {
        std::cout << name << " barks: Woof! Woof!" << std::endl;
    }
};

class Cat : public Animal {
public:
    Cat(std::string n) : Animal(n) {}

    void makeSound() override {
        std::cout << name << " meows: Meow! Meow!" << std::endl;
    }
};

class Bird : public Animal {
public:
    Bird(std::string n) : Animal(n) {}

    void makeSound() override {
        std::cout << name << " chirps: Tweet! Tweet!" << std::endl;
    }
};

int main() {
    Dog dog("Rex");
    Cat cat("Whiskers");
    Bird bird("Tweety");

    // Polymorphic calls
    dog.makeSound();
    cat.makeSound();
    bird.makeSound();

    // Array of pointers
    Animal* animals[] = {&dog, &cat, &bird};
    std::cout << "\nPolymorphic behavior:" << std::endl;
    for (int i = 0; i < 3; i++) {
        animals[i]->makeSound();
    }

    return 0;
}
