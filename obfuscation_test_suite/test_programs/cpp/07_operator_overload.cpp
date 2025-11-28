/**
 * Test Program 7: Operator Overloading
 * Purpose: Test operator overloading implementations
 * Expected output: Operator results
 */
#include <iostream>
#include <cmath>

class Complex {
private:
    double real, imag;

public:
    Complex(double r = 0, double i = 0) : real(r), imag(i) {}

    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }

    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }

    Complex operator*(const Complex& other) const {
        double r = real * other.real - imag * other.imag;
        double i = real * other.imag + imag * other.real;
        return Complex(r, i);
    }

    double magnitude() const {
        return std::sqrt(real * real + imag * imag);
    }

    void display() const {
        if (imag >= 0)
            std::cout << real << " + " << imag << "i";
        else
            std::cout << real << " - " << -imag << "i";
    }
};

int main() {
    Complex c1(3, 4);
    Complex c2(1, 2);

    std::cout << "c1 = ";
    c1.display();
    std::cout << ", |c1| = " << c1.magnitude() << std::endl;

    std::cout << "c2 = ";
    c2.display();
    std::cout << ", |c2| = " << c2.magnitude() << std::endl;

    Complex sum = c1 + c2;
    std::cout << "c1 + c2 = ";
    sum.display();
    std::cout << std::endl;

    Complex product = c1 * c2;
    std::cout << "c1 * c2 = ";
    product.display();
    std::cout << std::endl;

    return 0;
}
