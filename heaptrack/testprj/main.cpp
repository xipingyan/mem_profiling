#include <string>
#include <iostream>
#include <thread>
#include <vector>

void fun1() {
    std::cout << "fun1: std::vector<int> vec(1024);" << std::endl;
    std::vector<int> vec(1024);
    vec[1023] = 20;
    std::cout << "vec[1023] = "<< vec[1023] << std::endl;
}

float* fun2() {
    std::cout << "fun2: float *p = new float[512]; return p;" << std::endl;
    float *p = new float[512];
    return p;
}

void fun3(float* p) {
    std::cout << "fun3: delete[] p;" << std::endl;
    if (p) {
        delete[] p;
        p = nullptr;
    }
}

void fun4(float* p) {
    std::cout << "fun4: delete p;" << std::endl;
    if (p) {
        delete p;
        p = nullptr;
    }
}

void fun5() {
    std::cout << "fun5: float *p = new float[512]; delete[] p;" << std::endl;
    float *p = new float();
    float *p1 = new float();
    p[0] = 1024;
    std::cout << "  p[0]=" << p[0] << std::endl;
    delete p;
}

int main(int argc, char** argv) {
    std::cout << "Start test." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    fun1();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    float* p = fun2();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    fun3(p);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // fun4(p);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    fun5();
    return 0;
}