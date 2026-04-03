#include <cstdlib>
#include <iostream>
#include "opti.hpp"

using namespace std;

int main(){
    cout << "Hello, World!" << endl;

    Func f = [](const Vec& x) {
        return x(0) * x(0) * sin(x(0)) + x(1) * x(1);
    };
    Vec x0(2);
    x0 << 0.3f, 0.50f;
    Mat H0_inv = opti::hessian(f, x0).inverse();
    Vec result = opti::L_BFGS_B(f, x0, H0_inv, 5, Vector2f(0.0f, 0.05f), Vector2f(0.75f, 0.1f));
    cout << "Minimum at: " << result(0) << ", " << result(1) << endl;

    return EXIT_SUCCESS;
}