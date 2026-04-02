#ifndef OPTI_HPP
#define OPTI_HPP
#include <functional>
#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

using Mat = MatrixXf;
using Vec = VectorXf;
using Func = function<float(Vec)>;

#define eps 1e-6f

class opti {
private:
    static Vec clamp(const Vec& x, const Vec& l, const Vec& u);
    static Vec project_gradient(const Vec& g, const Vec& x, const Vec& l, const Vec& u);

public :
    static Mat gradient_descent_quasi_update(
        const Mat& H_inv, const Vec& s, const Vec& y);
    static Mat BFGS_quasi_update(
        const Mat& H_inv, const Vec& s, const Vec& y);

    static Mat BFGS_multiply(const Mat& H0_inv, const vector<Vec>& s, const vector<Vec>& y, const Vec& d);

    static Mat hessian(const Func f, const Vec& x, float h = 1e-2f);
    static Vec gradient(const Func f, const Vec& x, float h = 1e-2f);
    static Vec newton_raphson(const Func f, const Vec& x0, int max_iter = 1000, float tol = 1e-4f);
    static Vec quasi_newton(const Func f, const Vec& x0, const Mat& H0_inv, 
                            Mat (*quasi_update) (const Mat&, const Vec&, const Vec&),
                            int max_iter = 1000, float tol = 1e-4f);
    static Vec L_BFGS(const Func f, const Vec& x0, const Mat& H0_inv, int m, int max_iter = 1000, float tol = 1e-4f);
    static Vec L_BFGS_B(const Func f, const Vec& x0, const Mat& H0_inv, int m, const Vec& l, const Vec& u,
                        int max_iter = 1000, float tol = 1e-4f);
};

#endif