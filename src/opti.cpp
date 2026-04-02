#include "opti.hpp"
#include <iostream>

using namespace std;

Vec opti::clamp(const Vec& x, const Vec& l, const Vec& u){
    Vec clamped = x;
    for(int i = 0; i < x.size(); i++){
        clamped[i] = std::clamp(x[i], l[i], u[i]);
    }
    return clamped;
}

Mat opti::gradient_descent_quasi_update(
        const Mat& H_inv, const Vec& s, const Vec& y) {
    return Mat::Identity(H_inv.rows(), H_inv.cols());
}

Mat opti::BFGS_quasi_update(
        const Mat& H_inv, const Vec& s, const Vec& y) {
    float rho = 1.0f / y.dot(s);
    Mat H_inv_new = H_inv;
    Mat I = Mat::Identity(H_inv.rows(), H_inv.cols());
    H_inv_new = (I - rho * y * s.transpose()) * H_inv_new * (I - rho * s * y.transpose()) + rho * s * s.transpose();
    return H_inv_new;
}

Mat opti::BFGS_multiply(const Mat& H0_inv, const vector<Vec>& s, const vector<Vec>& y, const Vec& d) {
    int m = s.size();
    vector<float> alpha(m);
    Vec r = d;
    for(int i = m - 1; i >= 0; i--){
        alpha[i] = s[i].dot(r) / y[i].dot(s[i]);
        r = r - alpha[i] * y[i];
    }
    r = H0_inv * r;
    for(int i = 0; i < m; i++){
        float beta = y[i].dot(r) / y[i].dot(s[i]);
        r = r + s[i] * (alpha[m - 1 - i] - beta);
    }
    return r;
}

Mat opti::hessian(const Func f, const Vec& x, float h){
    int n = x.size();
    Mat H(n, n);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            Vec x_plus_h_i = x;
            Vec x_minus_h_i = x;
            Vec x_plus_h_j = x;
            Vec x_minus_h_j = x;

            x_plus_h_i[i] += h;
            x_minus_h_i[i] -= h;
            x_plus_h_j[j] += h;
            x_minus_h_j[j] -= h;

            float fpp = f(x_plus_h_i + x_plus_h_j);
            float fpm = f(x_plus_h_i + x_minus_h_j);
            float fmp = f(x_minus_h_i + x_plus_h_j);
            float fmm = f(x_minus_h_i + x_minus_h_j);

            H(i,j) = (fpp - fpm - fmp + fmm) / (4 * h * h);
        }
    }
    return H;
}

Vec opti::gradient(const Func f, const Vec& x, float h){
    Vec g(x.size());
    for(int i = 0; i < x.size(); i++){
        Vec x_plus_h = x;
        Vec x_minus_h = x;
        x_plus_h[i] += h;
        x_minus_h[i] -= h;
        g[i] = (f(x_plus_h) - f(x_minus_h)) / (2 * h);
    }
    return g;
}

Vec opti::newton_raphson(const Func f, const Vec& x0, int max_iter, float tol){
    Vec xi = x0;
    for(int i = 0; i < max_iter; i++){
        Vec gi = gradient(f, xi);
        Mat Hi_inv = hessian(f, xi).inverse();
        Vec d = Hi_inv * gi;

        float alpha = 1;
        Vec xi_new = xi - d * alpha;
        for(int j = 0; j < 50; j++){
            if(f(xi_new) < f(xi)){
                break;
            }
            alpha *= 0.5f;
            xi_new = xi - d * alpha;
        }

        if((xi_new - xi).norm() < tol){
            return xi_new;
        }

        xi = xi_new;
    }
    return xi;
}

Vec opti::quasi_newton(const Func f, const Vec& x0, const Mat& H0_inv, 
                           Mat (*quasi_update) (const Mat&, const Vec&, const Vec&),
                           int max_iter, float tol) {
    Vec xi = x0;
    Vec gi = gradient(f, xi);
    Mat Hi_inv = H0_inv;
    for(int i = 0; i < max_iter; i++){
        Vec d = Hi_inv * gi;

        float alpha = 1;
        Vec xi_new = xi - d * alpha;
        for(int j = 0; j < 50; j++){
            if(f(xi_new) < f(xi)){
                break;
            }
            alpha *= 0.5f;
            xi_new = xi - d * alpha;
        }

        if((xi_new - xi).norm() < tol){
            return xi_new;
        }

        Vec gi_new = gradient(f, xi_new);
        Mat Hi_inv = quasi_update(Hi_inv, xi_new - xi, gi_new - gi);
        
        xi = xi_new;
        gi = gi_new;
    }
    return xi;
}

Vec opti::L_BFGS(const Func f, const Vec& x0, const Mat& H0_inv, int m, int max_iter, float tol){
    Vec xi = x0;
    Vec gi = gradient(f, xi);
    Vec d = H0_inv * gi;
    vector<Vec> s_list;
    vector<Vec> y_list;
    for(int i = 0; i < max_iter; i++){

        float alpha = 1;
        Vec xi_new = xi - d * alpha;
        for(int j = 0; j < 50; j++){
            if(f(xi_new) < f(xi)){
                break;
            }
            alpha *= 0.5f;
            xi_new = xi - d * alpha;
        }

        if((xi_new - xi).norm() < tol){
            return xi_new;
        }

        Vec gi_new = gradient(f, xi_new);
        Vec s = xi_new - xi;
        Vec y = gi_new - gi;
        s_list.push_back(s);
        y_list.push_back(y);
        if(s_list.size() > m){
            s_list.erase(s_list.begin());
            y_list.erase(y_list.begin());
        }

        d = BFGS_multiply(H0_inv, s_list, y_list, d);
        
        xi = xi_new;
        gi = gi_new;
    }
    return xi;
}

Vec opti::project_gradient(const Vec& g, const Vec& x, const Vec& l, const Vec& u){
    Vec g_projected = g;
    for(int i = 0; i < g_projected.size(); i++){
        if ((x[i] <= l[i] + eps && g_projected[i] < 0) || (x[i] >= u[i] - eps && g_projected[i] > 0)){
            g_projected[i] = 0;
        }
    }
    return g_projected;
}

Vec opti::L_BFGS_B(const Func f, const Vec& x0, const Mat& H0_inv, int m, const Vec& l, const Vec& u, 
                   int max_iter, float tol) {
    int dim = x0.size();
    vector<Vec> s_list;
    vector<Vec> y_list;

    Vec xi = opti::clamp(x0, l, u);
    Vec gi = gradient(f, xi);
    Vec d = - H0_inv * gi;
    for(int i = 0; i < max_iter; i++){

        if (project_gradient(gi, xi, l, u).lpNorm<Infinity>() < tol){
            cout << "projected " << project_gradient(gi, xi, l, u).lpNorm<Infinity>() << endl;
            return xi;
        }

        d = project_gradient(d, xi, l, u);

        //max step size
        float alpha = 1;
        for(int j = 0; j < dim; j++){
            if (d(j) > 0) alpha = std::min(alpha, (u(j) - xi(j)) / d(j));
            else if (d(j) < 0) alpha = std::min(alpha, (l(j) - xi(j)) / d(j));
        }
        cout << "alpha: " << alpha << endl;

        //backtracking line search
        Vec xi_new = opti::clamp(xi + d * alpha, l, u);
        for(int j = 0; j < 50; j++){
            if(f(xi_new) < f(xi)){
                break;
            }
            alpha *= 0.5f;
            xi_new = opti::clamp(xi + d * alpha, l, u);
        }

        if((xi_new - xi).norm() < tol){
            return xi_new;
        }
        
        Vec gi_new = gradient(f, xi_new);
        Vec s = xi_new - xi;
        Vec y = gi_new - gi;
        s_list.push_back(s);
        y_list.push_back(y);
        if(s_list.size() > m){
            s_list.erase(s_list.begin());
            y_list.erase(y_list.begin());
        }
        d = BFGS_multiply(H0_inv, s_list, y_list, d);
        
        xi = xi_new;
        gi = gi_new;
    }
    return xi;
}