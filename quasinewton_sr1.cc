// unconstrained optimization, Quasi-Newton - SR1 method
// Qi Liu, qiliu.nyu@gmail.com
//
// Algorithm 6.2 (SR1 Trust-region method), Nocedal, Wright, p146
// Hess inverse (Hinv) initialize: p142
#include <iostream>
#include <cmath>
#include "utilities.h"
using namespace std;

// function prototypes

// function definitions
void fminunc_quasinewton_sr1(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                double x0[],
                                double x[],
                                double &fmin) {
    /**
     * @brief find the mininum of uncosntrained function using quasi-Newton method - SR1 Trust-region
     * @arg dim: dimension
     * @arg gradient: gradient function
     * @arg x0: stating point
     * @arg x: output result x
     * @arg fmin: optimal obj
     * H initialized to be identity
     */

    double DELTA_MAX = 4.0; // trust-region radis max
    double delta = 1.0; // trust-region radius
    double on_boundary_threshold = 0.01; // tell whether p is on boundary
    double rho; // ratio
    double eta = 0.2; // threshold for rho; within [0, 1/4)

    copy(dim, x0, x);  // initialize x
    int max_itr_x = 100000;
    double x_next[dim];
    double x_next_if_updated[dim];
    double threshold_x = 0.0001; // threshold for stopping x iter
    double grad[dim];
    (*gradient)(x, grad); // initialize grad
    double grad_next[dim]; // grad_k+1
    double grad_next_if_updated[dim]; // grad_k+1

    // Initialize Hessian to identity matrix
    double B[dim * dim];
    identity(dim, B);
    double B_next[dim * dim];

    // vectors s, k
    // even x not updated, we still have s and k updated
    double s[dim], y[dim];

    // start iteration
    int i_x = 0;
    while (true) {
        // check terminating condition - grad at x close to 0
        if (l2norm(dim, grad) < threshold_x)
            break;
        if (i_x == max_itr_x) {
            cerr << "Not converging! Error code: 01" << endl;
            exit(1);
        }

        // solve trust region subproblem, find s, y, x_next
        // Note: even we dont change x, we still have s and y to update B
        trust_region_subproblem(dim, B, grad, delta, s);

        // compute quadratic approx and fs' matching degree
        rho = (fmin - phi(dim, f, x, s, 1)) / 
            (fmin - quadratic_approximation(dim, B, grad, fmin, s));

        if (rho < eta)
            delta = delta / 2;
        else
            if (rho > 0.75 && abs(l2norm(dim, s) - delta) < on_boundary_threshold * delta)
                //            on the boundary
                //            don't require doubles to be exactly equal
                delta = min(2*delta, DELTA_MAX);

        // compute x_next_if_updated even if x not updated; it's used to update B
        x_plus_c_multi_y(dim, x, 1, s, x_next_if_updated);
        (*gradient)(x_next_if_updated, grad_next_if_updated);
        // update x if needed
        if (rho > eta) {
            copy(dim, x_next_if_updated, x_next);
            copy(dim, grad_next_if_updated, grad_next);
        } else {
            copy(dim, x, x_next);
            copy(dim, grad, grad_next);
        }

        // define y
        x_plus_c_multi_y(dim, grad_next_if_updated, -1, grad, y);

        // Hessian update - SR1 (6.24), p144
        double Bs[dim];
        matrix_multiplication(dim, dim, 1, B, s, Bs);
        double y_minus_Bs[dim];
        x_plus_c_multi_y(dim, y, -1, Bs, y_minus_Bs);
        double r = abs(inner_product(dim, s, y_minus_Bs)) / l2norm(dim, s) / l2norm(dim, y_minus_Bs);
        // check condition (6.26), p145
        if (r >= 1.0e-4) {
            // use (6.24) to update
            double tmp_matrix[dim * dim];
            matrix_multiplication(dim, 1, dim, y_minus_Bs, y_minus_Bs, tmp_matrix);
            double tmp_denominator = inner_product(dim, y_minus_Bs, s);
            x_plus_c_multi_y(dim * dim, B, 1/tmp_denominator, tmp_matrix, B_next);
        } // else B no change

        // update and iterate
        copy(dim, x_next, x);
        copy(dim, grad_next, grad);
        copy(dim * dim, B_next, B);
        // update f
        fmin = (*f)(x);
        ++i_x;
    }
}