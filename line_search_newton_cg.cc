// unconstrained optimization, line search Netwon conjugate gradient method
// inexact Netwon method
// Qi Liu, qiliu.nyu@gmail.com
//
// Gradient, Hessian are needed
// Hessian^(-1) * gradient is approximatedly solved by using conjugate gradient (CG) method
// Nocedal, Wright, p169
// backtracking line search, p37
#include <iostream>
#include <cmath>
#include "utilities.h"
using namespace std;

// funciton prototypes
void  solve_Bz_equal_minus_g_approximately(int dim, double *B, double *grad, double *direct);

// function definitions
void fminunc_line_search_newton_cg(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                void (*hessian)(double x[], double hess[]),
                                double x0[],
                                double x[],
                                double &fmin) {
    /**
     * @brief find the mininum of uncosntrained function using newton direction possibly with hessian modifications
     * @arg dim: dimension
     * @arg gradient: gradient function
     * @arg hessian: hession function Note: hess is stored as a array
     * @arg x0: stating point
     * @arg x: output result x
     * @arg fmin: optimal obj
     */
    copy(dim, x0, x);  // initialize x
    int max_itr_x = 20;
    double threshold_x = 0.0001; // threshold for stopping x iter
    double grad[dim];
    double neg_grad[dim];  // -grad
    double direct[dim];  // search direction
    double alpha; // step size
    double c = 0.001;  // descending requirement, namely c1 in Wolfe condition
    double hess[dim * dim];

    int i_x = 1;
    while (true) {
        // evaluate f
        fmin = (*f)(x);

        // find gradient, grad
        (*gradient)(x, grad);
        copy_and_negate(dim, grad, neg_grad);
        
        // check terminating condition - grad at x close to 0
        if (l2norm(dim, grad) < threshold_x)
            return;
        
        if (i_x == max_itr_x) {
            cerr << "x not converging!" << endl;
            exit(1);
        }

        // find hessian
        // stored in hess as a vector of dimension dim*dim
        (*hessian)(x, hess);

        // solve Bp = -gradf approximately
        // result stored in direct, vector of dimension dim
        solve_Bz_equal_minus_g_approximately(dim, hess, grad, direct);

        // find step size
        alpha = backtracking(dim, f, x, direct, grad, c);

        // update x
        x_plus_c_multi_y(dim, x, alpha, direct, x);
        ++i_x;
    }
}

void  solve_Bz_equal_minus_g_approximately(int dim, double *B, double *grad, double *z) {
    // conjugate gradient method is used to solve Bz = -g approximately
    // Nocedal, Wright, p169
    double epsilon = min(0.5, sqrt(l2norm(dim, grad))) / l2norm(dim, grad); // tolerance for residual r
    // z is solution point
    zero_vector(dim, z); // solution initialized to be zero
    double r[dim]; // residual
    double r_next[dim];
    copy(dim, grad, r);
    double d[dim]; // searh direction
    copy_and_negate(dim, grad, d);
    double d_next[dim];
    double threshold_z = 0.0001; // threshold for stopping z iter

    int i_z = 0; // start with iteration 1
    // variables for temp use
    double B_d[dim];
    double dT_B_d;
    double alpha;
    double beta;
    while (true) {
        // check terminating condition - negative curvature
        matrix_multiplication(dim, dim, 1, B, d, B_d);
        dT_B_d = inner_product(dim, d, B_d);
        if (dT_B_d <= 0) {
            if (i_z == 0) {
                copy_and_negate(dim, grad, z);
                return;
            } else {
                return;
            }
        }
        if (i_z == dim) {
            return; // Note: z must converge within dim iterations; Just return if reached.
        }

        // at z_k, find step size - alpha
        alpha = inner_product(dim, r, r) / dT_B_d;
        
        // update z_k to z_k+1
        x_plus_c_multi_y(dim, z, alpha, d, z);
        
        // r_next
        x_plus_c_multi_y(dim, r, alpha, B_d, r_next);
        
        // check terminating condition - residual
        if (l2norm(dim, r_next) < epsilon) {
            return;
        }

        // find beta
        beta = inner_product(dim, r_next, r_next) / inner_product(dim, r, r);

        // d_next
        a_x_plus_b_y(dim, -1, r_next, beta, d, d_next);

        // update variables for next iteration
        copy(dim, r_next, r);
        copy(dim, d_next, d);
        ++i_z;
    }
}
