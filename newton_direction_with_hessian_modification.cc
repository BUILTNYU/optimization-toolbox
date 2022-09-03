// unconstrained optimization, Newton direction with Hessian modification
// Qi Liu, qiliu.nyu@gmail.com
//
// Gradient, Hessian are needed.
// Nocedal, Wright, algorithm 3.2, p48
// Cholesky with added multiple of the identity, p51
// backtracking line search, p37
#include <iostream>
#include <cmath>
#include "utilities.h"
using namespace std;

// funciton prototypes
void cholesky_with_added_multiple_of_the_identity(int dim, double hess[], double L[]);

// function definitions
void fminunc_newton_direction_with_hessian_modification(int dim,
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
    int max_itr_x = 100000;
    double threshold_x = 0.0001; // threshold for stopping x iter
    double grad[dim];
    double neg_grad[dim];  // -grad
    double direct[dim];  // search direction
    double alpha; // step size
    double c = 0.001;  // descending requirement, namely c1 in Wolfe condition
    double hess[dim * dim];
    double L[dim*dim];  // vector to store cholesky decomp resul matrix

    int i = 1;
    while (true) {
        // evaluate f
        fmin = (*f)(x);

        // find gradient, grad
        (*gradient)(x, grad);
        copy_and_negate(dim, grad, neg_grad);
        
        // check terminating condition - grad at x close to 0
        if (l2norm(dim, grad) < threshold_x)
            break;
        
        if (i == max_itr_x) {
            cerr << "Not converging!";
            exit(1);
        }

        // find hessian
        // stored in hess as a vector of dimension dim*dim
        (*hessian)(x, hess);

        // Cholesky with added multiple of the identity, and find the direct
        // B = LL'
        // result stored in L
        cholesky_with_added_multiple_of_the_identity(dim, hess, L);

        // solve Bp = -gradf
        // result stored in direct, vector of dimension dim
        solve_linear_equations(dim, L, neg_grad, direct);

        // find step size
        alpha = backtracking(dim, f, x, direct, grad, c);

        // update x
        x_plus_c_multi_y(dim, x, alpha, direct, x);
        ++i;
    }
}

void cholesky_with_added_multiple_of_the_identity(int dim, double hess[], double L[]) {
    // p51
    // multiples of id will be added to hess;
    // direct is the result of search direction
    double beta = 0.001;  // p52 setting
    double tau;
    double hess_modified[dim*dim];

    // find the min elem of hess diagonal
    double min_diag_hess = 1000000;
    for (int i = 0; i < dim; ++i)
        if (hess[dim * i + i] < min_diag_hess)
            min_diag_hess = hess[dim * i + i];

    if (min_diag_hess > 0) {
        tau = 0;
    } else {
        tau = (-min_diag_hess) + beta;
    }

    while(true) {
        add_identity(dim, hess, hess_modified, tau);
        if (cholesky_decomposition(dim, hess_modified, L))
            return;
        else {
            tau = max(2*tau, beta);
        }
    }
}
