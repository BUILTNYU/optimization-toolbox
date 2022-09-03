// unconstrained optimization, conjugate gradient - Fletcher-Reeves
// Qi Liu, qiliu.nyu@gmail.com
//
// Nocedal, Wright, p121
// with restart every dim iterations, Nocedal, p127
// line search satisfy Wolfe condtion with 0 < c1 < c2 < 1/2, Nocedal, p125
#include <iostream>
#include <cmath>
#include "utilities.h"
using namespace std;

// funciton prototypes

// function definitions
void fminunc_conjugate_gradient_fletcher_reeves(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                double x0[],
                                double x[],
                                double &fmin) {
    /**
     * @brief find the mininum of uncosntrained function using conjugate gradient method (Fletcher-Reevers)
     * @arg dim: dimension
     * @arg gradient: gradient function
     * @arg x0: stating point
     * @arg x: output result x
     * @arg fmin: optimal obj
     */

    copy(dim, x0, x);  // initialize x
    int max_itr_x = 100000;
    double threshold_x = 0.0001; // threshold for stopping x iter
    double grad[dim];
    double grad_next[dim]; // grad_k+1
    double p[dim];  // search direction
    double p_next[dim]; // p_k+1

    // initialize grad and p
    (*gradient)(x, grad);
    copy_and_negate(dim, grad, p); // initialize p0

    int i_x = 1; // start with iteration 1
    while (true) {
        // check terminating condition - grad at x close to 0
        if (l2norm(dim, grad) < threshold_x)
            break;
        if (i_x == max_itr_x) {
            cerr << "Not converging! Error code: 01" << endl;
            exit(1);
        } 

        // at x_k, find step size - alpha*
        // alpha related params
        double c1 = 0.0001; // c1 setting, cf p 33, "loose line search" settings, p62
        double c2 = 0.4;  // < 1/2 to ensure direction descending
        double alpha_max = 8; // max step size
        double alpha_star = line_search_wolfe_condiditons(dim, f, gradient, x, p, c1, c2, alpha_max);
        
        // update x_k to x_k+1
        x_plus_c_multi_y(dim, x, alpha_star, p, x);
        // evaluate f
        fmin = (*f)(x);
        // find gradient, grad_k+1
        (*gradient)(x, grad_next);

        // find new search direction - p_next
        if (i_x % dim == 0) {
            // reinitialize p to be neg gradient every dim times of iteration
            copy_and_negate(dim, grad_next, p_next); // initialize p
        } else {
            // use (5.41) to find next p
            double beta = inner_product(dim, grad_next, grad_next) / inner_product(dim, grad, grad);
            a_x_plus_b_y(dim, -1, grad_next, beta, p, p_next);
        }

        copy(dim, grad_next, grad);
        copy(dim, p_next, p);
        ++i_x;
    }
}
