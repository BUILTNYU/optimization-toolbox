// unconstrained optimization, steepest descent method
// Qi Liu, qiliu.nyu@gmail.com
//
// Wolfe condtion is used for line search methods
// Cf. Nocedal, Wright, p33
// The way to find step size that satisfies Wolfe condition: Nocedal, p60
#include <iostream>
#include <cmath>
#include "utilities.h"
using namespace std;

// function prototypes
                  
// function definitions
void fminunc_steepest_descent(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                double x0[],
                                double x[],
                                double &fmin) {
    /**
     * @brief 
     * @arg dim: dimension of input
     * @arg f: objective function
     * @arg gradent: gradient function
     * @arg x0: starting point
     * @arg x: return optimal x
     * @arg fmin: return minimum f
     */
    copy(dim, x0, x); // initlize x
    double grad[dim];  // store gradient at x
    double direct[dim]; // store moving direction at x - negtive gradient for steepest descent method
    double threshold_x = 0.0001; // threshold for stopping x iter
    int max_itr_x = 100000; // max num of x iterations

    // start iteration on x
    int i_x = 1;
    while (true) {   
        // evaluate f
        fmin = (*f)(x);

        // find gradient, grad
        (*gradient)(x, grad);
        // update search direction - negative gradient, direct
        copy_and_negate(dim, grad, direct);
        
        // check terminating condition - grad at x close to 0
        if (l2norm(dim, grad) < threshold_x)
            break;

        if (i_x == max_itr_x) {
            cerr << "Not converging! Error code: 01" << endl;
            exit(1);
        } 

        // find step size - alpha*
        double c1 = 0.0001; // c1 setting, cf p 33, "loose line search" settings, p62
        double c2 = 0.9;  // "loose line search" settings, p62
        double alpha_max = 4; // max step size
        double alpha_star = line_search_wolfe_condiditons(dim, f, gradient, x, direct, c1, c2, alpha_max);
        
        // update x
        x_plus_c_multi_y(dim, x, alpha_star, direct, x);
        ++i_x;
    }
}


