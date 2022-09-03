// unconstrained optimization, trust region method
// Qi Liu, qiliu.nyu@gmail.com
//
// Algorith 4.1 Trust region, Cf. Nocedal, p69
// Solving trust region subproblem (acurate method), Cf. Nocedal, p87
// for other methods to (approximately) solve the subproblem like dog-leg, two-dimensinal searching, cf. p71
#include <iostream>
#include <cmath>
#include "utilities.h"
using namespace std;

// function prototypes

// function definitions
void fminunc_trust_region(int dim,
                            double (*f)(double x[]),
                            void (*gradient)(double x[], double grad[]),
                            void (*hessian)(double x[], double hess[]),
                            double x0[],
                            double x[],
                            double &fmin) {
    /**
     * @brief Construct a new copy object
     * @arg dim: dimension
     * @arg gradient: gradient function
     * @arg hessian: hession function Note: hess is stored as a array
     * @arg x0: stating point
     * @arg x: output result x
     * @arg fmin: optimal obj
     */

    double DELTA_MAX = 4.0; // trust-region radis max
    double delta = 1.0; // trust-region radius
    double rho; // ratio
    double eta = 0.2; // threshold for rho; within [0, 1/4)
    double p[dim];  // next step
    double hess[dim*dim];  // hessian, "B"
    double grad[dim];  // gradient, "g"
    int max_itr_x = 100000;  // max number of iterations of x
    double threshold_x = 0.0001; // threshold for stopping x iter
    double on_boundary_threshold = 0.01; // tell whether p is on boundary
    copy(dim, x0, x);  // initialize x

    int i_x = 1; // iteration count
    while (true) {
        // evaluate f
        fmin = (*f)(x);

        // find gradient, grad
        (*gradient)(x, grad);
        
        // check terminating condition - grad at x close to 0
        if (l2norm(dim, grad) < threshold_x)
            break;
        
        if (i_x == max_itr_x) {
            cerr << "Not converging! Error code: 01" << endl;
            exit(1);
        }

        // find hessian
        // stored in hess as a vector of dimension dim*dim
        (*hessian)(x, hess);

        // solve subproblem
        trust_region_subproblem(dim, hess, grad, delta, p);

        // compute quadratic approx and fs' matching degree
        rho = (fmin - phi(dim, f, x, p, 1)) / 
            (fmin - quadratic_approximation(dim, hess, grad, fmin, p));

        if (rho < eta)
            delta = delta / 4;
        else
            if (rho > 0.75 && abs(l2norm(dim, p) - delta) < on_boundary_threshold * delta)
                //            on the boundary
                //            don't require doubles to be exactly equal
                delta = min(2*delta, DELTA_MAX);

        // update x
        if (rho > eta)
            x_plus_c_multi_y(dim, x, 1, p, x);

        ++i_x;
    }
}