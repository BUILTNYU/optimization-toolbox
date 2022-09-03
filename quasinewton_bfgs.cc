// unconstrained optimization, Quasi-Newton - BFGS method
// Qi Liu, qiliu.nyu@gmail.com
//
// BFGS algorithm: Nocedal, Wright, p140
// Hess inverse (Hinv) initialize: p142
// line search start with alpha = 1, p142
#include <iostream>
#include <cmath>
#include "utilities.h"
using namespace std;

// funciton prototypes
void bfgs_hess_inv_update(int dim, double *Hinv, double *s, double *y, double *Hinv_next);

// function definitions
void fminunc_quasinewton_bfgs(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                double x0[],
                                double x[],
                                double &fmin) {
    /**
     * @brief find the mininum of uncosntrained function using quasi-Newton method - BFGS
     * @arg dim: dimension
     * @arg gradient: gradient function
     * @arg x0: stating point
     * @arg x: output result x
     * @arg fmin: optimal obj
     * H initialized to be identity
     */

    copy(dim, x0, x);  // initialize x
    int max_itr_x = 100000;
    double x_next[dim];
    double threshold_x = 0.0001; // threshold for stopping x iter
    double grad[dim];
    (*gradient)(x, grad); // initialize grad
    double grad_next[dim]; // grad_k+1
    double p[dim];  // search direction, to be computed later

    // Initialize Hessian inverse to identity matrix
    double Hinv[dim * dim];
    identity(dim, Hinv);
    double Hinv_next[dim * dim];

    // vectors s, k
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

        // compute search direction, p = - H * grad
        matrix_multiplication(dim, dim, 1, Hinv, grad, p);
        copy_and_negate(dim, p, p);
        
        // line search
        double c1 = 0.0001; // c1 setting, cf p 33, "loose line search" settings, p62
        double c2 = 0.9;  // "loose line search" settings, p62
        double alpha_max = 4; // max step size
        double alpha = line_search_wolfe_condiditons(dim, f, gradient, x, p, c1, c2, alpha_max, 1);
        // start with alpha = 1, Cf. p142
        x_plus_c_multi_y(dim, x, alpha, p, x_next); // compute x_next
        (*gradient)(x_next, grad_next); // compute grad_next
        
        // define s, y
        x_plus_c_multi_y(dim, x_next, -1, x, s);
        x_plus_c_multi_y(dim, grad_next, -1, grad, y);

        // reset H0
        // Cf. Nocedal, Wright, p143 (6.20)
        if (i_x == 0) {
            double tmp = inner_product(dim, y, s) / inner_product(dim, y, y);
            scaler_multiplication(dim * dim, tmp, Hinv, Hinv);
        }

        // compute Hinv_next - BFGS update, p140 (6.17)
        bfgs_hess_inv_update(dim, Hinv, s, y, Hinv_next);
        
        // update and iterate
        copy(dim, x_next, x);
        copy(dim, grad_next, grad);
        copy(dim * dim, Hinv_next, Hinv);
        ++i_x;
    }
}

void bfgs_hess_inv_update(int dim, double *Hinv, double *s, double *y, double *Hinv_next) {
    // compute rho
    double rho = 1 / inner_product(dim, y, s);
    // compute y * s^T
    double y_sT[dim * dim];
    matrix_multiplication(dim, 1, dim, y, s, y_sT);
    // compute s * yT
    double s_yT[dim * dim];
    matrix_multiplication(dim, 1, dim, s, y, s_yT);
    // compute I - rho * ys^T
    double id[dim * dim];
    identity(dim, id);
    double I_minus_rho_y_sT[dim * dim];
    x_plus_c_multi_y(dim * dim, id, (-1) * rho, y_sT, I_minus_rho_y_sT);
    // compute I - rho * sy^T
    double I_minus_rho_s_yT[dim * dim];
    x_plus_c_multi_y(dim * dim, id, (-1) * rho, s_yT, I_minus_rho_s_yT);
    // compute term1
    double term1[dim * dim];
    matrix_multiplication(dim, dim, dim, Hinv, I_minus_rho_y_sT, term1);
    matrix_multiplication(dim, dim, dim, I_minus_rho_s_yT, term1, term1);
    // compute term2
    double term2[dim * dim];
    matrix_multiplication(dim, 1, dim, s, s, term2);
    scaler_multiplication(dim * dim, rho, term2, term2);
    // compute Hinv_next
    x_plus_c_multi_y(dim * dim, term1, 1, term2, Hinv_next);
}