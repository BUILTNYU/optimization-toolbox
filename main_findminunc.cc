// Test cases of unconstrained minimization algorithms
// Qi Liu, qiliu.nyu@gmail.com
//
// Six algorithms are tested:
// - Steepest descent (Wolfe condition);
// - Newton direction (possibly with Hessian modification);
// - Trust region;
// - Congjugate gradeint (Fletcher-Reeves);
// - Quasi-Newton method - BFGS;
// - Quasi-Newton method - SR1 with trust region.
//
// These code are faitheful implementations of methods in books (mostely from Nocedal, Wright)
// These code do not represent the most effiient way to implement the methods; sacrifaces are made for clarity;
// For exmaple:
// - cosntants variables should be declared outside of loop
// - matrix operations should have two versions, one allow dest to coincide with source, one does not;
//   or test internally whether they are euqal. I only provided one version that allows dest to be the same with source, 
//   which means internal copy will happen whatsoever.
//
// Usage: \
g++ -std=c++11 main_findminunc.cc \
steepest_descent.cc newton_direction_with_hessian_modification.cc \
trust_region.cc conjugate_gradient_fletcher_reeves.cc quasinewton_bfgs.cc \
quasinewton_sr1.cc line_search_newton_cg.cc \
utilities.cc -I ./eigen-main/src/
#include <iostream>
#include <cmath>
#include "optimization.h"

using namespace std;

// example definiton
double f(double x[]) {
    // test function
    // @arg dim: input dimension
    // @arg x: input array
    return x[0] * x[0] + sin(x[1]);
}

void gradient(double x[], double grad[]) {
    // gradient function is provided algorithms that require it
    grad[0] = 2 * x[0];
    grad[1] = cos(x[1]);
}

void hessian(double x[], double hess[]) {
    // gradient function is provided algorithms that require it
    // Note: hess is an array on heap
    hess[0] = 2;
    hess[1] = 0;
    hess[2] = 0;
    hess[3] = - x[1];
}

// tests
int main(int argc, char **argv) {
    // for test only
    int dim = 2;
    double x0[2] = {0.5, 0.5};
    double x[2];
    double fmin;

    cout << "Unconstrained optimiation" << endl;

    cout << "    *    *    *" << endl;
    cout << "Method 1: steepest descent"  << endl;
    fminunc_steepest_descent(dim, &f, &gradient, x0, x, fmin);
    cout << "optimal x: [" << x[0] << ", " << x[1] << "]" << endl;
    cout << "optimal f: " << fmin << endl << endl;

    cout << "    *    *    *" << endl;
    cout << "Method 2: Newton direction with Hessian modification"  << endl;
    fminunc_newton_direction_with_hessian_modification(dim, f, gradient, hessian, x0, x, fmin);
    cout << "optimal x: [" << x[0] << ", " << x[1] << "]" << endl;
    cout << "optimal f: " << fmin << endl << endl;

    cout << "    *    *    *" << endl;
    cout << "Method 3: trust region"  << endl;
    fminunc_trust_region(dim, f, gradient, hessian, x0, x, fmin);
    cout << "optimal x: [" << x[0] << ", " << x[1] << "]" << endl;
    cout << "optimal f: " << fmin << endl << endl;
    
    cout << "    *    *    *" << endl;
    cout << "Method 4: conjugate gradient (Fletcher-Reeves with restart)"  << endl;
    fminunc_conjugate_gradient_fletcher_reeves(dim, f, gradient, x0, x, fmin);
    cout << "optimal x: [" << x[0] << ", " << x[1] << "]" << endl;
    cout << "optimal f: " << fmin << endl << endl;

    cout << "    *    *    *" << endl;
    cout << "Method 5: Quasi-Newton - BFGS"  << endl;
    fminunc_quasinewton_bfgs(dim, f, gradient, x0, x, fmin);
    cout << "optimal x: [" << x[0] << ", " << x[1] << "]" << endl;
    cout << "optimal f: " << fmin << endl << endl;

    cout << "    *    *    *" << endl;
    cout << "Method 6: Quasi-Newton - SR1 with trust region"  << endl;
    fminunc_quasinewton_sr1(dim, f, gradient, x0, x, fmin);
    cout << "optimal x: [" << x[0] << ", " << x[1] << "]" << endl;
    cout << "optimal f: " << fmin << endl << endl;

    cout << "    *    *    *" << endl;
    cout << "Method 7: line search Newton-CG"  << endl;
    fminunc_line_search_newton_cg(dim, f, gradient, hessian, x0, x, fmin);
    cout << "optimal x: [" << x[0] << ", " << x[1] << "]" << endl;
    cout << "optimal f: " << fmin << endl << endl;
    return 0;


}

