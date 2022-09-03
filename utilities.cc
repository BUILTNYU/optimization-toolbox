#include <cmath>
#include <iostream>
#include <Eigen/Dense> // for solving trust region subproblem
using namespace std;
using namespace Eigen;

void copy(int dim, double source[], double dest[]) {
    /**
     * @brief copy from source to dest
     */
    for (int i = 0; i < dim; ++i)
        dest[i] = source[i];
}

void copy_and_negate(int dim, double source[], double dest[]) {
    /**
     * @brief copy and negate
     * dest and source can be the same
     */
    for (int i = 0; i < dim; ++i)
        dest[i] = (-1) * source[i];
}

void zero_vector(int dim, double *v) {
    for (int i = 0; i < dim; ++i)
        v[i] = 0;
}

double l2norm(int dim, double v[]) {
    /**
     * @brief L2-norm of vector v with dimension dim
     */
    double ans = 0;

    for (int i = 0; i < dim; ++i) {
        ans += pow(v[i], 2);
    }
    return sqrt(ans);
}

double inner_product(int dim, double x[], double y[]) {
    double ans = 0;

    for (int i = 0; i < dim; ++i) {
        ans += x[i] * y[i];
    }
    return ans;
}

void x_plus_c_multi_y(int dim, double x[], double c, double y[], double output[]) {
    /**
     * @brief compute x + cy and store in ans;
     * x and y are vectors of dimensiom dim; c is a scalar; 
     * output can conincide with x or y;
     */
    for (int i = 0; i < dim; ++i)
        output[i] = x[i] + c * y[i];   
}

void a_x_plus_b_y(int dim, double a, double x[], double b, double y[], double output[]) {
    /**
     * @brief compute ax + by and store in ans;
     * x and y are vectors of dimensiom dim; a, b are scalars; 
     * output can conincide with x or y;
     */
    for (int i = 0; i < dim; ++i)
        output[i] = a * x[i] + b * y[i];   
}

void scaler_multiplication(int dim, double c, double x[], double output[]) {
    /**
     * @brief compute cx and store in output;
     * x is vector of dimensiom dim; c is a scalar; 
     * output can conincide with x;
     */
    for (int i = 0; i < dim; ++i)
        output[i] = c * x[i];   
}

double phi(int dim,
            double (*f)(double x[]),
            double x[],
            double direct[],
            double alpha) {
    double x_alpha[dim];

    x_plus_c_multi_y(dim, x, alpha, direct, x_alpha);
    return (*f)(x_alpha);
}

double phi_derivative(int dim,
                        void (*gradient)(double x[], double grad[2]),
                        double x[],
                        double direct[],
                        double alpha) {
    double x_alpha[dim];
    double grad_x_alpha[dim];

    x_plus_c_multi_y(dim, x, alpha, direct, x_alpha);
    (*gradient)(x_alpha, grad_x_alpha);
    return inner_product(dim, direct, grad_x_alpha);
}

void matrix_multiplication(int m, int l, int n, double A[], double B[], double output[]) {
    /**
     * @brief 
     * A is a matrix of size mxl; B is a matrix of size lxn; both represented as vector
     * output is a matrix of size mxn
     * matrices are represented as vectors; M[0,1], M[0,2], ..., M[1,0], ...
     * output can be the same matrix as input
     */
    int i, j, k;
    double tmp_sum;
    double ans[m * n];

    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            tmp_sum = 0;
            for (k = 0; k < l; ++k) {
                tmp_sum += A[l * i + k] * B[n * k + j];
            }
            ans[n * i + j] = tmp_sum;
        }
    }
    copy(m * n, ans, output);
}

double quadratic_approximation(int dim, double B[], double g[], double fx, double p[]) {
    double x_alpha[dim];

    // m(p) = fx + g'p + 1/2*p'Bp
    double ans = fx;
    ans += inner_product(dim, g, p);
    double tmp[dim];
    matrix_multiplication(1, dim, dim, p, B, tmp);
    ans += 0.5 * inner_product(dim, tmp, p);
    return ans;
}

bool cholesky_decomposition(int dim, double A[], double L[]) {
    // Cholesky–Crout algorithm is used for Cholesky decomposition; Cf. wikipedia
    // A = LL*
    int i, j, k;

    for (j = 0; j < dim; ++j) {
        float sum = 0;
        for (k = 0; k < j; ++k) {
            sum += L[dim * j + k] * L[dim * j + k];
        }
        if (A[dim * j + j] - sum < 0)
            return false;  // in order to be sqrted, it needs to be positive
        L[dim * j + j] = sqrt(A[dim * j + j] - sum);

        for (i = j + 1; i < dim; ++i) {
            sum = 0;
            for (k = 0; k < j; ++k) {
                sum += L[dim * i + k] * L[dim * j + k];
            }
            if (L[dim * j + j] == 0)
                return false;  // in order to be divided, it needs to be nonzero
            L[dim * i + j] = (1.0 / L[dim * j + j] * (A[dim * i + j] - sum));
        }
    }

    for (i = 0; i < dim; ++i)
        for (j = i + 1; j < dim; ++j)
            L[dim * i + j] = 0.0;
    return true;
}

void transpose(int m, int n, double *source, double *dest) {
    /**
     * @brief transpose a matrix of mxn, stored in dest of n*m
     * if m = n, dest can be the same with source
     */
    double ans[n * m];

    for (int i = 0; i < n; ++i) 
        for (int j = 0; j < m; ++j)
            ans[m * i + j] = source[n * j + i];
    
    copy(m * n, ans, dest);
}

void transpose(int dim, double source[], double dest[]) {
    // transpose square matrix of dimension dim * dim
    // dest can be the same with source
    double tmp[dim * dim];
    int i, j;

    for (i = 0; i < dim; ++i)
        for (j = 0; j < dim; ++j)
            tmp[dim * i + j] = source[dim * j + i];
    for (i = 0; i < dim; ++i)
        for (j = 0; j < dim; ++j)
            dest[dim * i + j] = tmp[dim * i + j];
}

void solve_linear_equations(int dim, double L[], double b[], double x[]) {
    // solve Ax = b
    // A = LL'
    // L(L'x) = b; let y:=L'x
    // Ly = b; solve y first
    double y[dim];
    int i, j;
    double sum;

    for (i = 0; i < dim; ++i) {
        sum = 0;
        for (j = 0; j < i; ++j)
            sum += L[dim * i + j] * y[j];
        
        y[i] = (b[i] - sum) / L[dim * i + i];
    }

    // then solve L'x = y
    for (i = dim - 1; i >= 0; --i) {
        sum = 0;
        for (j = i + 1; j < dim; ++j)
            sum += L[dim * j + i] * x[i];

        x[i] = (y[i] - sum) / L[dim * i + i];
    }

}

void solve_lower_triangular_linear_equations(int dim, double L[], double b[], double x[]) {
    // solve Lx = b
    int i, j;
    double sum;

    for (i = 0; i < dim; ++i) {
        sum = 0;
        for (j = 0; j < i; ++j)
            sum += L[dim * i + j] * x[j];
        
        x[i] = (b[i] - sum) / L[dim * i + i];
    }
}

void add_identity(int dim, double source[], double dest[], double tau) {
    /**
     * @brief add tau multiples of id matrix to source matrix, then store in dest
     * 
     */
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (i == j)
                dest[dim * i + j] = source[dim * i + j] + tau;
            else
                dest[dim * i + j] = source[dim * i + j];
        }
    }
}

double zoom(int dim,
            double (*f)(double x[]),
            void (*gradient)(double x[], double grad[2]),
            double x[],
            double direct[],
            double c1,
            double c2,
            double alpha_lo,
            double alpha_hi) {
    // Nocedal, p61
    double alpha;
    double phi_alpha;
    double phi_alpha_0 = phi(dim, f, x, direct, 0);
    double phi_deriv_alpha_0 = phi_derivative(dim, gradient, x, direct, 0);
    double phi_alpha_lo;
    double phi_deriv_alpha;
    int max_itr = 1000;

    // start iteration
    int i = 0;
    while (true) {
        
        if (i == max_itr) {
            cerr << "Zoom fails! Error code: 03" << endl;
            exit(3);
        }

        phi_alpha_lo = phi(dim, f, x, direct, alpha_lo);
        // interpolate
        alpha = (alpha_lo + alpha_hi) / 2;  // bisection
        // evaluate phi_alpha
        phi_alpha = phi(dim, f, x, direct, alpha);
        if (phi_alpha > (phi_alpha_0 + c1 * alpha * phi_deriv_alpha_0) || phi_alpha >= phi_alpha_lo) {
            alpha_hi = alpha;
        } else {
            phi_deriv_alpha = phi_derivative(dim, gradient, x, direct, alpha);
            if (abs(phi_deriv_alpha) <= (-1) * c2 * phi_deriv_alpha_0) {
                return alpha;
            }

            if (phi_deriv_alpha * (alpha_hi - alpha_lo) >= 0) {
                alpha_hi = alpha_lo;
            }
            alpha_lo = alpha;
        }
        ++i;
    }
}

double line_search_wolfe_condiditons(int dim,
            double (*f)(double x[]),
            void (*gradient)(double x[], double grad[2]),
            double x[],
            double direct[],
            double c1, 
            double c2,
            double alpha_max,
            double alpha_init) {
    // if you want to start with intila alpha being 1.0, then set alpha_init = 1
    // c1: cf p 33, "loose line search" settings, p62
    // c2: "loose line search" settings, p62
    // alpha related params
    double alpha_0 = 0;
    double phi_alpha_0 = phi(dim, f, x, direct, 0);
    double phi_deriv_alpha_0 = phi_derivative(dim, gradient, x, direct, 0);
    double alpha_prev = alpha_0; // used for iteration
    double phi_alpha_prev = phi_alpha_0;
    double alpha = alpha_init;  // step size; used for iteration
    double phi_alpha;  // phi(alpha)
    double phi_deriv_alpha; // phi'(alpha)
    double alpha_star; // used to store final step size; introduced purely to match notations in Nocedal.
    int max_itr_alpha = 10000; // max number of alpha iterations

    // check that the diretion is descending direction
    double grad[dim];
    (*gradient)(x, grad);
    if (inner_product(dim, grad, direct) > 0) {
        cerr << "Line search fails! Direction not descending!" << endl;
        exit(1);
    }

    // start iteration on alpha
    int i_alpha = 1;
    while (true) {
        // check terminating condition
        if (i_alpha == max_itr_alpha) {
            cerr << "Line search fails! Error code: 02" << endl;
            exit(2);
        }

        // evaluate phi(alpha)
        phi_alpha = phi(dim, f, x, direct, alpha);
        if ((phi_alpha > (phi_alpha_0 + c1 * phi_deriv_alpha_0))
            || (phi_alpha >= phi_alpha_prev && i_alpha > 1)) {
                alpha_star = zoom(dim, f, gradient, x, direct, c1, c2, alpha_prev, alpha);
                break;
        }
        // evaluate phi deriv alpha
        phi_deriv_alpha = phi_derivative(dim, gradient, x, direct, alpha);
        if (abs(phi_deriv_alpha) <= (-1) * c2 * phi_deriv_alpha_0) {
            alpha_star = alpha;
            break;
        }

        if (phi_deriv_alpha >= 0) {
            alpha_star = zoom(dim, f, gradient, x, direct, c1, c2, alpha, alpha_prev);
            break;
        }

        // update alpha, alpha_prev
        alpha_prev = alpha;
        alpha = (alpha + alpha_max) / 2;     
        ++i_alpha;
    }
    return alpha_star;
}

double line_search_wolfe_condiditons(int dim,
            double (*f)(double x[]),
            void (*gradient)(double x[], double grad[2]),
            double x[],
            double direct[],
            double c1, 
            double c2,
            double alpha_max) {
    return line_search_wolfe_condiditons(dim, f, gradient, x, direct, c1, c2, alpha_max, alpha_max / 2);
}

void print_vector(int dim, double *x) {
    /**
     * @brief print a vector x of dimension dim
     * 
     */
    for (int i = 0; i < dim; ++i)
        cout << x[i] << " ";
    cout << endl;
}

void identity(int dim, double *M) {
    /**
     * @brief set a square matrix M of dimension dim x dim to identity
     * 
     */
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (i == j)
                M[dim * i + j] = 1.0;
            else
                M[dim * i + j] = 0.0;
        }
    }
}

void trust_region_subproblem(int dim,
                            double B[],
                            double g[],
                            double delta,
                            double p[]) {
    /**
     * @brief Solve the trust region subproblem
     * @arg dim: dimesnion
     * @arg B: coeff for quadratic terms, represented as an array
     * @arg g: coeff for linear terms
     * @arg delta: radius
     * @arg p: output
     */

    double L[dim * dim];  // B = LL'
    double B_lambda_I [dim * dim];  // B + lambda * I
    double g_negate[dim];  // -g
    double r[dim];  // Lr = p
    double zero_threshold = 0.0001; // threshold for whether a number os zero

    // update g_negate
    copy_and_negate(dim, g, g_negate);

    // Case 1): lambda = 0, p(0) <= delta
    if (cholesky_decomposition(dim, B, L)) {
        solve_linear_equations(dim, L, g_negate, p);  // Solve Bp = -g
        if (l2norm(dim, p) <= delta) {
            return;
        }
    }

    // diagnolize B
    Matrix<double, Dynamic, Dynamic> B_alias;
    B_alias = MatrixXd::Identity(dim,dim);
    // filll B values into B_alias
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            B_alias(i, j) = B[dim * i + j];

    EigenSolver<Matrix<double, Dynamic, Dynamic>> es(B_alias);
    Matrix<double, Dynamic, Dynamic> D = es.pseudoEigenvalueMatrix();
    Matrix<double, Dynamic, Dynamic> V = es.pseudoEigenvectors();

    double lambda1 = 100000;  // minimum eigen value of B
    int lambda1_index = -1;  // the index of lambda_1

    // find lambda1
    for (int i = 0; i < dim; ++i) {
        if (D(i, i) < lambda1) {
            lambda1 = D(i, i);
            lambda1_index = i;   
        }
    }

    double q1[dim];  // eigvector of lambda1
    for (int i = 0; i < dim; i++)
        q1[i] = V(i, lambda1_index);

    //cout << "The pseudo-eigenvalue matrix S is:" << endl << D << endl;
    //cout << "The pseudo-eigenvector matrix V is:" << endl << V << endl;

    if (abs(inner_product(dim, q1, g)) >= zero_threshold) {
        //cout << "--- case 2.1 ---" << endl;
        // Case 2.1): lambda > 0, <g, q1> != 0
        // newton's method to find lamba*
        // stop when lambda is converging good enough, output p
        double lambda = 10.0;
        double threshold_lambda = 0.0001;
        double max_itr_lambda = 6;  // Three or four iterations is enough. Cf Nocedal, 91
        double L2[dim * dim]; // for B + lambda*I decomposition

        int i_lambda = 0;
        while (true) {
            // decompose B + lambda*I
            while (true) {
                add_identity(dim, B, B_lambda_I, lambda);  // B + lambda*I
                if (cholesky_decomposition(dim, B_lambda_I, L2)) {  // B + lambda*I = L2L2'
                    break;
                } else
                    lambda *= 2;
            }
            solve_linear_equations(dim, L2, g_negate, p);  // B + lambda*I = -g
            
            // convergence test
            if (abs(l2norm(dim, p) - delta) < threshold_lambda) {
                return;
            }
            
            // max iter test
            // No need to solve subproblem with high accuracy.
            // Three or four iterations is enough. Cf Nocedal, 91
            if (i_lambda == max_itr_lambda) {
                return;
                //std::cerr << "Solving subproblem not convergent!" << std::endl;
                //exit(1);
            }
            
            // update lambda
            solve_lower_triangular_linear_equations(dim, L2, p, r);

            lambda = lambda + pow(l2norm(dim, p) / l2norm(dim, r), 2) * (l2norm(dim, p) - delta) / delta;
            i_lambda++;
        }
    } else {
        //cout << "--- case 2.2 ---"<<endl;
        // Case 2.2): lambda > 0, <g, q1> == 0
        // Nocedal, p88
        // lambda = - lambda_1
        // find tau
        double lambda = - lambda1;
        double tau;  // coeff of z
        double lambda_j; // j-th lambda
        double qj[dim];  // i-th eigen vector
        double accu = 0; // for accumulation

        // compute tau
        for (int j = 0; j < dim; ++j) {
            if (j != lambda1_index) {
                // update lambda_j
                lambda_j = D(j, j);
                // update qj
                for (int i = 0; i < dim; ++i)
                    qj[i] = V(i, j);
                accu += pow(inner_product(dim, qj, g) / (lambda_j + lambda), 2);
            }
        }
        tau = sqrt(pow(delta, 2) - accu);

        // find z
        double z[dim];
        double tmp_norm = l2norm(dim, q1);
        for (int i = 0; i < dim; ++i)
            z[i] = q1[i] / tmp_norm;

        // update p
        // initialize p to be z
        for (int i = 0; i < dim; ++i)
            p[i] = z[i];

        double tmp_scalar;
        for (int j = 0; j < dim; ++j) {
            if (j != lambda1_index) {
                // update lambda_j
                lambda_j = D(j, j);
                // update qj
                for (int i = 0; i < dim; ++i)
                    qj[i] = V(i, j);
                tmp_scalar = inner_product(dim, qj, g) / (lambda_j + lambda);
                x_plus_c_multi_y(dim, p, tmp_scalar, qj, p);
            }
        }
    }
}

double backtracking(int dim,
                    double (*f)(double x[]),
                    double x[],
                    double direct[],
                    double grad_x[],
                    double c) {
    // p37
    double rho = 0.9;
    int max_itr_alpha = 10000;
    double alpha = 1.0;  // alpha start form 1.0 for Newton methods
    double f_x = (*f)(x);
    double f_alpha_x;

    int i = 1;    
    while (true) {
        if (i == max_itr_alpha) {
            cerr << "Backing tracking algorithm fails!";
            exit(1);
        }

        f_alpha_x = phi(dim, f, x, direct, alpha);
        if (f_alpha_x <= (f_x + c * alpha * inner_product(dim, grad_x, direct))) {
            return alpha;
        } else {
            alpha *= rho;
        }
        ++i;
    }
}
