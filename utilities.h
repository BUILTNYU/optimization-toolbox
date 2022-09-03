#ifndef UTILITIES_H
#define UTILITIES_H
void copy(int dim, double source[], double dest[]);
void copy_and_negate(int dim, double source[], double dest[]);
void zero_vector(int dim, double *v);
double l2norm(int dim, double v[]);
double inner_product(int dim, double x[], double y[]);
void x_plus_c_multi_y(int dim, double x[], double c, double y[], double output[]);
void a_x_plus_b_y(int dim, double a, double x[], double b, double y[], double output[]);
void scaler_multiplication(int dim, double c, double x[], double output[]);
double phi(int dim,
            double (*f)(double x[]),
            double x[],
            double direct[],
            double alpha);
double phi_derivative(int dim,
                        void (*gradient)(double x[], double grad[2]),
                        double x[],
                        double direct[],
                        double alpha);
double quadratic_approximation(int dim, double B[], double g[], double fx, double p[]);
double matrix_multiplication(int m, int l, int n, double A[], double B[], double output[]);
bool cholesky_decomposition(int dim, double A[], double L[]);
void transpose(int m, int n, double *source, double *dest);
void transpose(int dim, double source[], double dest[]);
void solve_linear_equations(int dim, double L[], double b[], double x[]);
void solve_lower_triangular_linear_equations(int dim, double L[], double b[], double x[]);
void add_identity(int dim, double source[], double dest[], double tau);
double zoom(int dim,
            double (*f)(double x[]),
            void (*gradient)(double x[], double grad[2]),
            double x[],
            double direct[],
            double c1,
            double c2,
            double alpha_lo,
            double alpha_hi);
double line_search_wolfe_condiditons(int dim,
            double (*f)(double x[]),
            void (*gradient)(double x[], double grad[2]),
            double x[],
            double direct[],
            double c1, 
            double c2,
            double alpha_max,
            double alpha_init);
double line_search_wolfe_condiditons(int dim,
            double (*f)(double x[]),
            void (*gradient)(double x[], double grad[2]),
            double x[],
            double direct[],
            double c1,
            double c2,
            double alpha_max);
void print_vector(int dim, double *x);
void identity(int dim, double *M);
void trust_region_subproblem(int dim,
                            double B[],
                            double g[],
                            double delta,
                            double p[]);
double backtracking(int dim,
                    double (*f)(double x[]),
                    double x[],
                    double direct[],
                    double grad_x[],
                    double c);
#endif