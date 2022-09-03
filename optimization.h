#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H
void fminunc_steepest_descent(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                double x0[],
                                double x[],
                                double &fmin);
void fminunc_newton_direction_with_hessian_modification(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                void (*hessian)(double x[], double hess[]),
                                double x0[],
                                double x[],
                                double &fmin);
void fminunc_trust_region(int dim,
                            double (*f)(double x[]),
                            void (*gradient)(double x[], double grad[]),
                            void (*hessian)(double x[], double hess[]),
                            double x0[],
                            double x[],
                            double &fmin);

void fminunc_conjugate_gradient_fletcher_reeves(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                double x0[],
                                double x[],
                                double &fmin);

void fminunc_quasinewton_bfgs(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                double x0[],
                                double x[],
                                double &fmin);
void fminunc_quasinewton_sr1(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                double x0[],
                                double x[],
                                double &fmin);
void fminunc_line_search_newton_cg(int dim,
                                double (*f)(double x[]),
                                void (*gradient)(double x[], double grad[]),
                                void (*hessian)(double x[], double hess[]),
                                double x0[],
                                double x[],
                                double &fmin);
#endif