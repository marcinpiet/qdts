#include "SC_PlugIn.h"
#include "Eigen/Dense"
#include <algorithm>

using namespace Eigen;

// InterfaceTable contains pointers to functions in SuperCollider's API
static InterfaceTable *ft;

// Declare the UGen class
struct Solver : public Unit {
    VectorXd X;     // Input vector
    VectorXd T;     // Target vector
    VectorXd solution;  // Solution vector

    int maxIterations;  // Max iterations for Newton's method
    double tolerance;   // Tolerance for minimization
};

// Function prototypes
VectorXd S(VectorXd X);
VectorXd R(VectorXd X);
VectorXd SN(VectorXd X, int N);
VectorXd RN(VectorXd X, int N);
VectorXd A(VectorXd X);
VectorXd F(VectorXd X, VectorXd T);
MatrixXd DF(VectorXd X);
VectorXd MIN_FUNCTION(VectorXd X, VectorXd T, int maxIterations, double tolerance);

// Define the functions here

// S function: Left shift
VectorXd S(VectorXd X) {
    int N = X.rows();
    VectorXd shifted = VectorXd::Zero(N);
    for (int i = 0; i < N - 1; i++) {
        shifted(i) = X(i + 1);
    }
    shifted(N - 1) = 0;
    return shifted;
}

// R function: Right shift
VectorXd R(VectorXd X) {
    int N = X.rows();
    VectorXd shifted = VectorXd::Zero(N);
    for (int i = 0; i < N - 1; i++) {
        shifted(i + 1) = X(i);
    }
    shifted(0) = 0;
    return shifted;
}

// SN function: Apply left shift N times
VectorXd SN(VectorXd X, int N) {
    VectorXd nshifted = X;
    for (int i = 0; i < N; i++) {
        nshifted = S(nshifted);
    }
    return nshifted;
}

// RN function: Apply right shift N times
VectorXd RN(VectorXd X, int N) {
    VectorXd nshifted = X;
    for (int i = 0; i < N; i++) {
        nshifted = R(nshifted);
    }
    return nshifted;
}

// A function: Generates the right part of the equation
VectorXd A(VectorXd X) {
    int N = X.rows();
    VectorXd Y(N + 1);
    Y(0) = 1;
    for (int i = 1; i < N + 1; i++) {
        Y(i) = X(i - 1);
    }
    VectorXd Z = VectorXd::Zero(N);
    for (int i = 0; i < N; i++) {
        Z(i) = Y.head(i + 1).dot(SN(Y, i + 1));
    }
    return Z;
}

// F function: Calculates F(X) = A(X) - T
VectorXd F(VectorXd X, VectorXd T) {
    return A(X) - T;
}

// DF function: Calculates the Jacobian matrix
MatrixXd DF(VectorXd X) {
    int N = X.rows();
    MatrixXd Z = MatrixXd::Zero(N, N);
    VectorXd temp;
    for (int i = 0; i < N - 1; i++) {
        temp = SN(X, i + 1) + RN(X, i + 1);
        for (int j = 0; j < N; j++) {
            Z(i, j) = temp(j);
        }
    }
    Z += MatrixXd::Identity(N, N);
    return Z;
}

// Newton's method: MIN_FUNCTION
VectorXd MIN_FUNCTION(VectorXd X, VectorXd T, int maxIterations, double tolerance) {
    VectorXd solution = VectorXd::Zero(X.rows());
    int step = 0;
    VectorXd f_init = F(X, T);
    double dot_f_init = f_init.dot(f_init);

    while (step < maxIterations && dot_f_init > tolerance) {
        MatrixXd df_val = DF(X);
        X = df_val.colPivHouseholderQr().solve(df_val * X - F(X, T));

        VectorXd f_temp = F(X, T);
        double dot_f_temp = f_temp.dot(f_temp);

        if (dot_f_temp < dot_f_init) {
            solution = X;
            dot_f_init = dot_f_temp;
        }
        step++;
    }
    return solution;
}

// Plugin constructor
extern "C" {
    void Solver_Ctor(Solver* unit) {
        // Initialize solver
        int n = 8; // Default number of harmonics
        unit->maxIterations = 100; // Max iterations
        unit->tolerance = 1e-9; // Tolerance
        unit->X = VectorXd::Random(n);
        unit->T = VectorXd::Random(n);

        // Call the MIN_FUNCTION to compute the solution
        unit->solution = MIN_FUNCTION(unit->X, unit->T, unit->maxIterations, unit->tolerance);

        // Send output values (for debugging)
        for (int i = 0; i < n; ++i) {
            float output = unit->solution(i);
            Out(unit, i, output); // Output the solution for each harmonic
        }
    }
}

// Entry point: Define the entry point to compile as a plugin
PluginLoad(Solver) {
    ft = inTable; // Get function table
    DefineDtorUnit(Solver); // Define the UGen
}