#ifndef EXTRAORDINARY_H
#define EXTRAORDINARY_H

#include <QDebug>

#include <array>
#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>
#include <functional>
#include <type_traits>
#include <cassert>
#include <initializer_list>

// Todo list
// DONE 1. Implement GaussJordan.solve(A, b, x)
// DONE 2. Implement COO and BiCGStab for sparse systems
// DONE 3. Implement adaptive time integration
// DONE 4. Implement discontinuity detection
// DONE 5. Implement optimal control (Bellman, Pontryagin)
// 6. Get rid of StepInputs and StepOutputs, which are needless copying
// 7. Implement fully implicit Runge-Kutta
// 8. Add more Butcher tableaus and Adaptive schemes
// 9. Implement Newton-Krylov
// 10. Merge GaussJordan and BicgStab into LinearSolver class
// 11. Merge NewtonRaphson and NewtonKrylov into NewtonSolver class
// 12. Implement Pontryagin::solveIVP

namespace XO
{
const int DYNAMIC_DENSE = -1;
const int COO_SPARSE = -2;
const int MAX_RK_STAGES = 9;

enum RungeKuttaMethod
{
    FORWARD_EULER = 1110,
    HEUN_2 = 1120,
    CLASSIC_RK4 = 1140,
    EULER_HEUN_12 = 1220,
    BACKWARD_EULER = 2110,
    IMPLICIT_MIDPOINT = 2120,
    DIRK_L_S3O3 = 2130,
};

enum TimeSteppingScheme
{
    UNIFORM_STEPPING = 0,
    H321_CLASSIC = 1,
    H321_PPID = 2,
};

enum RungeKuttaCategory
{
    EXPLICIT = 1,
    DIRK = 2,
    IMPLICIT = 3,
};

enum RungeKuttaFunctionType
{
    IN_T_X_OUT_V = 0,
    IN_T_X_OUT_V_JVX = 1,
    IN_T_X_V_OUT_R = 2,
    IN_T_X_V_OUT_R_JRX_JRV = 3,
};

template<typename T, int N, typename SubType = void> class Vector {};

template<typename T, int M, int N = M, typename SubType = void> class Matrix {};

template<typename T, int N> class Vector<T, N, typename std::enable_if<(N>=0)>::type>
{
private:
    std::array<T,N> data;
public:
    inline Vector() {}
    inline Vector(const int &n) {resize(n);}
    inline Vector(const std::initializer_list<T> &list) : data(list) {}
    inline void resize(const int &n) {}
    inline void rezero() {std::fill(data.begin(), data.end(), T(0));}
    inline void rezero(const int &n) {resize(n); rezero();}
    inline void fill(const T &initVal) {std::fill(data.begin(), data.end(), initVal);}
    inline void assign(const int &n, const T &initVal) {resize(n); rezero(initVal);}
    inline int size() const {return (int)data.size();}
    inline T& operator[](const int &i) {return data[i];}
    inline T operator[](const int &i) const {return data[i];}
    inline Vector& operator+=(const Vector &other) {for (int i = 0; i < size(); i++) data[i] += other[i]; return (*this);}
    inline Vector operator+(const Vector &other) const {Vector result(*this); result += other; return result;}
    inline Vector& operator-=(const Vector &other) {for (int i = 0; i < size(); i++) data[i] -= other[i]; return (*this);}
    inline Vector operator-(const Vector &other) const {Vector result(*this); result -= other; return result;}
    inline Vector& operator*=(const T &other) {for (int i = 0; i < size(); i++) data[i] *= other; return (*this);}
    inline Vector operator*(const T &other) const {Vector result(*this); result *= other; return result;}
    inline T dot(const Vector &other) const {T result = T(0); for (int i = 0; i < size(); i++) {result += operator[](i) * other[i];} return result;}
    inline T lengthSquared() const {return dot(*this);}
    inline T length() const {return std::sqrt(lengthSquared());}
    inline void normalize() {operator*=(T(1) / length());}
    inline void normalized() {T result = *this; result.normalize(); return result;}
    inline T sum() {T result = T(0); for (int i = 0; i < size(); i++) result += data[i]; return result;}
};

template<typename T, int N> class Vector<T, N, typename std::enable_if<(N<0)>::type>
{
private:
    std::vector<T> data;
public:
    inline Vector() {}
    inline Vector(const int &n) {resize(n);}
    inline Vector(const std::initializer_list<T> &list) : data(list) {}
    inline void resize(const int &n) {data.resize(n);}
    inline void rezero() {std::fill(data.begin(), data.end(), T(0));}
    inline void rezero(const int &n) {resize(n); rezero();}
    inline void fill(const T &initVal) {std::fill(data.begin(), data.end(), initVal);}
    inline void assign(const int &n, const T &initVal) {resize(n); rezero(initVal);}
    inline int size() const {return (int)data.size();}
    inline T& operator[](const int &i) {return data[i];}
    inline T operator[](const int &i) const {return data[i];}
    inline Vector& operator+=(const Vector &other) {for (int i = 0; i < size(); i++) data[i] += other[i]; return (*this);}
    inline Vector operator+(const Vector &other) const {Vector result(*this); result += other; return result;}
    inline Vector& operator-=(const Vector &other) {for (int i = 0; i < size(); i++) data[i] -= other[i]; return (*this);}
    inline Vector operator-(const Vector &other) const {Vector result(*this); result -= other; return result;}
    inline Vector& operator*=(const T &other) {for (int i = 0; i < size(); i++) data[i] *= other; return (*this);}
    inline Vector operator*(const T &other) const {Vector result(*this); result *= other; return result;}
    inline T dot(const Vector &other) const {assert(size() == other.size()); T result = T(0); for (int i = 0; i < size(); i++) {result += operator[](i) * other[i];} return result;}
    inline T lengthSquared() const {return dot(*this);}
    inline T length() const {return std::sqrt(lengthSquared());}
    inline void normalize() {operator*=(T(1) / length());}
    inline void normalized() {T result = *this; result.normalize(); return result;}
    inline T sum() {T result = T(0); for (int i = 0; i < size(); i++) result += data[i]; return result;}
};

template<typename T, int M, int N> class Matrix<T, M, N, typename std::enable_if<((M>=0)&&(N>=0))>::type>
{
private:
    std::array<T,M*N> data;
public:
    inline Matrix() {}
    inline Matrix(const int &m, const int &n) {resize(m, n);}
    inline void resize(const int &m, const int &n) {}
    inline void rezero() {std::fill(data.begin(), data.end(), T(0));}
    inline void rezero(const int &m, const int &n) {resize(m, n); rezero();}
    inline void fill(const T &initVal) {std::fill(data.begin(), data.end(), initVal);}
    inline void assign(const int &m, const int &n, const T &initVal) {resize(m, n); rezero(initVal);}
    inline int rows() const {return M;}
    inline int cols() const {return N;}
    inline void add(const int &i, const int &j, const T &coeff) {operator()(i,j) += coeff;}
    inline T& operator()(const int &i, const int &j) {return data[i*N+j];}
    inline T operator()(const int &i, const int &j) const {return data[i*N+j];}
    inline Matrix transposed() {Matrix result(rows(),cols()); for (int i = 0; i < rows(); i++) for (int j = 0; j < cols(); j++) result(i,j)=operator()(j,i); return result;}
    inline Matrix& operator+=(const Matrix &other) {for (int i = 0; i < data.size(); i++) data[i] += other.data[i]; return (*this);}
    inline Matrix operator+(const Matrix &other) const {Matrix result(*this); result += other; return result;}
    inline Matrix& operator-=(const Matrix &other) {for (int i = 0; i < data.size(); i++) data[i] -= other.data[i]; return (*this);}
    inline Matrix operator-(const Matrix &other) const {Matrix result(*this); result -= other; return result;}
    inline Matrix& operator*=(const T &other) {for (int i = 0; i < data.size(); i++) data[i] *= other; return (*this);}
    inline Matrix operator*(const T &other) const {Matrix result(*this); result *= other; return result;}
    inline Vector<T,N> operator*(const Vector<T,N> other) const
    {
        Q_ASSERT(cols() == other.size());
        Vector<T,N> result;
        result.resize(rows());
        for (int i = 0; i < rows(); i++)
        {
            T &r = result[i];
            r = T(0);
            for (int j = 0; j < cols(); j++)
            {
                r += operator()(i, j) * other[j];
            }
        }
        return result;
    }
};

template<typename T, int M, int N> class Matrix<T, M, N, typename std::enable_if<((M==DYNAMIC_DENSE)||(N==DYNAMIC_DENSE))>::type>
{
private:
    int mrows;
    int ncols;
    std::vector<T> data;
public:
    inline Matrix() {}
    inline Matrix(const int &m, const int &n) {resize(m, n);}
    inline void resize(const int &m, const int &n) {mrows=m; ncols=n; data.resize(m*n);}
    inline void rezero() {std::fill(data.begin(), data.end(), T(0));}
    inline void rezero(const int &m, const int &n) {resize(m, n); rezero();}
    inline void fill(const T &initVal) {std::fill(data.begin(), data.end(), initVal);}
    inline void assign(const int &m, const int &n, const T &initVal) {resize(m, n); rezero(initVal);}
    inline int rows() const {return mrows;}
    inline int cols() const {return ncols;}
    inline void add(const int &i, const int &j, const T &coeff) {operator()(i,j) += coeff;}
    inline T& operator()(const int &i, const int &j) {return data[i*ncols+j];}
    inline T operator()(const int &i, const int &j) const {return data[i*ncols+j];}
    inline Matrix transposed() {Matrix result(rows(),cols()); for (int i = 0; i < rows(); i++) for (int j = 0; j < cols(); j++) result(i,j)=operator()(j,i); return result;}
    inline Matrix& operator+=(const Matrix &other) {for (int i = 0; i < data.size(); i++) data[i] += other.data[i]; return (*this);}
    inline Matrix operator+(const Matrix &other) const {Matrix result(*this); result += other; return result;}
    inline Matrix& operator-=(const Matrix &other) {for (int i = 0; i < data.size(); i++) data[i] -= other.data[i]; return (*this);}
    inline Matrix operator-(const Matrix &other) const {Matrix result(*this); result -= other; return result;}
    inline Matrix& operator*=(const T &other) {for (int i = 0; i < data.size(); i++) data[i] *= other; return (*this);}
    inline Matrix operator*(const T &other) const {Matrix result(*this); result *= other; return result;}
    inline Vector<T,N> operator*(const Vector<T,N> other) const
    {
        Q_ASSERT(cols() == other.size());
        Vector<T,N> result;
        result.resize(rows());
        for (int i = 0; i < rows(); i++)
        {
            T &r = result[i];
            r = T(0);
            for (int j = 0; j < cols(); j++)
            {
                r += operator()(i, j) * other[j];
            }
        }
        return result;
    }
};

template<typename T> struct CooEntry
{
    int i;
    int j;
    T coeff;
    CooEntry(const int &i_, const int &j_, const int &coeff_) : i(i_), j(j_), coeff(coeff_) {}
};

template<typename T, int M, int N> class Matrix<T, M, N, typename std::enable_if<((M==COO_SPARSE)&&(N==COO_SPARSE))>::type>
{
private:
    int mrows;
    int ncols;
    bool isOptimized = false;
    std::vector<CooEntry<T>> data;
public:
    inline Matrix(const int &m = 0, const int &n = 0) {resize(m, n);}
    inline void resize(const int &m, const int &n) {mrows=m; ncols=n;}
    inline void rezero() {data.clear();}
    inline int rows() const {return mrows;}
    inline int cols() const {return ncols;}
    inline void add(const int &i, const int &j, const T &coeff) {data.push_back(CooEntry<T>(i,j,coeff));}
    inline T& operator()(const int &i, const int &j) {assert(false); return data[0].coeff;} // note: you should never use this
    inline T operator()(const int &i, const int &j) const {assert(false); return data[0].coeff;} // note: you should never use this
    inline Matrix transposed() {Matrix result = *this; for (int k = 0; k < result.data.size(); k++) std::swap(result.data[k].i,result.data[k].j); return result;}
    inline void finalize() {} // fixme: sort into banded alternating format and remove non-unique entries
    inline Vector<T,N> operator*(const Vector<T,N> other) const
    {
        Q_ASSERT(cols() == other.size());
        Vector<T,N> result;
        result.resize(rows());
        result.rezero();
        for (int iEntry = 0; iEntry < data.size(); iEntry++)
        {
            const CooEntry<T> &entry = data[iEntry];
            result[entry.i] += entry.coeff * other[entry.j];
        }
        return result;
    }
};

template<typename T> struct LinearSolverSettings
{
    int maximumIterations;
    T relConvTolerance;
    T absConvTolerance;
    int maxKrylovSubspaceSize;
    LinearSolverSettings(const int &maximumIterations_ = 1000,
                         const T &relConvTolerance_ = 1e-12,
                         const T &absConvTolerance_ = 1e-12)
        : maximumIterations(maximumIterations_),
        relConvTolerance(relConvTolerance_),
        absConvTolerance(absConvTolerance_) {}
};

template<typename T, int N> class LinearSolver
{
public:
    static void solveGJ(const Matrix<T,N,N> &A, const Vector<T,N> &b, Vector<T,N> &x, bool &success)
    {
        Q_ASSERT(A.rows() == A.cols());
        Q_ASSERT(A.cols() == b.size());
        success = true;
        x.resize(A.cols());
        if (A.rows() == 1)
        {
            x.resize(1);
            x[0] = b[0] / A(0,0);
            if (A(0,0) == T(0)) success = false;
            return;
        }
        if (A.rows() == 2)
        {
            T det = A(0,0) * A(1,1) - A(1,0) * A(0,1);
            T det0 = b[0] * A(1,1) - b[1] * A(0,1);
            T det1 = A(0,0) * b[1] - A(1,0) * b[0];
            x.resize(2);
            x[0] = det0 / det;
            x[1] = det1 / det;
            if (det == T(0)) success = false;
            return;
        }
        Matrix<T,N,N> U = A;
        Vector<T,N> c = b;
        // scale the matrix and vector to improve numerical stability
        T rowMagnitude, rowScalingFactor;
        int i, j, k, iPivot;
        for (i = 0; i < U.rows(); i++)
        {
            rowMagnitude = (T)0.0;
            for (j = 0; j < U.rows(); j++)
            {
                rowMagnitude += std::fabs(U(i, j));
            }
            rowScalingFactor = (T)U.cols() / rowMagnitude;
            for (j = 0; j < U.cols(); j++)
            {
                U(i, j) *= rowScalingFactor;
            }
            c[i] *= rowScalingFactor;
        }
        // perform row reduction by partial pivoting
        Vector<int,N> isRowEchelon; // note: bool would be better, but you cannot assign to vector<bool>
        Vector<int,N> echelonRowOrder;
        isRowEchelon.resize(A.rows());
        isRowEchelon.rezero();
        echelonRowOrder.resize(A.rows());
        for (k = 0; k < U.rows(); k++)
        {
            // perform partial pivoting
            iPivot = -1;
            for (i = 0; i < U.rows(); i++)
            {
                if (isRowEchelon[i] == 1) continue;
                if ((iPivot == -1) || (std::fabs(U(i,k)) < std::fabs(U(iPivot,k)))) iPivot = i;
            }
            isRowEchelon[iPivot] = 1;
            echelonRowOrder[k] = iPivot;
            // perform row reduction
            for (i = 0; i < U.rows(); i++)
            {
                if (isRowEchelon[i] == 1) continue;
                rowScalingFactor = U(i, k) / U(iPivot, k);
                for (j = 0; j < U.cols(); j++)
                {
                    U(i, j) -= (U(iPivot, j) * rowScalingFactor);
                }
                c[i] -= (c[iPivot] * rowScalingFactor);
            }
        }
        // perform back substitution
        for (k = U.rows()-1; k >= 0; k--)
        {
            i = echelonRowOrder[k];
            x[i] = c[i];
            for (j = U.rows()-1; j > i; j--)
            {
                x[i] -= (U(i, j) * x[j]);
            }
            if (U(i,i) == T(0)) success = false;
            x[i] /= U(i,i);
        }
    }
    static void solveBCGS(const Matrix<T,N,N> &A, const Vector<T,N> &b, Vector<T,N> &x, bool &success, const LinearSolverSettings<T> &settings = LinearSolverSettings<T>())
    {
        assert(A.rows() == A.cols());
        assert(A.rows() == b.size());
        success = true;
        const int n = A.rows();
        const T bss = b.dot(b);
        const T absTol = std::fmax(settings.absConvTolerance, settings.relConvTolerance * settings.relConvTolerance * bss);
        if (bss == T(0)) {x.rezero(n); return;}
        Vector<T,N> r  = (x.size() == n ? b - A * x : b);
        if (x.size() != n) x.rezero(n);
        T rss = r.dot(r);
        if (rss < absTol) return;
        Vector<T,N> r0 = r;
        T r0ss = rss;
        T restartTol = std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon();
        Vector<T,N> p, s, t, v, y, z;
        p.rezero(n), s.rezero(n), t.rezero(n), v.rezero(n), y.rezero(n), z.rezero(n);
        T rho = T(1.0), alpha = T(1.0), w = T(1.0), beta, rhoOld, tsScheme;
        for (int iIter = 0; iIter < settings.maximumIterations; iIter++)
        {
            rhoOld = rho;
            rho = r0.dot(r);
            if (std::fabs(rho) < restartTol * r0ss) // r & r0 are orthogonal, so restart
            {
                r  = b - A * x;
                r0 = r;
                rss = r.dot(r);
                r0ss = rss;
                rho = r0ss;
            }
            beta = (rho / rhoOld) * (alpha / w);
            p = r + (p - v * w) * beta;
            y = p; // fixme: use diagonal preconditioner
            v = A * y;
            alpha = rho / r0.dot(v);
            s = r - v * alpha;
            z = s; // fixme: use diagonal preconditioner
            t = A * z;
            tsScheme = t.dot(t);
            w = (tsScheme > T(0) ? t.dot(s) / tsScheme : T(0));
            x += (y * alpha + z * w);
            r = s - t * w;
            rss = r.dot(r);
            qDebug() << "rss= " << rss <<"  "<< absTol <<"  "<< iIter;
            if (rss <= absTol) break;
        }
    }
    static void solve(const Matrix<T,N,N> &A, const Vector<T,N> &b, Vector<T,N> &x, bool &success, const LinearSolverSettings<T> &settings = LinearSolverSettings<T>())
    {
        if (N == COO_SPARSE) {solveBCGS(A, b, x, success, settings);}
        else {solveGJ(A, b, x, success);}
    }
};

template <typename T> struct NewtonSettings
{
    int maximumIterations;
    T convergenceTolerance;
    T trustRegionSize;
    LinearSolverSettings<T> linearSolverSettings;
    NewtonSettings(const T &maximumIterations_ = 1000,
                   const T &convergenceTolerance_ = 1e-9,
                   const T &trustRegionSize_ = std::numeric_limits<T>::max(),
                   const LinearSolverSettings<T> &linearSolverSettings_ = LinearSolverSettings<T>())
    {
        maximumIterations = maximumIterations_;
        convergenceTolerance = convergenceTolerance_;
        trustRegionSize = trustRegionSize_;
        linearSolverSettings = linearSolverSettings_;
    }
};

template <typename T, int N> class NewtonSolver
{
    NewtonSolver() {} // private constructor, this is a namespace class
public:
    typedef std::function<void(const Vector<T,N> &x, Vector<T,N> &r)> NewtonKrylovFunction;
    typedef std::function<void(const Vector<T,N> &x, Vector<T,N> &r, Matrix<T,N,N> &J)> NewtonRaphsonFunction;
public:
    static void solve(const NewtonRaphsonFunction &f, const Vector<T,N> &guess,
                      Vector<T,N> &solution, bool &success,
                      const NewtonSettings<T> &settings = NewtonSettings<T>())
    {
        success = false;
        Vector<T,N> &x = solution;
        x = guess;
        Vector<T,N> r, dx;
        Matrix<T,N,N> J;
        T maxAbsError;
        for (int iIter = 0; iIter < settings.maximumIterations; iIter++)
        {
            f(x, r, J);
            LinearSolver<T,N>::solve(J, r, dx, success, settings.linearSolverSettings);
            if (success == false)
            {
                qDebug() << "WARNING: FOUND A SINGULARITY SO THE SOLUTION WAS PERTURBED!";
                for (int i = 0; i < r.size(); i++) dx[i] = -0.5;
            }
            //qDebug() << "r= " << r[0] <<" "<< r[1] <<" "<< r[2];
            maxAbsError = (T)0.0;
            for (int i = 0; i < r.size(); i++)
            {
                maxAbsError = std::fmax(maxAbsError, std::fabs(r[i]));
            }
            //qDebug() << "iteration= " << iIter <<", maxAbsError= " << maxAbsError;
            if (maxAbsError <= settings.convergenceTolerance)
            {
                success = true;
                return;
            }
            T relaxationFactor = (T)1.0;
            for (int i = 0; i < dx.size(); i++)
            {
                relaxationFactor = std::fmin(relaxationFactor, std::fabs(settings.trustRegionSize / dx[i]));
            }
            //if (N == 2) qDebug() << "NEWTON: " << iIter <<" | "<< x[0] <<"  "<< x[1] <<" | "<< r[0] <<"  "<< r[1] <<" | "<< J(0,0) <<"  "<< J(0,1) <<"  "<< J(1,0) <<"  "<< J(1,1) <<" | "<< dx[0] <<"  "<< dx[1];
            //if (N == 2) qDebug() << "NEWTON: " << iIter <<"  "<< maxAbsError <<"  "<< settings.convergenceTolerance <<"  "<< x[0] <<"  "<< relaxationFactor <<" | "<< J(0,0) <<" "<< r[0] <<" "<< dx[0];
            x -= dx * relaxationFactor;
        }
    }
    static void solve(const NewtonKrylovFunction &f, const Vector<T,N> &guess,
                      Vector<T,N> &solution, bool &success,
                      const NewtonSettings<T> &settings = NewtonSettings<T>())
    {
        int n = guess.size();
        //qDebug() << "nnnnnnnn= " << n <<", NNNNNNN= " << N;
        if (true) //n == 1) // secant method for now
        {
            NewtonRaphsonFunction g = [f,n](const Vector<T,N> &x, Vector<T,N> &r, Matrix<T,N,N> &J)
            {
                const T FDSTEP = 1e-3;
                Vector<T,N> xp = x, rp;
                xp[0] += FDSTEP;
                f(xp, rp);
                f(x, r); // note: do unperturbed calculation last so good value will be saved
                J.resize(n,n);
                J(0,0) = (rp[0] - r[0]) / FDSTEP;
                //if (N==2) qDebug() << "YYYYYY: " << x[0] <<"  "<< x[1] <<" | "<< r[0] <<"  "<< r[1];
                //if (N==2) qDebug() << "ZZZZZZ: " << xp[0] <<"  "<< xp[1] <<" | "<< rp[0] <<"  "<< rp[1];
                if (N > 1)
                {
                    r[1] = 0.0;
                    J(0,1) = 0.0;
                    J(1,0) = 0.0;
                    J(1,1) = 1.0;
                }
            };
            solve(g, guess, solution, success, settings);
        }
        else
        {
            assert(false); // fixme: implement Newton-LGMRES method
        }
    }
};

template <typename T> class NewtonSolver1
{
    NewtonSolver1() {} // private constructor, this is a namespace class
public:
    static void solve(const std::function<void(T,T&,T&)> &f, const T &guess,
                      T &solution, bool &success,
                      const NewtonSettings<T> &settings = NewtonSettings<T>())
    {
        Vector<T,1> x0, x;
        x0.resize(1); x0[0] = guess;
        typename NewtonSolver<T,1>::NewtonRaphsonFunction g = [f](const Vector<T,1> &x, Vector<T,1> &r, Matrix<T,1,1> &J)
        {
            f(x[0], r[0], J(0,0));
        };
        NewtonSolver<T,1>::solve(g, x0, x, success, settings);
        solution = x[0];
    }
    static void solve(const std::function<void(T,T&)> &f, const T &guess,
                      T &solution, bool &success,
                      const NewtonSettings<T> &settings = NewtonSettings<T>())
    {
        Vector<T,1> x0, x;
        x0.resize(1); x0[0] = guess;
        typename NewtonSolver<T,1>::NetwonKrylovFunction g = [f](const Vector<T,1> &x, Vector<T,1> &r)
        {
            f(x[0], r[0]);
        };
        NewtonSolver<T,1>::solve(g, x0, x, success, settings);
        solution = x[0];
    }
};

template <typename T> struct ButcherTableau
{
    RungeKuttaMethod method = CLASSIC_RK4;
    RungeKuttaCategory rkCategory = EXPLICIT;
    bool isAdaptive = false;
    T p = 0;
    T pHat = 0;
    int numStages = 0;
    Matrix<T,MAX_RK_STAGES,MAX_RK_STAGES> a;
    Vector<T,MAX_RK_STAGES> b, c, e;
    ButcherTableau() {}
    ButcherTableau(const RungeKuttaMethod &rkMethod);
};

template <typename T> struct TimeSteppingParameters
{
    T alpha;
    T beta;
    T gamma;
    T a;
    T b;
    TimeSteppingParameters() {}
    TimeSteppingParameters(const TimeSteppingScheme &tsScheme, const T &p, const T &pHat);
};

template <typename T, int N> struct SingleStepInputs
{
    T t0;
    T t1;
    T dt;
    Vector<T,N> x0;
    Vector<T,N> vGuess;
    ButcherTableau<T> bt;
    NewtonSettings<T> newtonSettings;
};

template <typename T, int N> struct SingleStepOutputs
{
public:
    Vector<T,N> x1;
    Vector<T,N> x1e;
    Vector<T,N> v0;
    Vector<T,N> v1;
    bool success;
};

template <typename T> struct TimeIntegration
{
    RungeKuttaMethod rkMethod = CLASSIC_RK4;
    TimeSteppingScheme tsScheme = UNIFORM_STEPPING;
    T startTime = 0.0;
    T finishTime;
    T minTimeStep;
    T maxTimeStep;
    T minAdjustment = 0.5;
    T maxAdjustment = 2.0;
    T safetyFactor = 0.9;
    T relativeTolerance = 1e-5;
    T absoluteTolerance = 1e-10;
    T errorNormLevel = 2.0;
    NewtonSettings<T> newtonSettings;
};

template <typename T, int N> struct InitialConditions
{
    T time;
    Vector<T,N> value;
};

template <typename T> struct BoundaryCondition
{
    T time;
    int index;
    T value;
    BoundaryCondition() {}
    BoundaryCondition(const T &time_, const int &index_, const T &value_)
        : time(time_), index(index_), value(value_) {}
};

template <typename T, int Ns, int Nc=0> struct OdeSolution
{
public:
    struct Snapshot
    {
        T t; // time
        Vector<T,Ns> x; // state variables
        Vector<T,Ns> v; // state variable time-derivatives
        Vector<T,Nc> u; // control variables
    };
    std::vector<Snapshot> snapshots;
    T error;
    T integratedCost;
    void print()
    {
        qDebug() << "SOLUTION:";
        qDebug() << "integratedCost= " << integratedCost;
        T t, x, v, u;
        for (int i = 0; i < snapshots.size(); i++)
        {
            t = snapshots[i].t;
            x = snapshots[i].x[0];
            v = snapshots[i].v[0];
            u = (Nc >= 1 ? snapshots[i].u[0] : 0.0);
            qDebug() << "i,t,x,v,u= " << i <<", "<< t <<", "<< x <<", "<< v <<", "<< u;
        }
    }
};

template <typename T, int N> struct RungeKuttaFunction
{
    typedef std::function<Vector<T,N>(const T &t, const Vector<T,N>& x)> FunctionE;
    typedef std::function<void(const T &t, const Vector<T,N>& x, Vector<T,N>& v, Matrix<T,N,N>& Jvx)> FunctionEJ;
    typedef std::function<T(const T &t, const Vector<T,N>& x, const Vector<T,N>& v)> FunctionI;
    typedef std::function<void(const T &t, const Vector<T,N>& x, const Vector<T,N>& v, Vector<T,N>& r, Matrix<T,N,N>& Jrx, Matrix<T,N,N>& Jrv)> FunctionIJ;
    RungeKuttaFunctionType functionType;
    FunctionE fe; // note: this is a tagged union, but union does not allow std::function
    FunctionEJ fej;
    FunctionI fi;
    FunctionIJ fij;
    RungeKuttaFunction(const FunctionE &fe_) : functionType(IN_T_X_OUT_V) {fe = fe_;}
    RungeKuttaFunction(const FunctionEJ &fej_) : functionType(IN_T_X_OUT_V_JVX) {fej = fej_;}
    RungeKuttaFunction(const FunctionI &fi_) : functionType(IN_T_X_V_OUT_R) {fi = fi_;}
    RungeKuttaFunction(const FunctionIJ &fij_) : functionType(IN_T_X_V_OUT_R_JRX_JRV) {fij = fij_;}
};

template <typename T, int N> class RungeKutta
{
    RungeKutta() {} // private constructor, this is a namespace class
public:
    static SingleStepOutputs<T,N> stepOdeSystem(const RungeKuttaFunction<T,N> &f, const SingleStepInputs<T,N> inputs);
    static T calculateEmbeddedError(const TimeIntegration<T> &ti, const TimeSteppingParameters<T> &ts, const Vector<T,N> &x, const Vector<T,N> &xe);
    static void adaptTimeStep(const TimeIntegration<T> &ti, const TimeSteppingParameters<T> &ts, const std::deque<T> &embErrorHistory, const std::deque<T> &timeStepHistory, bool &repeat, T &dt);
    static OdeSolution<T,N> integrateIVP(const RungeKuttaFunction<T,N> &f, const InitialConditions<T,N> &ic, const TimeIntegration<T> &ti);
    static OdeSolution<T,N> solveIVP(const RungeKuttaFunction<T,N> &f, const InitialConditions<T,N> &ic, const TimeIntegration<T> &ti);
    static OdeSolution<T,N> solveBVP(const RungeKuttaFunction<T,N> &f, const std::vector<BoundaryCondition<T>> &bc, const TimeIntegration<T> &ti, const InitialConditions<T,N> &icGuess);
};

template <typename T, int Ns, int Nc> class Pontryagin
{
    Pontryagin() {} // private constructor, this is a namespace class
public:
    typedef std::function<void(const T& t, const Vector<T,Ns>& x, const Vector<T,Nc>& u, Vector<T,Ns>& v, Matrix<T,Ns,Ns>& Jvx, Matrix<T,Ns,Nc>& Jvu)> GoverningFunction;
    typedef std::function<void(const T& t, const Vector<T,Ns>& x, const Vector<T,Nc>& u, T& L, Vector<T,Ns>& Lx, Vector<T,Nc>& Lu)> InstantaneousCostFunction;
    typedef std::function<void(const T& t, const Vector<T,Ns>& x, T& Psi, Vector<T,Ns>& Psix)> FinalCostFunction;
public:
    static OdeSolution<T,Ns,Nc> solveIVP(const GoverningFunction &f, const InstantaneousCostFunction &L,
                                         const InitialConditions<T,Ns> &icx, const InitialConditions<T,Ns> &icu,
                                         const TimeIntegration<T> &ti);
    static OdeSolution<T,Ns,Nc> solveBVP(const GoverningFunction &f, const InstantaneousCostFunction &L,
                                         const InitialConditions<T,Ns> &icx, const InitialConditions<T,Ns> &fcx,
                                         const TimeIntegration<T> &ti, const InitialConditions<T,Nc> &uGuess);
};

template <typename T> class RungeKutta1
{
    RungeKutta1() {} // private constructor, this is a namespace class
public:
    static OdeSolution<T,1> solveIVP1(const std::function<T(T,T)> &f, const T &t0, const T &x0, const TimeIntegration<T> &ti);
    static OdeSolution<T,1> solveIVP1(const std::function<void(T,T,T&,T&)> &f, const T &t0, const T &x0, const TimeIntegration<T> &ti);
    static OdeSolution<T,1> solveIVP1(const std::function<void(T,T,T,T&,T&,T&)> &f, const T &t0, const T &x0, const TimeIntegration<T> &ti);
    static std::vector<std::array<T,4>> solveIVP2(const std::function<T(T,T,T)> &f, const T &t0, const T &x0, const T &v0, const TimeIntegration<T> &ti);
};

template <typename T> class Pontryagin1
{
    Pontryagin1() {} // private constructor, this is a namespace class
public:
    static OdeSolution<T,1,1> solveIVP(const std::function<void(T,T,T,T&,T&,T&)> &f,
                                       const std::function<void(T,T,T,T&,T&,T&)> &L,
                                       const T &t0, const T &x0, const T &u0,
                                       const TimeIntegration<T> &ti);
    static OdeSolution<T,1,1> solveBVP(const std::function<void(T,T,T,T&,T&,T&)> &f,
                                       const std::function<void(T,T,T,T&,T&,T&)> &L,
                                       const T &t0, const T &x0, const T &t1, const T &x1,
                                       const TimeIntegration<T> &ti, const T &u0Guess = T(0));
};

template <typename T> ButcherTableau<T>::ButcherTableau(const RungeKuttaMethod &rkMethod)
{
    method = rkMethod;
    rkCategory = EXPLICIT;
    isAdaptive = false;
    p = 0;
    pHat = 0;
    numStages = 0;
    a.rezero();
    b.rezero();
    c.rezero();
    e.rezero();
    switch (rkMethod)
    {
        case FORWARD_EULER:
        {
            rkCategory = EXPLICIT;
            isAdaptive = false;
            p = 1;
            numStages = 1;
            b[0] = 1.0;
            c[0] = 0.0;
            break;
        }
        case HEUN_2:
        {
            rkCategory = EXPLICIT;
            isAdaptive = false;
            p = 2;
            numStages = 2;
            a(1,0) = 1.0;
            b[0] = 0.5;
            b[1] = 0.5;
            c[0] = 0.0;
            c[1] = 1.0;
            break;
        }
        case EULER_HEUN_12:
        {
            rkCategory = EXPLICIT;
            isAdaptive = true;
            p = 2;
            pHat = 1;
            numStages = 2;
            a(1,0) = 1.0;
            b[0] = 0.5;
            b[1] = 0.5;
            c[1] = 1.0;
            e[0] = 1.0;
            break;
        }
        case CLASSIC_RK4:
        {
            rkCategory = EXPLICIT;
            isAdaptive = false;
            p = 4;
            numStages = 4;
            a(1,0) = 0.5;
            a(2,1) = 0.5;
            a(3,2) = 1.0;
            b[0] = 1.0/6.0;
            b[1] = 1.0/3.0;
            b[2] = 1.0/3.0;
            b[3] = 1.0/6.0;
            c[1] = 0.5;
            c[2] = 0.5;
            c[3] = 1.0;
            break;
        }
        case BACKWARD_EULER:
        {
            rkCategory = DIRK;
            isAdaptive = false;
            p = 1;
            numStages = 1;
            a(0,0) = 1.0;
            b[0] = 1.0;
            c[0] = 1.0;
            break;
        }
        case IMPLICIT_MIDPOINT:
        {
            rkCategory = DIRK;
            isAdaptive = false;
            p = 2;
            numStages = 1;
            a(0,0) = 0.5;
            b[0] = 1.0;
            c[0] = 0.5;
            break;
        }
        case DIRK_L_S3O3:
        {
            rkCategory = DIRK;
            isAdaptive = false;
            p = 3;
            numStages = 3;
            T x = 0.4358665215;
            T x2 = x * x;
            a(0,0) = x;
            a(1,1) = x;
            a(2,2) = x;
            a(1,0) = 0.5 - 0.5 * x;
            a(2,0) = -1.5 * x2 + 4.0 * x - 0.25;
            a(2,1) = 1.5 * x2 - 5.0 * x + 1.25;
            b[0] = a(2,0);
            b[1] = a(2,1);
            b[2] = a(2,2);
            c[0] = x;
            c[1] = 0.5 + 0.5 * x;
            c[2] = 1.0;
            break;
        }
        default:
        {
            assert(false);
            break;
        }
    }
}

template <typename T> TimeSteppingParameters<T>::TimeSteppingParameters(const TimeSteppingScheme &tsScheme, const T &p, const T &pHat)
{
    if (tsScheme == H321_CLASSIC)
    {
        alpha = 1.0 / (3.0 * pHat);
        beta = -1.0 / (18.0 * pHat);
        gamma = -5.0 / (18.0 * pHat);
        a = 5.0 / 6.0;
        b = 1.0 / 6.0;
    }
    else if (tsScheme == H321_PPID)
    {
        alpha = 6.0 / (20.0 * pHat);
        beta = -1.0 / (20.0 * pHat);
        gamma = -5.0 / (20.0 * pHat);
        a = 1.0;
        b = 0.0;
    }
    else
    {
        assert(false);
    }
}

template <typename T, int N> SingleStepOutputs<T,N> RungeKutta<T,N>::stepOdeSystem(const RungeKuttaFunction<T,N> &f, const SingleStepInputs<T,N> inputs)
{
    SingleStepOutputs<T,N> outputs;
    Vector<T,MAX_RK_STAGES> t;
    std::array<Vector<T,N>,MAX_RK_STAGES> x, v, dx;
    Matrix<T,N,N> Jvx;
    T dt = inputs.t1 - inputs.t0;
    outputs.success = true;
    outputs.x1 = inputs.x0;
    outputs.x1e = inputs.x0;
    assert(inputs.bt.rkCategory != IMPLICIT); // fixme: implement fully implicit soon
    for (int iStage = 0; iStage < inputs.bt.numStages; iStage++)
    {
        t[iStage] = inputs.t0 + inputs.bt.c[iStage] * dt;
        x[iStage] = inputs.x0;
        for (int jStage = 0; jStage < iStage; jStage++)
        {
            x[iStage] += dx[jStage] * inputs.bt.a(iStage,jStage);
        }
        if ((inputs.bt.rkCategory == EXPLICIT) || (inputs.bt.a(iStage,iStage) == (T)0.0))
        {
            if (f.functionType == IN_T_X_OUT_V)
            {
                v[iStage] = f.fe(t[iStage], x[iStage]);
            }
            else if (f.functionType == IN_T_X_OUT_V_JVX)
            {
                f.fej(t[iStage], x[iStage], v[iStage], Jvx);
            }
            else if (f.functionType == IN_T_X_V_OUT_R)
            {
                assert(false); // fixme: remove this once Newton-Krylov is completed
                outputs.success = false; // fixme: remove this once Newton-Krylov is completed
                return outputs; // fixme: remove this once Newton-Krylov is completed

                typename NewtonSolver<T,N>::NewtonKrylovFunction g = [f,iStage,t,x](const Vector<T,N> &vGuess, Vector<T,N> &r)
                {
                    r = f.fi(t[iStage], x[iStage], vGuess);
                };
                NewtonSolver<T,N>::solve(g, inputs.vGuess, v[iStage], outputs.success, inputs.newtonSettings);
                if (outputs.success == false)
                {
                    assert(inputs.bt.rkCategory != IMPLICIT);
                    return outputs;
                }
            }
            else if (f.functionType == IN_T_X_V_OUT_R_JRX_JRV)
            {
                typename NewtonSolver<T,N>::NewtonRaphsonFunction g = [f,iStage,t,x](const Vector<T,N> &vGuess, Vector<T,N> &r, Matrix<T,N,N> &Jrv)
                {
                    Matrix<T,N,N> Jrx;
                    f.fij(t[iStage], x[iStage], vGuess, r, Jrx, Jrv);
                };
                NewtonSolver<T,N>::solve(g, inputs.vGuess, v[iStage], outputs.success, inputs.newtonSettings);
                if (outputs.success == false)
                {
                    assert(inputs.bt.rkCategory != IMPLICIT);
                    return outputs;
                }
            }
        }
        else if (inputs.bt.rkCategory == DIRK)
        {
            if (f.functionType == IN_T_X_OUT_V)
            {
                assert(false); // fixme: remove this once Newton-Krylov is completed
                outputs.success = false; // fixme: remove this once Newton-Krylov is completed
                return outputs; // fixme: remove this once Newton-Krylov is completed

                typename NewtonSolver<T,N>::NewtonKrylovFunction g = [f,iStage,t,x,dt,inputs](const Vector<T,N> &vGuess, Vector<T,N> &r)
                {
                    Vector<T,N> vCalc, xGuess;
                    xGuess = x[iStage] + vGuess * (inputs.bt.a(iStage,iStage) * dt);
                    vCalc = f.fe(t[iStage], xGuess);
                    r = vCalc - vGuess;
                };
                NewtonSolver<T,N>::solve(g, inputs.vGuess, v[iStage], outputs.success, inputs.newtonSettings);
            }
            else if (f.functionType == IN_T_X_OUT_V_JVX)
            {
                typename NewtonSolver<T,N>::NewtonRaphsonFunction g = [f,iStage,t,x,dt,inputs](const Vector<T,N> &vGuess, Vector<T,N> &r, Matrix<T,N,N> &Jrv)
                {
                    Vector<T,N> vCalc, xGuess;
                    Matrix<T,N,N> Jvx;
                    xGuess = x[iStage] + vGuess * (inputs.bt.a(iStage,iStage) * dt);
                    f.fej(t[iStage], xGuess, vCalc, Jvx);
                    r = vCalc - vGuess;
                    Jrv = Jvx * (inputs.bt.a(iStage,iStage) * dt);
                    for (int i = 0; i < N; i++) Jrv(i,i) -= 1.0; // subtract off the identity matrix
                };
                NewtonSolver<T,N>::solve(g, inputs.vGuess, v[iStage], outputs.success, inputs.newtonSettings);
            }
            else if (f.functionType == IN_T_X_V_OUT_R)
            {
                assert(false); // fixme: remove this once Newton-Krylov is completed
                outputs.success = false; // fixme: remove this once Newton-Krylov is completed
                return outputs; // fixme: remove this once Newton-Krylov is completed

                typename NewtonSolver<T,N>::NewtonKrylovFunction g = [f,iStage,t,x,dt,inputs](const Vector<T,N> &vGuess, Vector<T,N> &r)
                {
                    Vector<T,N> xGuess = x[iStage] + vGuess * (inputs.bt.a(iStage,iStage) * dt);
                    r = f.fi(t[iStage], xGuess, vGuess);
                };
                NewtonSolver<T,N>::solve(g, inputs.vGuess, v[iStage], outputs.success, inputs.newtonSettings);
            }
            else if (f.functionType == IN_T_X_V_OUT_R_JRX_JRV)
            {
                typename NewtonSolver<T,N>::NewtonRaphsonFunction g = [f,iStage,t,x,dt,inputs](const Vector<T,N> &vGuess, Vector<T,N> &r, Matrix<T,N,N> &Jrv)
                {
                    Vector<T,N> vCalc, xGuess;
                    Matrix<T,N,N> Jrx;
                    xGuess = x[iStage] + vGuess * (inputs.bt.a(iStage,iStage) * dt);
                    f.fij(t[iStage], xGuess, vCalc, r, Jrx, Jrv);
                    Jrv += Jrx * (inputs.bt.a(iStage,iStage) * dt);
                };
                NewtonSolver<T,N>::solve(g, inputs.vGuess, v[iStage], outputs.success, inputs.newtonSettings);
            }
            if (outputs.success == false)
            {
                assert(false);
                return outputs;
            }
            x[iStage] += v[iStage] * (inputs.bt.a(iStage,iStage) * dt);
        }
        dx[iStage] = v[iStage] * dt;
        outputs.x1 += dx[iStage] * inputs.bt.b[iStage];
        outputs.x1e += dx[iStage] * inputs.bt.e[iStage];
    }
    outputs.v0 = v[0];
    outputs.v1 = v[inputs.bt.numStages - 1];
    return outputs;
}

template <typename T, int N> T RungeKutta<T,N>::calculateEmbeddedError(const TimeIntegration<T> &ti, const TimeSteppingParameters<T> &ts, const Vector<T,N> &x, const Vector<T,N> &xe)
{
    T xMagnitude = T(0);
    T delta = T(0);
    if ((ti.errorNormLevel <= 0.0) || (ti.errorNormLevel > 10.0))
    {
        for (int i = 0; i < x.size(); i++)
        {
            xMagnitude = std::fmax(xMagnitude, std::fabs(x[i]));
            delta = std::fmax(delta, std::fabs(x[i] - xe[i]));
        }
    }
    else
    {
        for (int i = 0; i < x.size(); i++)
        {
            xMagnitude += std::pow(std::fabs(x[i]), ti.errorNormLevel);
            delta += std::pow(std::fabs(x[i] - xe[i]), ti.errorNormLevel);
        }
        xMagnitude = std::pow(xMagnitude / T(x.size()), T(1) / ti.errorNormLevel);
        delta = std::pow(delta / T(x.size()), T(1) / ti.errorNormLevel);
    }
    T epsilon = std::fmax(ti.absoluteTolerance, ti.relativeTolerance * xMagnitude);
    T embError = delta / epsilon;
    return embError;
}

template <typename T, int N> void RungeKutta<T,N>::adaptTimeStep(const TimeIntegration<T> &ti, const TimeSteppingParameters<T> &ts, const std::deque<T> &embErrorHistory, const std::deque<T> &timeStepHistory, bool &repeat, T &dt)
{
    T adjustment = ti.safetyFactor;
    adjustment *= std::pow(embErrorHistory[0], -ts.alpha);
    adjustment *= std::pow(embErrorHistory[1], ts.beta);
    adjustment *= std::pow(embErrorHistory[2], -ts.gamma);
    adjustment *= std::pow(timeStepHistory[0] / timeStepHistory[1], ts.a);
    adjustment *= std::pow(timeStepHistory[1] / timeStepHistory[2], ts.b);
    adjustment = std::fmin(std::fmax(ti.minAdjustment, adjustment), ti.maxAdjustment);
    dt = timeStepHistory[0] * adjustment;
    repeat = (dt <= ti.minAdjustment * timeStepHistory[0]); // discontinuity was detected
    if (repeat == false) // continuous, but needs time step to be adapted
    {
        T dtMin = std::fmin(ti.minTimeStep, ti.maxTimeStep);
        T dtMax = std::fmax(ti.minTimeStep, ti.maxTimeStep);
        dt = std::fmin(std::fmax(dtMin, dt), dtMax);
    }
    else if (std::fabs(timeStepHistory[0]) > std::fabs(ti.minTimeStep)) // discontinuous
    {
        dt = timeStepHistory[0] * T(0.5); // note: use bisection to find the discontinuity
        if (std::fabs(dt) < std::fabs(ti.minTimeStep)) dt = ti.minTimeStep;
    }
    else  // discontinuous, but already up again minTimeStep, so don't repeat anymore
    {
        dt = ti.minTimeStep;
        repeat = false;
    }
}

template <typename T, int N> OdeSolution<T,N> RungeKutta<T,N>::integrateIVP(const RungeKuttaFunction<T,N> &f, const InitialConditions<T,N> &ic, const TimeIntegration<T> &ti_)
{
    TimeIntegration<T> ti = ti_;
    if (ti.startTime <= ti.finishTime)
    {
        ti.minTimeStep = std::fabs(ti.minTimeStep);
        ti.maxTimeStep = std::fabs(ti.maxTimeStep);
    }
    else
    {
        ti.minTimeStep = -std::fabs(ti.minTimeStep);
        ti.maxTimeStep = -std::fabs(ti.maxTimeStep);
    }
    SingleStepInputs<T,N> stepInputs;
    stepInputs.bt = ButcherTableau<T>(ti.rkMethod);
    stepInputs.newtonSettings = ti.newtonSettings;
    assert((ti.tsScheme == UNIFORM_STEPPING) || (stepInputs.bt.isAdaptive == true));
    SingleStepOutputs<T,N> stepOutputs;
    TimeSteppingParameters<T> ts;
    OdeSolution<T,N> solution;
    T dt, numSteps, embError;
    bool repeat;
    if (ti.tsScheme == UNIFORM_STEPPING)
    {
        numSteps = (int)std::ceil(std::fabs((ti.finishTime - ti.startTime) / ti.minTimeStep));
        dt = (ti.finishTime - ti.startTime) / (T)numSteps;
    }
    else
    {
        ts = typename TimeSteppingParameters<T>::TimeSteppingParameters(ti.tsScheme, stepInputs.bt.p, stepInputs.bt.pHat);
        dt = ti.minTimeStep;
    }
    std::deque<T> embErrorHistory(3, T(1));
    std::deque<T> timeStepHistory(3, ti.minTimeStep);
    solution.snapshots.resize(1);
    solution.snapshots[0].t = ti.startTime;
    solution.snapshots[0].x = ic.value;
    int iStep = 0;
    do
    {
        stepInputs.x0 = solution.snapshots[iStep].x;
        stepInputs.t0 = solution.snapshots[iStep].t;
        while (true)
        {
            if (ti.startTime <= ti.finishTime)
            {
                stepInputs.dt = std::fmin(dt, ti.finishTime - stepInputs.t0);
            }
            else
            {
                stepInputs.dt = std::fmax(dt, ti.finishTime - stepInputs.t0);
            }
            stepInputs.t1 = stepInputs.t0 + stepInputs.dt;
            stepOutputs = stepOdeSystem(f, stepInputs);
            embError = calculateEmbeddedError(ti, ts, stepOutputs.x1, stepOutputs.x1e);
            embErrorHistory.push_front(embError);
            timeStepHistory.push_front(stepInputs.dt);
            if (ti.tsScheme == UNIFORM_STEPPING) break; // if not adaptive, then break here
            adaptTimeStep(ti, ts, embErrorHistory, timeStepHistory, repeat, dt);
            if (repeat == false) break; // if not a repeated (discontinuity) step, then break here
            embErrorHistory.pop_front();
            timeStepHistory.pop_front();
        }
        if (stepOutputs.success == false)
        {
            solution.snapshots.clear();
            return solution;
        }
        solution.snapshots.resize(solution.snapshots.size() + 1);
        solution.snapshots[iStep + 1].t = stepInputs.t1;
        solution.snapshots[iStep + 1].x = stepOutputs.x1;
        solution.snapshots[iStep + 1].v = stepOutputs.v1;
        solution.snapshots[iStep].v = stepOutputs.v0;
        stepInputs.vGuess = stepOutputs.v0;
        iStep++;
    } while (std::fabs(stepInputs.t1 - ti.finishTime) > 1e-12);
    return solution;
}

template <typename T, int N> OdeSolution<T,N> RungeKutta<T,N>::solveIVP(const RungeKuttaFunction<T,N> &f, const InitialConditions<T,N> &ic, const TimeIntegration<T> &ti_)
{
    OdeSolution<T,N> solution;
    TimeIntegration<T> ti = ti_;
    if (ti.startTime > ti.finishTime) std::swap(ti.startTime, ti.finishTime);
    if (std::fabs(ic.time - ti.startTime) < 1e-12)
    {
        solution = integrateIVP(f, ic, ti);
    }
    else if (std::fabs(ic.time - ti.finishTime) < 1e-12)
    {
        std::swap(ti.startTime, ti.finishTime);
        solution = integrateIVP(f, ic, ti);
        std::reverse(solution.snapshots.begin(), solution.snapshots.end());
    }
    else if (ic.time < ti.startTime)
    {
        ti.startTime = ic.time;
        ti.finishTime = ti_.startTime;
        solution = integrateIVP(f, ic, ti);
        InitialConditions<T,N> ic0;
        ic0.time = ti_.startTime;
        for (int i = 0; i < solution.snapshots.back().x.size(); i++)
        {
            ic0.value[i] = solution.snapshots.back().x[i];
        }
        ti.startTime = ti_.startTime;
        ti.finishTime = ti_.finishTime;
        solution = integrateIVP(f, ic0, ti);
    }
    else if (ti.finishTime < ic.time)
    {
        ti.startTime = ic.time;
        ti.finishTime = ti_.finishTime;
        solution = integrateIVP(f, ic, ti);
        InitialConditions<T,N> ic1;
        ic1.time = ti_.finishTime;
        for (int i = 0; i < solution.snapshots.back().x.size(); i++)
        {
            ic1.value[i] = solution.snapshots.back().x[i];
        }
        ti.startTime = ti_.finishTime;
        ti.finishTime = ti_.startTime;
        solution = integrateIVP(f, ic1, ti);
        std::reverse(solution.snapshots.begin(), solution.snapshots.end());
    }
    else
    {
        ti.startTime = ic.time;
        ti.finishTime = ti_.startTime;
        solution = integrateIVP(f, ic, ti);
        std::reverse(solution.snapshots.begin(), solution.snapshots.end());
        solution.snapshots.pop_back();
        ti.startTime = ic.time;
        ti.finishTime = ti_.finishTime;
        OdeSolution<T,N> solution2 = integrateIVP(f, ic, ti);
        solution.snapshots.insert(solution.snapshots.end(), solution2.snapshots.begin(), solution2.snapshots.end());
    }
    return solution;
}

template <typename T, int N> OdeSolution<T,N> RungeKutta<T,N>::solveBVP(const RungeKuttaFunction<T,N> &f, const std::vector<BoundaryCondition<T>> &bc, const TimeIntegration<T> &ti_, const InitialConditions<T,N> &icGuess)
{
    qDebug() << "icGuessX= " << icGuess.time <<"  "<< icGuess.value[0] <<"  "<< icGuess.value[1];
    const bool USE_REDUCED_DOF = true;
    assert(bc.size() == icGuess.value.size());
    int n = bc.size();
    OdeSolution<T,N> solution;
    InitialConditions<T,N> ic = icGuess;
    //qDebug() << "ic0= " << ic.time <<"  "<< ic.value[0] <<"  "<< ic.value[1];
    std::vector<BoundaryCondition<T>> sortedBCs = bc;
    auto compare = [](const BoundaryCondition<T> &a, const BoundaryCondition<T> &b) {return a.time < b.time;};
    std::sort(sortedBCs.begin(), sortedBCs.end(), compare);
    TimeIntegration<T> ti = ti_;
    ti.startTime = std::fmin(std::fmin(ti.startTime,ti.finishTime), sortedBCs.front().time);
    ti.finishTime = std::fmax(std::fmax(ti.startTime,ti.finishTime), sortedBCs.back().time);
    std::vector<int> reducedToFullDofMapping;
    for (int iFull = 0; iFull < bc.size(); iFull++)
    {
        const BoundaryCondition<T> &b = bc[iFull];
        if ((USE_REDUCED_DOF == true) && (std::fabs(b.time - ic.time) < 1e-9))
        {
            ic.value[b.index] = b.value;
        }
        else
        {
            reducedToFullDofMapping.push_back(iFull);
        }
    }
    int nReduced = (int)reducedToFullDofMapping.size();
    //qDebug() << "ic1= " << ic.time <<"  "<< ic.value[0] <<"  "<< ic.value[1];
    typename NewtonSolver<T,N>::NewtonKrylovFunction g = [f, sortedBCs, ti, ic, n, nReduced, reducedToFullDofMapping]
                                                         (const Vector<T,N> &x, Vector<T,N> &r)
    {
        int i, iReduced, iFull, iSnapshot;
        r.rezero(nReduced);
        InitialConditions<T,N> icTest;
        icTest = ic;
        for (iReduced = 0; iReduced < nReduced; iReduced++)
        {
            iFull = reducedToFullDofMapping[iReduced];
            icTest.value[iFull] = x[iReduced];
        }
        //qDebug() << "icTest= " << icTest.time <<" | "<< icTest.value[0] <<"  "<< icTest.value[1] <<" | "<< x.size();
        OdeSolution<T,N> solution = solveIVP(f, icTest, ti);
        //solution.print();
        iReduced = 0, iSnapshot = 0;
        while ((iReduced < nReduced) && (iSnapshot < (int)solution.snapshots.size()))
        {
            iFull = reducedToFullDofMapping[iReduced];
            const BoundaryCondition<T> &bc = sortedBCs[iFull];
            const typename OdeSolution<T,N>::Snapshot snapshot = solution.snapshots[iSnapshot];
            if (std::fabs(bc.time - snapshot.t) < 1e-9)
            {
                r[iReduced] = snapshot.x[bc.index] - bc.value;
                iReduced++;
            }
            else
            {
                iSnapshot++;
            }
        }
        //qDebug() << "residual= " << r[0] <<"  "<< r[1];
    };
    bool success;
    Vector<T,N> xGuess, x;
    xGuess.rezero(n);
    for (int iReduced = 0; iReduced < nReduced; iReduced++)
    {
        int iFull = reducedToFullDofMapping[iReduced];
        xGuess[iReduced] = icGuess.value[iFull];
        qDebug() << "iReduced,iFull: " << iReduced <<"  "<< iFull;
    }
    qDebug() << "icGuessY= " << icGuess.time <<"  "<< icGuess.value[0] <<"  "<< icGuess.value[1];
    qDebug() << "xGuess.size= " << xGuess.size();
    qDebug() << "xGuess= " << xGuess[0] <<"  "<< xGuess[1];
    NewtonSolver<T,N>::solve(g, xGuess, x, success, ti.newtonSettings);
    qDebug() << "x: " << x[0] <<"  "<< x[1];
    if (success == false) {solution.snapshots.clear(); return solution;}
    for (int iReduced = 0; iReduced < nReduced; iReduced++)
    {
        int iFull = reducedToFullDofMapping[iReduced];
        ic.value[iFull] = x[iReduced];
    }
    solution = solveIVP(f, ic, ti);
    return solution;
}

template <typename T, int Ns, int Nc> OdeSolution<T,Ns,Nc> Pontryagin<T,Ns,Nc>::solveIVP(const GoverningFunction &f, const InstantaneousCostFunction &L,
                                                                                         const InitialConditions<T,Ns> &icx, const InitialConditions<T,Ns> &icu,
                                                                                         const TimeIntegration<T> &ti)
{
    const int N = (Ns >= 0 ? 2 * Ns : Ns);
    OdeSolution<T,Ns,Nc> solution;
    int ns = icx.value.size();
    int nc = icu.value.size();
    int n = 2 * ns;
    bool success;
    T Lcost, tPrev, LcostPrev;
    Vector<T,Ns> x, v, p, q, Lx;
    Vector<T,Nc> u, r, Lu;
    Matrix<T,Ns,Ns> Jvx;
    Matrix<T,Ns,Nc> Jvu;
    Matrix<T,Nc,Ns> JvuT;
    auto solveForControlVariables = [&](const T &t, const Vector<T,N> &y, const Vector<T,Nc> &u0, Vector<T,Nc> &u, bool &success)
    {
        x.resize(ns); for (int i = 0; i < ns; i++) x[i] = y[i];
        p.resize(ns); for (int i = 0; i < ns; i++) p[i] = y[ns+i];
        typename NewtonSolver<T,Nc>::NewtonKrylovFunction Hu = [&](const Vector<T,Nc>& u, Vector<T,Nc>& r)
        {
            f(t, x, u, v, Jvx, Jvu);
            L(t, x, u, Lcost, Lx, Lu);
            JvuT = Jvu.transposed();
            r = Lu + JvuT * p;
        };
        NewtonSolver<T,Nc>::solve(Hu, u0, u, success, ti.newtonSettings);
        //qDebug() << "x,p|u= " << x[0] <<"  "<< p[0] <<" | "<< u[0];
        // fixme: add special cases for over-constrained and under-constrained
    };
    RungeKuttaFunction<T,N> g(
        [&](const T &t, const Vector<T,N> &y)
        {
            Vector<T,N> w; w.resize(n);
            solveForControlVariables(t, y, icu.value, u, success);
            if (success == false) {w[0] = std::numeric_limits<T>::max(); return w;}
            q = Lx + Jvx * p;
            for (int i = 0; i < ns; i++) w[i] = v[i];
            for (int i = 0; i < ns; i++) w[ns+i] = -q[i];
            return w; // fixme: how to return errors?
        }
    );
    auto solveForLagrangeVariables = [&](const T &t, const Vector<T,Ns> &x, const Vector<T,Nc> &u, Vector<T,Ns> &p)
    {
        f(t, x, u, v, Jvx, Jvu);
        L(t, x, u, Lcost, Lx, Lu);
        JvuT = Jvu.transposed();
        if (nc == ns)
        {
            bool success;
            LinearSolver<T,Ns>::solve(JvuT, Lu, p, success, ti.newtonSettings.linearSolverSettings); // fixme: need success flag
            if (success == false) p.rezero(p.size());
        }
        else // over- or under-constrained
        {
            assert(false);
            //Vector<T,Ns> rhs = Jvu * Lu;
            //Matrix<T,Ns,Ns> mat = Jvu * JvuT; // fixme: need to implement matrix-matrix multiplication
            //AutomaticLinearSolver<T,Ns>::solve(mat, rhs, p);
        }
        for (int i = 0; i < p.size(); i++) p[i] = -p[i]; // fixme: need a unary minus
    };
    InitialConditions<T,N> ic;
    OdeSolution<T,N> xSolution;
    if (std::fabs(icx.time - icu.time) < 1e-9)
    {
        ic.time = icx.time;
        ic.value.rezero(n);
        for (int i = 0; i < ns; i++) ic.value[i] = icx.value[i];
        solveForLagrangeVariables(icx.time, icx.value, icu.value, p);
        for (int i = 0; i < ns; i++) ic.value[ns+i] = p[i];
        //qDebug() << "p= " << p[0];
        xSolution = RungeKutta<T,N>::solveIVP(g, ic, ti);
    }
    else
    {
        assert(false); // fixme: x and u initial conditions at different times is not allowed currently
    }

    // package up the coupled solution as a nice data structure
    solution.snapshots.resize(xSolution.snapshots.size());
    solution.integratedCost = T(0);
    for (int iSnapshot = 0; iSnapshot < (int)xSolution.snapshots.size(); iSnapshot++)
    {
        const typename OdeSolution<T,N>::Snapshot &xSnapshot = xSolution.snapshots[iSnapshot];
        typename OdeSolution<T,Ns,Nc>::Snapshot &snapshot = solution.snapshots[iSnapshot];
        snapshot.t = xSnapshot.t;
        for (int i = 0; i < ns; i++)
        {
            snapshot.x[i] = xSnapshot.x[i];
            snapshot.v[i] = xSnapshot.v[i];
            //qDebug() << "t,x,w|p,q: " << xSnapshot.t <<"  "<< xSnapshot.x[0] <<"  "<< xSnapshot.v[0] <<" | "<< xSnapshot.x[1] <<"  "<< xSnapshot.v[1];
            if (success == false)
            {
                solution.snapshots.clear();
                return solution;
            }
        }
        solveForControlVariables(xSnapshot.t, xSnapshot.x, icu.value, snapshot.u, success);
        L(snapshot.t, snapshot.x, snapshot.u, Lcost, Lx, Lu);
        if (iSnapshot > 0) solution.integratedCost += T(0.5) * (LcostPrev + Lcost) * (snapshot.t - tPrev); // fixme: trapezoidal rule is low-order
        tPrev = snapshot.t;
        LcostPrev = Lcost;
    }
    return solution;
}

template <typename T, int Ns, int Nc> OdeSolution<T,Ns,Nc> Pontryagin<T,Ns,Nc>::solveBVP(const GoverningFunction &f, const InstantaneousCostFunction &L,
                                                                                         const InitialConditions<T,Ns> &icx, const InitialConditions<T,Ns> &fcx,
                                                                                         const TimeIntegration<T> &ti, const InitialConditions<T,Nc> &uGuess)
{
    qDebug() << "Pontryagin<T,Ns,Nc>::solveBVP";
    // solve the Pontrygin controls ODE-BVP
    const int N = (Ns >= 0 ? 2 * Ns : Ns);
    OdeSolution<T,Ns,Nc> solution;
    int ns = icx.value.size();
    int nc = uGuess.value.size();
    //if (nc < ns) return solution; // note: insufficient control
    int n = 2 * ns;
    bool success;
    T Lcost, tPrev, LcostPrev;
    Vector<T,Ns> x, v, p, q, Lx;
    Vector<T,Nc> u, r, Lu;
    Matrix<T,Ns,Ns> Jvx;
    Matrix<T,Ns,Nc> Jvu;
    Matrix<T,Nc,Ns> JvuT;
    auto solveForControlVariables = [&](const T &t, const Vector<T,N> &y, const Vector<T,Nc> &u0, Vector<T,Nc> &u, bool &success)
    {
        x.resize(ns); for (int i = 0; i < ns; i++) x[i] = y[i];
        p.resize(ns); for (int i = 0; i < ns; i++) p[i] = y[ns+i];
        typename NewtonSolver<T,Nc>::NewtonKrylovFunction Hu = [&](const Vector<T,Nc>& u, Vector<T,Nc>& r)
        {
            f(t, x, u, v, Jvx, Jvu);
            L(t, x, u, Lcost, Lx, Lu);
            JvuT = Jvu.transposed();
            r = Lu + JvuT * p;
        };
        NewtonSolver<T,Nc>::solve(Hu, u0, u, success, ti.newtonSettings);
        //qDebug() << "t,x,p,u: " << t <<"  "<< x[0] <<"  "<< p[0] <<"  "<< u[0];
        // fixme: add special cases for over-constrained and under-constrained
    };
    RungeKuttaFunction<T,N> g(
        [&](const T &t, const Vector<T,N> &y)
        {
            Vector<T,N> w; w.resize(n);
            solveForControlVariables(t, y, uGuess.value, u, success);
            if (success == false) {w[0] = std::numeric_limits<T>::max(); return w;}
            q = Lx + Jvx * p;
            for (int i = 0; i < ns; i++) w[i] = v[i];
            for (int i = 0; i < ns; i++) w[ns+i] = -q[i];
            //qDebug() << "GGGGGG: " << t <<" | "<< y[0] <<"  "<< y[1] <<" | "<< w[0] <<"  "<< w[1] <<" | "<< u[0];
            return w; // fixme: how to return errors?
        }
    );
    auto solveForLagrangeVariables = [&](const T &t, const Vector<T,Ns> &x, const Vector<T,Nc> &u, Vector<T,Ns> &p)
    {
        f(t, x, u, v, Jvx, Jvu);
        L(t, x, u, Lcost, Lx, Lu);
        JvuT = Jvu.transposed();
        bool success;
        if (nc == ns)
        {
            LinearSolver<T,Ns>::solve(JvuT, Lu, p, success); // fixme: need success flag
            if (success == false) p.rezero(p.size());
        }
        else // over- or under-constrained
        {
            assert(false);
            //Vector<T,Ns> rhs = Jvu * Lu;
            //Matrix<T,Ns,Ns> mat = Jvu * JvuT; // fixme: need to implement matrix-matrix multiplication
            //AutomaticLinearSolver<T,Ns>::solve(mat, rhs, p);
        }
        for (int i = 0; i < p.size(); i++) p[i] = -p[i]; // fixme: need a unary minus
    };
    std::vector<BoundaryCondition<T>> bc(n);
    for (int i = 0; i < ns; i++) bc[i] = BoundaryCondition(icx.time, i, icx.value[i]);
    for (int i = 0; i < ns; i++) bc[ns+i] = BoundaryCondition(fcx.time, i, fcx.value[i]);
    InitialConditions<T,N> icGuess;
    icGuess.time = icx.time;
    icGuess.value.rezero(n);
    for (int i = 0; i < ns; i++) icGuess.value[i] = icx.value[i];
    solveForLagrangeVariables(icx.time, icx.value, uGuess.value, p);
    for (int i = 0; i < ns; i++) icGuess.value[ns+i] = p[i];
    //qDebug() << "icGuess= " << icGuess.value[0] <<"  "<< icGuess.value[1];
    OdeSolution<T,N> xSolution = RungeKutta<T,N>::solveBVP(g, bc, ti, icGuess);
    //qDebug() << "AAAAAAAAA: " << xSolution.snapshots.size();

    // package up the coupled solution as a nice data structure
    solution.snapshots.resize(xSolution.snapshots.size());
    solution.integratedCost = T(0);
    for (int iSnapshot = 0; iSnapshot < (int)xSolution.snapshots.size(); iSnapshot++)
    {
        const typename OdeSolution<T,N>::Snapshot &xSnapshot = xSolution.snapshots[iSnapshot];
        typename OdeSolution<T,Ns,Nc>::Snapshot &snapshot = solution.snapshots[iSnapshot];
        snapshot.t = xSnapshot.t;
        for (int i = 0; i < ns; i++)
        {
            snapshot.x[i] = xSnapshot.x[i];
            snapshot.v[i] = xSnapshot.v[i];
            solveForControlVariables(xSnapshot.t, xSnapshot.x, uGuess.value, snapshot.u, success);
            if (success == false)
            {
                solution.snapshots.clear();
                return solution;
            }
        }
        L(snapshot.t, snapshot.x, snapshot.u, Lcost, Lx, Lu);
        if (iSnapshot > 0) solution.integratedCost += T(0.5) * (LcostPrev + Lcost) * (snapshot.t - tPrev); // fixme: trapezoidal rule is low-order
        tPrev = snapshot.t;
        LcostPrev = Lcost;
    }
    return solution;
}

template <typename T> OdeSolution<T,1> RungeKutta1<T>::solveIVP1(const std::function<T(T,T)> &f, const T &t0, const T &x0, const TimeIntegration<T> &ti)
{
    RungeKuttaFunction<T,1> g(
        [f](const T &t, const Vector<T,1> &x)
        {
            Vector<T,1> v;
            v[0] = f(t, x[0]);
            return v;
        }
    );
    InitialConditions<T,1> ic;
    ic.time = t0;
    ic.value.resize(1);
    ic.value[0] = x0;
    OdeSolution<T,1> solution = RungeKutta<T,1>::solveIVP(g, ic, ti);
    return solution;
}

template <typename T> OdeSolution<T,1> RungeKutta1<T>::solveIVP1(const std::function<void(T,T,T&,T&)> &f, const T &t0, const T &x0, const TimeIntegration<T> &ti)
{
    RungeKuttaFunction<T,1> g(
        [f](const T &t, const Vector<T,1> &x, Vector<T,1> &v, Matrix<T,1,1> &Jvx)
        {
            f(t, x[0], v[0], Jvx(0,0));
        }
    );
    InitialConditions<T,1> ic;
    ic.time = t0;
    ic.value.resize(1);
    ic.value[0] = x0;
    OdeSolution<T,1> solution = RungeKutta<T,1>::solveIVP(g, ic, ti);
    return solution;
}

template <typename T> OdeSolution<T,1> RungeKutta1<T>::solveIVP1(const std::function<void(T,T,T,T&,T&,T&)> &f, const T &t0, const T &x0, const TimeIntegration<T> &ti)
{
    RungeKuttaFunction<T,1> g(
        [f](const T &t, const Vector<T,1> &x, const Vector<T,1> &v, Vector<T,1>& r, Matrix<T,1,1>& Jrx, Matrix<T,1,1>& Jrv)
        {
            f(t, x[0], v[0], r(0,0), Jrx(0,0), Jrv(0,0));
        }
        );
    InitialConditions<T,1> ic;
    ic.time = t0;
    ic.value.resize(1);
    ic.value[0] = x0;
    OdeSolution<T,1> solution = RungeKutta<T,1>::solveIVP(g, ic, ti);
    return solution;
}

template <typename T> std::vector<std::array<T,4>> RungeKutta1<T>::solveIVP2(const std::function<T(T,T,T)> &f, const T &t0, const T &x0, const T &v0, const TimeIntegration<T> &ti)
{
    std::vector<std::array<T,4>> specialSolution;
    RungeKuttaFunction<T,2> g(
        [f](const T &t, const Vector<T,2> &x)
        {
            Vector<T,2> v;
            v[0] = x[1];
            v[1] = f(t, x[0], x[1]);
            return v;
        }
    );
    InitialConditions<T,2> ic;
    ic.time = t0;
    ic.value.resize(2);
    ic.value[0] = x0;
    ic.value[1] = v0;
    OdeSolution<T,2> solution = RungeKutta<T,2>::solveIVP(g, ic, ti);
    specialSolution.resize(solution.snapshots.size());
    for (int iSnapshot = 0; iSnapshot < solution.snapshots.size(); iSnapshot++)
    {
        const typename OdeSolution<T,2>::Snapshot &snapshot = solution.snapshots[iSnapshot];
        std::array<T,4> &s = specialSolution[iSnapshot];
        s[0] = snapshot.t;
        s[1] = snapshot.x[0];
        s[2] = snapshot.x[1]; // note: this could be snapshot.v[0] as well
        s[3] = snapshot.v[1];
    }
    return specialSolution;
}

template <typename T> OdeSolution<T,1,1> Pontryagin1<T>::solveIVP(const std::function<void(T,T,T,T&,T&,T&)> &f,
                                                                  const std::function<void(T,T,T,T&,T&,T&)> &L,
                                                                  const T &t0, const T &x0, const T &u0,
                                                                  const TimeIntegration<T> &ti)
{
    typename Pontryagin<T,1,1>::GoverningFunction ff = [f](const T& t, const Vector<T,1>& x, const Vector<T,1>& u, Vector<T,1>& v, Matrix<T,1,1>& Jvx, Matrix<T,1,1>& Jvu)
    {
        v.resize(1);
        Jvx.resize(1,1);
        Jvu.resize(1,1);
        f(t, x[0], u[0], v[0], Jvx(0,0), Jvu(0,0));
    };
    auto LL = [L](const T& t, const Vector<T,1>& x, const Vector<T,1>& u, T& Lcost, Vector<T,1>& Lx, Vector<T,1>& Lu)
    {
        Lx.resize(1);
        Lu.resize(1);
        L(t, x[0], u[0], Lcost, Lx[0], Lu[0]);
    };
    InitialConditions<T,1> icx;
    icx.time = t0;
    icx.value.resize(1);
    icx.value[0] = x0;
    InitialConditions<T,1> icu;
    icu.time = t0;
    icu.value.resize(1);
    icu.value[0] = u0;
    OdeSolution<T,1,1> solution = Pontryagin<T,1,1>::solveIVP(ff, LL, icx, icu, ti);
    return solution;
}

template <typename T> OdeSolution<T,1,1> Pontryagin1<T>::solveBVP(const std::function<void(T,T,T,T&,T&,T&)> &f,
                                                                  const std::function<void(T,T,T,T&,T&,T&)> &L,
                                                                  const T &t0, const T &x0, const T &t1, const T &x1,
                                                                  const TimeIntegration<T> &ti, const T &u0Guess)
{
    typename Pontryagin<T,1,1>::GoverningFunction ff = [f](const T& t, const Vector<T,1>& x, const Vector<T,1>& u, Vector<T,1>& v, Matrix<T,1,1>& Jvx, Matrix<T,1,1>& Jvu)
    {
        v.resize(1);
        Jvx.resize(1,1);
        Jvu.resize(1,1);
        f(t, x[0], u[0], v[0], Jvx(0,0), Jvu(0,0));
    };
    auto LL = [L](const T& t, const Vector<T,1>& x, const Vector<T,1>& u, T& Lcost, Vector<T,1>& Lx, Vector<T,1>& Lu)
    {
        Lx.resize(1);
        Lu.resize(1);
        L(t, x[0], u[0], Lcost, Lx[0], Lu[0]);
    };
    InitialConditions<T,1> icx;
    icx.time = t0;
    icx.value.resize(1);
    icx.value[0] = x0;
    InitialConditions<T,1> fcx;
    fcx.time = t1;
    fcx.value.resize(1);
    fcx.value[0] = x1;
    InitialConditions<T,1> uGuess;
    uGuess.time = t0;
    uGuess.value.resize(1);
    uGuess.value[0] = u0Guess;
    OdeSolution<T,1,1> solution = Pontryagin<T,1,1>::solveBVP(ff, LL, icx, fcx, ti, uGuess);
    //qDebug() << "BBBBBBBBB: " << solution.snapshots.size();
    return solution;
}

} // XO

#endif // EXTRAORDINARY_H
