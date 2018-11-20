//=================================================================================================
//                    Copyright (C) 2017 Olivier Mallet - All Rights Reserved                      
//=================================================================================================

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <stack>
#include <streambuf>
#include <string>
#include <vector>

/* eigen dependencies */
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/Polynomials>
#include <unsupported/Eigen/MatrixFunctions>

#include "Galgo.hpp"

/* deviation */
double deviation = 0.000000001;
#define MAXNUMBADPEAKS (2)
#define MAXNUMGRADS (10)
#define MINDIFFYSS (0.001)

Eigen::MatrixXd myA(2, 2);
Eigen::MatrixXd myB(2, 1);
Eigen::MatrixXd myC(1, 2);
Eigen::MatrixXd myD(1, 1);
Eigen::MatrixXd myx0(2, 1);

double y_k(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C,
           Eigen::MatrixXd D, double u, int k, Eigen::MatrixXd x0)
{
  int m;
  Eigen::MatrixXd y;
  y = C * A.pow(k) * x0;
  for(m = 0; m <= (k - 1); m++)
  {
    y += (C * A.pow(k - m - 1) * B * u) + D * u;
  }
  return y(0, 0);
}
double y_ss(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C,
            Eigen::MatrixXd D, double u)
{
  double yss;
  Eigen::MatrixXd AUX;
  Eigen::MatrixXd AUX2;
  Eigen::MatrixXd AUX3;
  Eigen::MatrixXd Id;
  // get the expression y_ss=(C(I-A)^(-1)B+D)u
  Id.setIdentity(A.rows(), A.cols());
  AUX = Id - A;
  AUX3 = AUX.inverse();
  AUX2 = (C * AUX3 * B + D);
  yss = AUX2(0, 0) * u;
  return yss;
}

bool isSameSign(double a, double b)
{
  if(((a >= 0) && (b >= 0)) || ((a <= 0) && (b <= 0)))
    return true;
  else
    return false;
}

double cplxMag(double real, double imag)
{
  return sqrt(real * real + imag * imag);
}
double maxMagEigVal(Eigen::MatrixXd A)
{
  double _real, _imag;
  double maximum = 0, aux;
  int i;
  Eigen::VectorXcd eivals = A.eigenvalues();
  for(i = 0; i < A.rows(); i++)
  {
    _real = eivals[i].real();
    _imag = eivals[i].imag();
    aux = cplxMag(_real, _imag);
    if(aux > maximum)
    {
      maximum = aux;
    }
  }
  return maximum;
}

bool isEigPos(Eigen::MatrixXd A)
{
  int isStable, i;
  std::complex<double> lambda;
  bool status;
  isStable = ((maxMagEigVal(A) <= 1)&&(maxMagEigVal(A) >= 0)) ? 1:0;
  Eigen::VectorXcd eivals = A.eigenvalues();
  for(i = 0; i < A.rows(); i++)
  {
    lambda = eivals[i];
    if(lambda.real() >= 0)
      status = true;
    else
    {
      status = false;
      break;
    }
  }
  if((isStable == 1) && (status == true))
    return true;
  else
    return false;
}

void peak_output(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C,
                 Eigen::MatrixXd D, Eigen::MatrixXd x0, double *out,
                 double yss, double u)
{
  double cur, pre, pos, greatest, peak, cmp, o;
  int i = 0, numBadPeaks = 0, firstGradSampleIdx, lastPeakIdx, lastGrad = 1, grad = 0;
  double lastPeak, firstGradSample;
  firstGradSample = y_k(A, B, C, D, u, i, x0);
  lastPeak = y_k(A, B, C, D, u, i, x0);
  lastPeakIdx = i;
  while(1)
  {
    if(fabs(y_k(A, B, C, D, u, i+1, x0)) >= fabs(y_k(A, B, C, D, u, i, x0)))
    {
      grad = (grad > 0)?(grad + 1):1;
      if(fabs(y_k(A, B, C, D, u, i+1, x0)) != fabs(y_k(A, B, C, D, u, i, x0)))
      {
        firstGradSample = y_k(A, B, C, D, u, i+1, x0);
        firstGradSampleIdx = i + 1;
      }
    }
    else
    {
      grad = (grad < 0)?(grad - 1):-1;
    }
    if((lastGrad > 0) && (grad < 0))
    {
      if(fabs(firstGradSample) <= fabs(lastPeak))
      {
        ++numBadPeaks;
        if(numBadPeaks > MAXNUMBADPEAKS)
        {
          break;
        }
      }
      else
      {
        lastPeak = firstGradSample;
        lastPeakIdx = firstGradSampleIdx;
      }
    }
    else if((grad > MAXNUMGRADS) && (fabs((y_k(A, B, C, D, u, i+1, x0) - yss)/yss) < MINDIFFYSS))
    {
      if(fabs(yss) > fabs(lastPeak))
      {
        lastPeak = yss;
        lastPeakIdx = 0;
      }
      break;
    }
    lastGrad = grad;
    ++i;
  }
  out[0] = lastPeakIdx;
  out[1] = lastPeak;
}

int objective_function_OS(Eigen::MatrixXd K)
{
  double peakV[2];
  double yss, yp, mp, _PO, u;
  int order = K.cols();
  Eigen::MatrixXd A(order, order), C(1, order);

  u = 1.0;

  myA(0,0) = -0.5; myA(0,1) = 0.4;
  myA(1,0) = -0.4; myA(1,1) = -0.5;

  myB(0,0) = 0.0; myB(1,0) = 2.5;

  myC(0,0) = 0.0; myC(0,1) = 2.6;

  myD(0.0) = 0.0;

  myx0(0,0) = 0.0; myx0(1,0) = 0.0;

  A = myA - myB * K;
  C = myC - myD * K;

  yss = y_ss(A, myB, C, myD, u);
  std::cout << "yss=" << yss << std::endl;
  peak_output(A, myB, C, myD, myx0, peakV, yss, u);
  yp = static_cast<double> (peakV[1]);
  std::cout << "yp=" << yp << std::endl;
  mp = cplxMag(cplxMag(yp,0)-cplxMag(yss,0),0);
  std::cout << "mp=" << mp << std::endl;
  _PO = cplxMag((mp/yss),0);
  std::cout << "PO=" << _PO*100 << std::endl;
  return _PO;
}

// objective class example
template <typename T>
class MyObjective
{
public:

  // objective function example : Rosenbrock function
  // minimizing f(x,y) = (1 - x)^2 + 100 * (y - x^2)^2
//  static std::vector<T> Objective(const Eigen::MatrixXd K)
  static std::vector<T> Objective(const std::vector<T>& x)
  {
//     T obj = -(pow(1-x[0],2)+100*pow(x[1]-x[0]*x[0],2));
    Eigen::MatrixXd K(1, x.size());
    for(int i = 0; i < static_cast<int>(x.size()); i++)
    {
      K(0,i) = static_cast<double>(x[i]);
    }
//	  T obj = my_function(K);
    T obj = -objective_function_OS(K);
    return {obj};
  }
  // NB: GALGO maximize by default so we will maximize -f(x,y)
};

// constraints example:
// 1) x * y + x - y + 1.5 <= 0
// 2) 10 - x * y <= 0
template <typename T>
std::vector<T> MyConstraint(const std::vector<T>& x)
{
  int order = static_cast<int> (x.size());
  Eigen::MatrixXd K(1, order), A(order, order), C(1, order);
  int POr = 6/100, kp;
  double yss, yp, mp, _PO, u;
  double peakV[2];
  u = 1.0;
  for(int i = 0; i < order; i++)
  {
    K(0, i) = static_cast<double>(x[i]);
  }

  myA(0,0) = -0.5; myA(0,1) = 0.4;
  myA(1,0) = -0.4; myA(1,1) = -0.5;

  myB(0,0) = 0.0; myB(1,0) = 2.5;

  myC(0,0) = 0.0; myC(0,1) = 2.6;

  myD(0.0) = 0.0;

  myx0(0,0) = 0.0; myx0(1,0) = 0.0;

  A = myA - myB * K;
  C = myC - myD * K;

  yss = y_ss(A, myB, C, myD, u);
  peak_output(A, myB, C, myD, myx0, peakV, yss, u);
  yp = static_cast<double> (peakV[1]);
  mp = cplxMag(cplxMag(yp,0)-cplxMag(yss,0),0);
  _PO = cplxMag((mp/yss),0);
//  return {K(0,0)*K(0,1)+K(0,0)-K(0,1)+1.5,10-K(0,0)*K(0,1)};
//  return {-k_bar(K), k_bar(K)-ksr, -check_state_space_stability(A), check_state_space_stability(A)-1};
  return {-_PO, _PO-POr, -maxMagEigVal(A), maxMagEigVal(A)-1};
//  -check_state_space_stability(A)
}
// NB: a penalty will be applied if one of the constraints is > 0 
// using the default adaptation to constraint(s) method

int main()
{
   // initializing parameters lower and upper bounds
   // an initial value can be added inside the initializer list after the upper bound
   std::vector<double> p1 = {0.0, 0.5};
//   gen_rand_controller(2, 0.0, 1.0);
   std::vector<double> p2 = {0.0, 1.0};
   galgo::Parameter<double> par1(p1);
   galgo::Parameter<double> par2(p2);
   // here both parameter will be encoded using 16 bits the default value inside the template declaration
   // this value can be modified but has to remain between 1 and 64

   // initiliazing genetic algorithm
   galgo::GeneticAlgorithm<double> ga(MyObjective<double>::Objective,100,50,true,par1,par2);

   // setting constraints
   ga.Constraint = MyConstraint;

   // running genetic algorithm
   ga.run();
}
