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
//  isStable = check_state_space_stability(A);
  isStable = ((maxMagEigVal(A) <= 1)&&(maxMagEigVal(A) >= 0)) ? 1:0;
  Eigen::VectorXcd eivals = A.eigenvalues();
//  std::cout << "test: " << std::endl;
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
//  std::cout << "test2: " << std::endl;
  if((isStable == 1) && (status == true))
    return true;
  else
    return false;
}
void peak_output(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd C,
                 Eigen::MatrixXd D, Eigen::MatrixXd x0, double *out,
                 double yss, double u)
{
  double cur, pre, pos, peak;
  int i = 0;
  bool test = isEigPos(A);
  if(test)
  {
//	std::cout << "inside: " << test << std::endl;
    out[1] = yss;
    out[0] = i;
  }
  else
  {
//	std::cout << "outside: " << std::endl;
    pre = y_k(A, B, C, D, u, i, x0);
    cur = y_k(A, B, C, D, u, i+1, x0);
    pos = y_k(A, B, C, D, u, i+2, x0);
    out[1] = pre;
    out[0] = i;
    peak = pre;
    while((fabs(out[1]) <= fabs(peak)) && !(std::isnan(fabs(cur))))
    {
      if((out[1] != cur) && !(std::isnan(fabs(cur))))
      {
//    	std::cout << "outside1: " << std::endl;
//    	std::cout << "fabs(cur)=" << fabs(cur) << std::endl;
//    	std::cout << "fabs(pre)=" << fabs(pre) << std::endl;
//    	std::cout << "fabs(pos)=" << fabs(pos) << std::endl;
        if((fabs(cur) >= fabs(pos)) && (fabs(cur) >= fabs(pre)))
        {
          peak = cur;
//          std::cout << "outside2: " << std::endl;
        }
        if((out[1] != peak) && (isSameSign(yss, peak)) &&
           (fabs(peak) > fabs(out[1])))
        {
          out[0] = i+1;
          out[1] = peak;
//          std::cout << "outside3: " << std::endl;
        }
      }
      i++;
      pre = cur;
      cur = pos;
      pos = y_k(A, B, C, D, u, i+2, x0);
    }
  }
}
double c_bar(double yp, double yss, double lambmax, int kp)
{
  double cbar;
  cbar = (yp-yss)/(pow(lambmax, kp));
  return cbar;
}

double log_b(double base, double x)
{
  return static_cast<double> (log(x) / log(base));
}
int objective_function_ST(Eigen::MatrixXd K)
{
  double k_ss, x, yp, yss, u;
  double p = 5;
  double peakV[2];
  int kp, order = K.cols();
  Eigen::MatrixXd A(order, order), C(1, order);

  myA(0,0) = -0.5; myA(0,1) = 0.4;
  myA(1,0) = -0.4; myA(1,1) = -0.5;

  myB(0,0) = 0.0; myB(1,0) = 2.5;

  myC(0,0) = 0.0; myC(0,1) = 2.6;

  myD(0.0) = 0.0;

  myx0(0,0) = 0.0; myx0(1,0) = 0.0;

  A = myA - myB * K;
  C = myC - myD * K;

  double lambdaMax;
  lambdaMax = maxMagEigVal(A);
  u = 1.0;
  yss = y_ss(A, myB, C, myD, u);
  std::cout << "lambdaMax= " << lambdaMax << std::endl;
  std::cout << "order= " << order << std::endl;
  std::cout << "yss= " << yss << std::endl;
  peak_output(A, myB, C, myD, myx0, peakV, yss, u);
  yp = static_cast<double> (peakV[1]);
  kp = static_cast<int> (peakV[0]);
  double cbar = c_bar(yp, yss, lambdaMax, kp);
  x = fabs((p * yss) / (100 * cbar));
  k_ss = log_b(lambdaMax, x);
  return abs(ceil(k_ss)) + order;
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
    T obj = -objective_function_ST(K);
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
  double lambdaMax, u, yss, yp, cbar, temp, k_ss;
  int kp, order, kh, ksr = 6;
  double p = 5;
  double peakV[2];
  Eigen::MatrixXd K(1, x.size()), A(x.size(), x.size()), C(1, x.size());
  for(int i = 0; i < static_cast<int>(x.size()); i++)
  {
    K(0, i) = static_cast<double>(x[i]);
  }
  order = K.cols();
  p = 5;
  myA(0,0) = -0.5; myA(0,1) = 0.4;
  myA(1,0) = -0.4; myA(1,1) = -0.5;

  myB(0,0) = 0.0; myB(1,0) = 2.5;

  myC(0,0) = 0.0; myC(0,1) = 2.6;

  myD(0.0) = 0.0;

  myx0(0,0) = 0.0; myx0(1,0) = 0.0;

  A = myA - myB * K;
  C = myC - myD * K;

  lambdaMax = maxMagEigVal(A);
  u = 1.0;
  yss = y_ss(A, myB, C, myD, u);
  peak_output(A, myB, C, myD, myx0, peakV, yss, u);
  yp = static_cast<double> (peakV[1]);
  kp = static_cast<int> (peakV[0]);
  cbar = c_bar(yp, yss, lambdaMax, kp);
  temp = fabs((p * yss) / (100 * cbar));
  k_ss = log_b(lambdaMax, temp);
  kh = abs(ceil(k_ss)) + order;
//  return {K(0,0)*K(0,1)+K(0,0)-K(0,1)+1.5,10-K(0,0)*K(0,1)};
//  return {-objective_function_ST(K), objective_function_ST(K)-ksr, -check_state_space_stability(A)+1};
//  return {-check_state_space_stability(A)+1};
  return {-lambdaMax, lambdaMax-1, -kh, kh-ksr};
  //return {-maxMagEigVal(A), maxMagEigVal(A)-1};
//  return {-check_state_space_stability(A), check_state_space_stability(A)-1};
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
