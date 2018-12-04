//=================================================================================================
//                    Copyright (C) 2017 Olivier Mallet - All Rights Reserved                      
//=================================================================================================

/* eigen dependencies */
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/Polynomials>
#include <unsupported/Eigen/MatrixFunctions>

#include "Galgo.hpp"

double my_function(Eigen::MatrixXd K)
{
 return -(pow(1-K(0,0),2)+100*pow(K(0,1)-K(0,0)*K(0,0),2));
}

// objective class example
template <typename T>
class MyObjective
{
public:
   // objective function example : Rosenbrock function
   // minimizing f(x,y) = (1 - x)^2 + 100 * (y - x^2)^2
   static std::vector<T> Objective(const std::vector<T>& x)
   {
      Eigen::MatrixXd K(1, x.size());
	  for(int i = 0; i < static_cast<int>(x.size()); i++)
	  {
	    K(0,i) = static_cast<double>(x[i]);
	  }
      T obj = my_function(K);
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
	Eigen::MatrixXd K(1, x.size()), A(x.size(), x.size());
	for(int i = 0; i < static_cast<int>(x.size()); i++)
	{
	  K(0, i) = static_cast<double>(x[i]);
	}
  return {K(0,0)*K(0,1)+K(0,0)-K(0,1)+1.5,10-K(0,0)*K(0,1)};
	//return {-my_function(K), my_function(K)-2.0};
}
// NB: a penalty will be applied if one of the constraints is > 0 
// using the default adaptation to constraint(s) method

int main()
{
   // initializing parameters lower and upper bounds
   // an initial value can be added inside the initializer list after the upper bound
   galgo::Parameter<double> par1({0.0,1.0});
   galgo::Parameter<double> par2({0.0,13.0});
   // here both parameter will be encoded using 16 bits the default value inside the template declaration
   // this value can be modified but has to remain between 1 and 64

   // initiliazing genetic algorithm
   galgo::GeneticAlgorithm<double> ga(MyObjective<double>::Objective,100,50,true,par1,par2);

   // setting constraints
   ga.Constraint = MyConstraint;

   // running genetic algorithm
   ga.run();
}
