#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <functional>

//using namespace std;

std::vector<double> gen_rand_controller(int n)
{
  // First create an instance of an engine.
  std::random_device rnd_device;
  // Specify the engine and distribution.
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_real_distribution<double> dist {-10, 10};

  auto gen = [&dist, &mersenne_engine](){
             return dist(mersenne_engine);
             };

  std::vector<double> vec(n);
  std::generate(begin(vec), end(vec), gen);
  return vec;
}
int main()
{
//    // First create an instance of an engine.
//    random_device rnd_device;
//    // Specify the engine and distribution.
//    mt19937 mersenne_engine {rnd_device()};  // Generates random integers
//    uniform_real_distribution<double> dist {-1000, 1000};
//
//    auto gen = [&dist, &mersenne_engine](){
//                   return dist(mersenne_engine);
//               };
//
//    vector<double> vec(10);
//    generate(begin(vec), end(vec), gen);

	std::vector<double> vec = gen_rand_controller(10);
    // Optional
    for (auto i : vec) {
    	std::cout << i << " ";
    }


}
