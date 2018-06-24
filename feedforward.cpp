





// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <rendercheck_gl.h>



// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>




#include <stdlib.h>  



#include <thread>
#include <stdint.h>

#include <cassert>

#include "loaddata.h"
//#include <direct.h>




extern "C" void intial_cu(const properties& p1, int32_t *x, int32_t *y);
extern "C" int32_t RMSE_cal(int32_t *Chromosom, int32_t *chromosomSC);


unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

// random uniform distributions 

std::default_random_engine Sgen(seed);
std::uniform_int_distribution<int> Signdist(0, 1);
std::default_random_engine randgen(seed);





int main(void) {

	float RMSE = 0;

	int zero_or_one, sign;
	
	int min_value, max_value;
	int consequences_size, premises_size;
	int *consequences, *premises;

	// properties class
 	properties p1;

 	consequences_size = p1.consequences_size;
 	premises_size = p1.premises_size;

 	consequences = new int32_t[consequences_size];
	premises = new int32_t[premises_size];
	memset(consequences, 0, 4 * consequences_size);  // array is 32 bit so we need to multiply size to 4 
	memset(premises, 0, 4 * premises_size);

	// caret random numbers for premise values 

	min_value = 0.01*one; // according to properties.h file one is equal to pow(2,fraction_bits) 
	max_value = 1*one;

	// caret random numbers for premise values 
	std::uniform_int_distribution<int> randdist(min_value, max_value);

	/* we assumed premise parameters order according to mentioned paper
	   premises = [cente1r , center 2, ..... Sigma, sigma, sigma,....]

	   for consequence parameters :
	   consequences = [L01, L11, L21, L31, L41, L02, L12, L22, L32, L42....]

	   first the linear parameters for node number one and then linear 
	   parameters for node number 2 and so on..

	   we arranged parameters like this because it is convenient for randomized evolutionary algorithms
	   like genetic and PSO. there is an example of genetic algorithm in mentioned paper.

	*/

		for (int i = 0; i < premises_size; ++i) {
		
			sign = 1;
			zero_or_one = Signdist(Sgen);
			sign *= zero_or_one ? (1) : (-1); //  in last 3 lines we are trying to generate a random sign
			premises[i] = sign*randdist(randgen);

		
	}

			for (int i = 0; i < consequences_size; ++i) {
		
			sign = 1;
			zero_or_one = Signdist(Sgen);
			sign *= zero_or_one ? (1) : (-1); //  in last 3 lines we are trying to generate a random sign
			consequences[i] = sign*randdist(randgen);

		
	}


	loaddata data(p1);
	data.insert_data();

	intial_cu(p1, data.X,data.Y);

	RMSE = (RMSE_cal(consequences, premises))/one;
	
	cout << "RMSE :" << RMSE << endl;

	//system("pause");

return 0;

}

