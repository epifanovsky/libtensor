#ifndef LIBTENSOR_DEFS_H
#define LIBTENSOR_DEFS_H

#include <cstddef>

/**	\brief Tensor library
	\ingroup libtensor
 **/
namespace libtensor {

/**	\brief Namespace name
 **/
extern const char *g_ns;

}

#ifdef __MINGW32__
#include <cstdlib>
inline void srand48(long seed) { srand(seed); }
inline double drand48() { return (double(rand())/RAND_MAX); }
inline long lrand48() { return rand(); }
#endif

/**	\defgroup libtensor Tensor library
 **/

/**	\defgroup libtensor_core Core components
	\ingroup libtensor
 **/

/**	\defgroup libtensor_mp Parallel processing components
 	\brief Basic components for parallel processing in libtensor.
	\ingroup libtensor
 **/

/**	\defgroup libtensor_tod Tensor operations (double)
	\brief Operations on tensors with real double precision elements
	\ingroup libtensor
 **/

/**	\defgroup libtensor_btod Block %tensor operations (double)
	\brief Operations on block tensors with real double precision elements
	\ingroup libtensor
 **/

/**	\defgroup libtensor_iface Block tensor interface
	\brief Easy to use interface to implement equations with block tensors.
	\ingroup libtensor
 **/

#endif // LIBTENSOR_DEFS_H

