#ifndef LIBTENSOR_DEFS_H
#define LIBTENSOR_DEFS_H

#include <cstddef>

/**	\brief Tensor library
	\ingroup libtensor
 **/
namespace libtensor {

/**	\brief Limits the maximum order of tensors
 **/
const size_t max_tensor_order = 6;

/**	\brief Library version (SVN revision number)
 **/
extern const char *version;

/**	\brief Namespace name
 **/
extern const char *g_ns;

}

#undef TENSOR_DEBUG
#undef LIBTENSOR_DEBUG
#ifdef DEBUG_CHECK
#define TENSOR_DEBUG
#define LIBTENSOR_DEBUG
#endif

#ifdef USE_MKL
#include <mkl_cblas.h>
#undef USE_BLAS
#endif
#ifdef USE_BLAS
#include <cblas.h>
#endif

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

/**	\defgroup libtensor_tests Tests
	\brief Unit tests of individual classes
	\ingroup libtensor
 **/

/**	\defgroup libtensor_tod Tensor operations (double)
	\brief Operations on tensors with real double precision elements
	\ingroup libtensor
 **/

#endif // LIBTENSOR_DEFS_H

