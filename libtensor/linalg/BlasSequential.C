#include "BlasSequential.h"

#ifdef HAVE_OPENBLAS
extern "C" int openblas_get_num_threads();
extern "C" void openblas_set_num_threads(int num_threads);
#elif HAVE_MKL
#include <mkl.h>
#define openblas_get_num_threads mkl_get_max_threads
#define openblas_set_num_threads mkl_set_num_threads
#else  // no openblas, no MKL
namespace {
int openblas_get_num_threads() { return 1; }
void openblas_set_num_threads(int) {}
}  // namespace
#endif

namespace libtensor {

BlasSequential::BlasSequential() : blas_num_threads(openblas_get_num_threads()) {
  openblas_set_num_threads(1);
}

BlasSequential::~BlasSequential() { openblas_set_num_threads(blas_num_threads); }

}  // namespace libtensor

