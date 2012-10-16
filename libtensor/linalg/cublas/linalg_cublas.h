#ifndef LIBTENSOR_LINALG_CUBLAS_H
#define LIBTENSOR_LINALG_CUBLAS_H

#include <cublas_v2.h>
#include "linalg_level1_cublas.h"
#include "linalg_level2_cublas.h"
#include "linalg_level3_cublas.h"

namespace libtensor {


/** \brief Linear algebra implementation based on NVIDIA CUDA BLAS (cuBLAS)

    \ingroup libtensor_linalg
 **/
class linalg_cublas :
    public linalg_level1_cublas,
    public linalg_level2_cublas,
    public linalg_level3_cublas {

public:
    typedef double element_type; //!< Data type
    typedef cublasHandle_t device_context_type; //!< Device context type
    typedef cublasHandle_t &device_context_ref; //!< Reference to device context

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_CUBLAS_H
