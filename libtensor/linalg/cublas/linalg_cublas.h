#ifndef LIBTENSOR_LINALG_CUBLAS_H
#define LIBTENSOR_LINALG_CUBLAS_H

#include "../generic/linalg_base_lowlevel.h"
#include "../generic/linalg_base_memory_generic.h"
#include "linalg_level1_cublas.h"
#include "linalg_level2_cublas.h"
#include "linalg_level3_cublas.h"

namespace libtensor {


/** \brief Linear algebra implementation based on NVIDIA CUDA BLAS (cuBLAS)

    \ingroup libtensor_linalg
 **/
struct linalg_cublas :
    public linalg_base_lowlevel<
        linalg_base_memory_generic,
        linalg_level1_cublas,
        linalg_level2_cublas,
        linalg_level3_cublas>
{ };


} // namespace libtensor

#endif // LIBTENSOR_LINALG_CUBLAS_H
