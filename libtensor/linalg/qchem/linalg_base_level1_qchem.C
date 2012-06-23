#include <qchem.h>
#include <libmathtools/general/blas_include.h>
#include "linalg_base_level1_qchem.h"

namespace libtensor {


void linalg_base_level1_qchem::mul_i_i_x_p11(size_t ni,
    const double *a, double b, double *c) {

    #pragma ivdep
    for(size_t i = 0; i < ni; i++) {
        c[i] += a[i];
    }
}


void linalg_base_level1_qchem::mul_i_i_x_pxx(size_t ni,
    const double *a, size_t sia, double b, double *c, size_t sic) {

    for(size_t i = 0; i < ni; i++) {
        c[sic * i] += a[sia * i];
    }
}


void linalg_base_level1_qchem::mul_i_i_x_m11(size_t ni,
    const double *a, double b, double *c) {

    #pragma ivdep
    for(size_t i = 0; i < ni; i++) {
        c[i] -= a[i];
    }
}


void linalg_base_level1_qchem::mul_i_i_x_mxx(size_t ni,
    const double *a, size_t sia, double b, double *c, size_t sic) {

    for(size_t i = 0; i < ni; i++) {
        c[sic * i] -= a[sia * i];
    }
}


} // namespace libtensor
