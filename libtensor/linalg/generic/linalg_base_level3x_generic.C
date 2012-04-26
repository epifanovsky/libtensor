#include "linalg_base_level3x_generic.h"

namespace libtensor {


void linalg_base_level3x_generic::ij_pji_p_x(
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sja, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    for(size_t p = 0; p < np; p++)
    for(size_t i = 0; i < ni; i++)
    for(size_t j = 0; j < nj; j++) {
        c[i * sic + j] += d * a[p * spa + j * sja + i] * b[p * spb];
    }
}


} // namespace libtensor
