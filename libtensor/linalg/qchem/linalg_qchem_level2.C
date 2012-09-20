#include <qchem.h>
#include <libmathtools/general/blas_include.h>
#include "linalg_qchem_level2.h"

namespace libtensor {


const char *linalg_qchem_level2::k_clazz = "linalg";


void linalg_qchem_level2::copy_ij_ji(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double *c, size_t sic) {

    timings_base::start_timer("dcopy");
    if(ni < nj) {
        double *c1 = c;
        for(size_t i = 0; i < ni; i++, c1 += sic) {
            CL_DCOPY(nj, (double*)a + i, sja, c1, 1);
        }
    } else {
        const double *a1 = a;
        for(size_t j = 0; j < nj; j++, a1 += sja) {
            CL_DCOPY(ni, (double*)a1, 1, c + j, sic);
        }
    }
    timings_base::stop_timer("dcopy");
}


void linalg_qchem_level2::mul2_i_ip_p_x(
    void*,
    size_t ni, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("dgemv");
    CL_DGEMV('T', np, ni, d, (double*)a, sia, (double*)b, spb, 1.0, c, sic);
    timings_base::stop_timer("dgemv");
}


void linalg_qchem_level2::mul2_i_pi_p_x(
    void*,
    size_t ni, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("dgemv");
    CL_DGEMV('N', ni, np, d, (double*)a, spa, (double*)b, spb, 1.0, c, sic);
    timings_base::stop_timer("dgemv");
}


void linalg_qchem_level2::mul2_ij_i_j_x(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("dger");
    CL_DGER(nj, ni, d, (double*)b, sjb, (double*)a, sia, c, sic);
    timings_base::stop_timer("dger");
}


} // namespace libtensor
