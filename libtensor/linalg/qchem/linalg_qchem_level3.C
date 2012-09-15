#include <qchem.h>
#include <libmathtools/general/blas_include.h>
#include "linalg_qchem_level3.h"

namespace libtensor {


const char *linalg_qchem_level3::k_clazz = "linalg";


void linalg_qchem_level3::mul2_ij_ip_jp_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemm");
    CL_DGEMM('T', 'N', nj, ni, np, d, (double*)b, sjb, (double*)a,
        sia, 1.0, c, sic);
    stop_timer("dgemm");
}


void linalg_qchem_level3::mul2_ij_ip_pj_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemm");
    CL_DGEMM('N', 'N', nj, ni, np, d, (double*)b, spb, (double*)a,
        sia, 1.0, c, sic);
    stop_timer("dgemm");
}


void linalg_qchem_level3::mul2_ij_pi_jp_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemm");
    CL_DGEMM('T', 'T', nj, ni, np, d, (double*)b, sjb, (double*)a,
        spa, 1.0, c, sic);
    stop_timer("dgemm");
}


void linalg_qchem_level3::mul2_ij_pi_pj_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    start_timer("dgemm");
    CL_DGEMM('N', 'T', nj, ni, np, d, (double*)b, spb, (double*)a,
        spa, 1.0, c, sic);
    stop_timer("dgemm");
}


} // namespace libtensor
