#include <liblas/liblas.h>
#include "linalg_qchem_level2_double.h"

namespace libtensor {


const char linalg_qchem_level2_double::k_clazz[] = "linalg";


void linalg_qchem_level2_double::copy_ij_ji(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double *c, size_t sic) {

    timings_base::start_timer("dcopy");
    INTEGER ni_ = ni, nj_ = nj, one = 1;
    INTEGER sja_ = sja, sic_ = sic;
    if(ni < nj) {
        double *c1 = c;
        for(size_t i = 0; i < ni; i++, c1 += sic) {
            dcopy(&nj_, (double*)a + i, &sja_, c1, &one);
        }
    } else {
        const double *a1 = a;
        for(size_t j = 0; j < nj; j++, a1 += sja) {
            dcopy(&ni_, (double*)a1, &one, c + j, &sic_);
        }
    }
    timings_base::stop_timer("dcopy");
}


void linalg_qchem_level2_double::mul2_i_ip_p_x(
    void*,
    size_t ni, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("dgemv");
    INTEGER np_ = np, ni_ = ni, sia_ = sia, spb_ = spb, sic_ = sic;
    double one = 1.0;
    dgemv("T", &np_, &ni_, &d, (double*)a, &sia_, (double*)b, &spb_,
        &one, c, &sic_);
    timings_base::stop_timer("dgemv");
}


void linalg_qchem_level2_double::mul2_i_pi_p_x(
    void*,
    size_t ni, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("dgemv");
    INTEGER ni_ = ni, np_ = np, spa_ = spa, spb_ = spb, sic_ = sic;
    double one = 1.0;
    dgemv("N", &ni_, &np_, &d, (double*)a, &spa_, (double*)b, &spb_,
        &one, c, &sic_);
    timings_base::stop_timer("dgemv");
}


void linalg_qchem_level2_double::mul2_ij_i_j_x(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("dger");
    INTEGER ni_ = ni, nj_ = nj, sia_ = sia, sjb_ = sjb, sic_ = sic;
    dger(&nj_, &ni_, &d, (double*)b, &sjb_, (double*)a, &sia_, c, &sic_);
    timings_base::stop_timer("dger");
}


} // namespace libtensor
