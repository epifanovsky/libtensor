#include <liblas/liblas.h>
#include "linalg_qchem_level3.h"

namespace libtensor {


const char linalg_qchem_level3::k_clazz[] = "linalg";


void linalg_qchem_level3::mul2_ij_ip_jp_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("dgemm");
    INTEGER ni_ = ni, nj_ = nj, np_ = np;
    INTEGER sjb_ = sjb, sia_ = sia, sic_ = sic;
    double one = 1.0;
    dgemm("T", "N", &nj_, &ni_, &np_, &d, (double*)b, &sjb_, (double*)a,
        &sia_, &one, c, &sic_);
    timings_base::stop_timer("dgemm");
}


void linalg_qchem_level3::mul2_ij_ip_pj_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("dgemm");
    INTEGER ni_ = ni, nj_ = nj, np_ = np;
    INTEGER spb_ = spb, sia_ = sia, sic_ = sic;
    double one = 1.0;
    dgemm("N", "N", &nj_, &ni_, &np_, &d, (double*)b, &spb_, (double*)a,
        &sia_, &one, c, &sic_);
    timings_base::stop_timer("dgemm");
}


void linalg_qchem_level3::mul2_ij_pi_jp_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("dgemm");
    INTEGER ni_ = ni, nj_ = nj, np_ = np;
    INTEGER sjb_ = sjb, spa_ = spa, sic_ = sic;
    double one = 1.0;
    dgemm("T", "T", &nj_, &ni_, &np_, &d, (double*)b, &sjb_, (double*)a,
        &spa_, &one, c, &sic_);
    timings_base::stop_timer("dgemm");
}


void linalg_qchem_level3::mul2_ij_pi_pj_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("dgemm");
    INTEGER ni_ = ni, nj_ = nj, np_ = np;
    INTEGER spb_ = spb, spa_ = spa, sic_ = sic;
    double one = 1.0;
    dgemm("N", "T", &nj_, &ni_, &np_, &d, (double*)b, &spb_, (double*)a,
        &spa_, &one, c, &sic_);
    timings_base::stop_timer("dgemm");
}


} // namespace libtensor
