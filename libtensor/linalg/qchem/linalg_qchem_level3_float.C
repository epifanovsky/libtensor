#include <liblas/liblas.h>
#include "linalg_qchem_level3_float.h"

namespace libtensor {


const char linalg_qchem_level3_float::k_clazz[] = "linalg";


void linalg_qchem_level3_float::mul2_ij_ip_jp_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const float *a, size_t sia,
    const float *b, size_t sjb,
    float *c, size_t sic,
    float d) {

    timings_base::start_timer("sgemm");
    INTEGER ni_ = ni, nj_ = nj, np_ = np;
    INTEGER sjb_ = sjb, sia_ = sia, sic_ = sic;
    float one = 1.0;
    sgemm("T", "N", &nj_, &ni_, &np_, &d, (float*)b, &sjb_, (float*)a,
        &sia_, &one, c, &sic_);
    timings_base::stop_timer("sgemm");
}


void linalg_qchem_level3_float::mul2_ij_ip_pj_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const float *a, size_t sia,
    const float *b, size_t spb,
    float *c, size_t sic,
    float d) {

    timings_base::start_timer("sgemm");
    INTEGER ni_ = ni, nj_ = nj, np_ = np;
    INTEGER spb_ = spb, sia_ = sia, sic_ = sic;
    float one = 1.0;
    sgemm("N", "N", &nj_, &ni_, &np_, &d, (float*)b, &spb_, (float*)a,
        &sia_, &one, c, &sic_);
    timings_base::stop_timer("sgemm");
}


void linalg_qchem_level3_float::mul2_ij_pi_jp_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const float *a, size_t spa,
    const float *b, size_t sjb,
    float *c, size_t sic,
    float d) {

    timings_base::start_timer("sgemm");
    INTEGER ni_ = ni, nj_ = nj, np_ = np;
    INTEGER sjb_ = sjb, spa_ = spa, sic_ = sic;
    float one = 1.0;
    sgemm("T", "T", &nj_, &ni_, &np_, &d, (float*)b, &sjb_, (float*)a,
        &spa_, &one, c, &sic_);
    timings_base::stop_timer("sgemm");
}


void linalg_qchem_level3_float::mul2_ij_pi_pj_x(
    void*,
    size_t ni, size_t nj, size_t np,
    const float *a, size_t spa,
    const float *b, size_t spb,
    float *c, size_t sic,
    float d) {

    timings_base::start_timer("sgemm");
    INTEGER ni_ = ni, nj_ = nj, np_ = np;
    INTEGER spb_ = spb, spa_ = spa, sic_ = sic;
    float one = 1.0;
    sgemm("N", "T", &nj_, &ni_, &np_, &d, (float*)b, &spb_, (float*)a,
        &spa_, &one, c, &sic_);
    timings_base::stop_timer("sgemm");
}


} // namespace libtensor
