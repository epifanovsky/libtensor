#include <mkl.h>
#ifdef HAVE_MKL_DOMATCOPY
#include <mkl_trans.h>
#include <cstring>
#endif // HAVE_MKL_DOMATCOPY
#include "linalg_mkl_level2.h"

namespace libtensor {


template<>
const char linalg_mkl_level2<double>::k_clazz[] = "mkl";
template<>
const char linalg_mkl_level2<float>::k_clazz[] = "mkl";


template<>
void linalg_mkl_level2<double>::add1_ij_ij_x(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

#ifdef HAVE_MKL_DOMATCOPY
    if(ni * nj <= 256 * 256) {
        timings_base::start_timer("mkl_domatadd");
        double t[256 * 256];
        mkl_domatadd('R', 'N', 'N', ni, nj, b, a, sia, 1.0, c, sic, t, nj);
        if(sic == nj) {
            memcpy(c, t, sizeof(double) * ni * nj);
        } else {
            for(size_t i = 0; i < ni; i++) {
                memcpy(c + i * sic, t + i * nj, sizeof(double) * nj);
            }
        }
        timings_base::stop_timer("mkl_domatadd");
    } else
#endif // HAVE_MKL_DOMATCOPY
    {
        timings_base::start_timer("daxpy");
        for(size_t i = 0; i < ni; i++) {
            cblas_daxpy(nj, b, a + i * sia, 1, c + i * sic, 1);
        }
        timings_base::stop_timer("daxpy");
    }
}


template<>
void linalg_mkl_level2<double>::add1_ij_ji_x(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double b,
    double *c, size_t sic) {

#ifdef HAVE_MKL_DOMATCOPY
    if(ni * nj <= 256 * 256 && sic == nj) {
        timings_base::start_timer("mkl_domatadd");
        double t[256 * 256];
        mkl_domatadd('R', 'T', 'N', ni, nj, b, a, sja, 1.0, c, sic, t, nj);
        memcpy(c, t, sizeof(double) * ni * nj);
        timings_base::stop_timer("mkl_domatadd");
    } else
#endif // HAVE_MKL_DOMATCOPY
    {
        timings_base::start_timer("daxpy");
        for(size_t i = 0; i < ni; i++) {
            cblas_daxpy(nj, b, a + i, sja, c + i * sic, 1);
        }
        timings_base::stop_timer("daxpy");
    }
}


template<>
void linalg_mkl_level2<double>::copy_ij_ij_x(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

#ifdef HAVE_MKL_DOMATCOPY
    timings_base::start_timer("mkl_domatcopy");
    mkl_domatcopy('R', 'N', ni, nj, b, a, sia, c, sic);
    timings_base::stop_timer("mkl_domatcopy");
#else // HAVE_MKL_DOMATCOPY
    timings_base::start_timer("dcopy+dscal");
    for(size_t i = 0; i < ni; i++) {
        cblas_dcopy(nj, a + i * sia, 1, c + i * sic, 1);
        cblas_dscal(nj, b, c + i * sic, 1);
    }
    timings_base::stop_timer("dcopy+dscal");
#endif // HAVE_MKL_DOMATCOPY
}


template<>
void linalg_mkl_level2<double>::copy_ij_ji(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double *c, size_t sic) {

#ifdef HAVE_MKL_DOMATCOPY
    timings_base::start_timer("mkl_domatcopy");
    mkl_domatcopy('R', 'T', nj, ni, 1.0, a, sja, c, sic);
    timings_base::stop_timer("mkl_domatcopy");
#else // HAVE_MKL_DOMATCOPY
    timings_base::start_timer("dcopy");
    if(ni < nj) {
        double *c1 = c;
        for(size_t i = 0; i < ni; i++, c1 += sic) {
            cblas_dcopy(nj, a + i, sja, c1, 1);
        }
    } else {
        const double *a1 = a;
        for(size_t j = 0; j < nj; j++, a1 += sja) {
            cblas_dcopy(ni, a1, 1, c + j, sic);
        }
    }
    timings_base::stop_timer("dcopy");
#endif // HAVE_MKL_DOMATCOPY
}


template<>
void linalg_mkl_level2<double>::copy_ij_ji_x(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sja,
    double b,
    double *c, size_t sic) {

#ifdef HAVE_MKL_DOMATCOPY
    timings_base::start_timer("mkl_domatcopy");
    mkl_domatcopy('R', 'T', nj, ni, b, a, sja, c, sic);
    timings_base::stop_timer("mkl_domatcopy");
#else // HAVE_MKL_DOMATCOPY
    timings_base::start_timer("dcopy+dscal");
    for(size_t i = 0; i < ni; i++) {
        cblas_dcopy(nj, a + i, sja, c + i * sic, 1);
        cblas_dscal(nj, b, c + i * sic, 1);
    }
    timings_base::stop_timer("dcopy+dscal");
#endif // HAVE_MKL_DOMATCOPY
}


template<>
void linalg_mkl_level2<double>::mul2_i_ip_p_x(
    void*,
    size_t ni, size_t np,
    const double *a, size_t sia,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("dgemv");
    cblas_dgemv(CblasRowMajor, CblasNoTrans, ni, np, d, a, sia, b, spb, 1.0,
        c, sic);
    timings_base::stop_timer("dgemv");
}


template<>
void linalg_mkl_level2<double>::mul2_i_pi_p_x(
    void*,
    size_t ni, size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("dgemv");
    cblas_dgemv(CblasRowMajor, CblasTrans, np, ni, d, a, spa, b, spb, 1.0,
        c, sic);
    timings_base::stop_timer("dgemv");
}


template<>
void linalg_mkl_level2<double>::mul2_ij_i_j_x(
    void*,
    size_t ni, size_t nj,
    const double *a, size_t sia,
    const double *b, size_t sjb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("dger");
    cblas_dger(CblasRowMajor, ni, nj, d, a, sia, b, sjb, c, sic);
    timings_base::stop_timer("dger");
}


template<>
double linalg_mkl_level2<double>::mul2_x_pq_pq(
    void*,
    size_t np, size_t nq,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    timings_base::start_timer("ddot");
    double c = 0.0;
    if(nq > 1) {
        for(size_t p = 0; p < np; p++) {
            c += cblas_ddot(nq, a + p * spa, 1, b + p * spb, 1);
        }
    } else {
        for(size_t p = 0; p < np; p++) {
            const double *a1 = a + p * spa, *b1 = b + p * spb;
            for(size_t q = 0; q < nq; q++) {
                c += a1[q] * b1[q];
            }
        }
    }
    timings_base::stop_timer("ddot");
    return c;
}


template<>
void linalg_mkl_level2<float>::add1_ij_ij_x(
    void*,
    size_t ni, size_t nj,
    const float *a, size_t sia,
    float b,
    float *c, size_t sic) {

#ifdef HAVE_MKL_DOMATCOPY
    if(ni * nj <= 256 * 256) {
        timings_base::start_timer("mkl_somatadd");
        float t[256 * 256];
        mkl_somatadd('R', 'N', 'N', ni, nj, b, a, sia, 1.0, c, sic, t, nj);
        if(sic == nj) {
            memcpy(c, t, sizeof(float) * ni * nj);
        } else {
            for(size_t i = 0; i < ni; i++) {
                memcpy(c + i * sic, t + i * nj, sizeof(float) * nj);
            }
        }
        timings_base::stop_timer("mkl_somatadd");
    } else
#endif // HAVE_MKL_DOMATCOPY
    {
        timings_base::start_timer("saxpy");
        for(size_t i = 0; i < ni; i++) {
            cblas_saxpy(nj, b, a + i * sia, 1, c + i * sic, 1);
        }
        timings_base::stop_timer("saxpy");
    }
}


template<>
void linalg_mkl_level2<float>::add1_ij_ji_x(
    void*,
    size_t ni, size_t nj,
    const float *a, size_t sja,
    float b,
    float *c, size_t sic) {

#ifdef HAVE_MKL_DOMATCOPY
    if(ni * nj <= 256 * 256 && sic == nj) {
        timings_base::start_timer("mkl_somatadd");
        float t[256 * 256];
        mkl_somatadd('R', 'T', 'N', ni, nj, b, a, sja, 1.0, c, sic, t, nj);
        memcpy(c, t, sizeof(float) * ni * nj);
        timings_base::stop_timer("mkl_somatadd");
    } else
#endif // HAVE_MKL_DOMATCOPY
    {
        timings_base::start_timer("saxpy");
        for(size_t i = 0; i < ni; i++) {
            cblas_saxpy(nj, b, a + i, sja, c + i * sic, 1);
        }
        timings_base::stop_timer("saxpy");
    }
}


template<>
void linalg_mkl_level2<float>::copy_ij_ij_x(
    void*,
    size_t ni, size_t nj,
    const float *a, size_t sia,
    float b,
    float *c, size_t sic) {

#ifdef HAVE_MKL_DOMATCOPY
    timings_base::start_timer("mkl_somatcopy");
    mkl_somatcopy('R', 'N', ni, nj, b, a, sia, c, sic);
    timings_base::stop_timer("mkl_somatcopy");
#else // HAVE_MKL_DOMATCOPY
    timings_base::start_timer("scopy+sscal");
    for(size_t i = 0; i < ni; i++) {
        cblas_scopy(nj, a + i * sia, 1, c + i * sic, 1);
        cblas_sscal(nj, b, c + i * sic, 1);
    }
    timings_base::stop_timer("scopy+sscal");
#endif // HAVE_MKL_DOMATCOPY
}


template<>
void linalg_mkl_level2<float>::copy_ij_ji(
    void*,
    size_t ni, size_t nj,
    const float *a, size_t sja,
    float *c, size_t sic) {

#ifdef HAVE_MKL_DOMATCOPY
    timings_base::start_timer("mkl_somatcopy");
    mkl_somatcopy('R', 'T', nj, ni, 1.0, a, sja, c, sic);
    timings_base::stop_timer("mkl_somatcopy");
#else // HAVE_MKL_DOMATCOPY
    timings_base::start_timer("scopy");
    if(ni < nj) {
        float *c1 = c;
        for(size_t i = 0; i < ni; i++, c1 += sic) {
            cblas_scopy(nj, a + i, sja, c1, 1);
        }
    } else {
        const float *a1 = a;
        for(size_t j = 0; j < nj; j++, a1 += sja) {
            cblas_scopy(ni, a1, 1, c + j, sic);
        }
    }
    timings_base::stop_timer("scopy");
#endif // HAVE_MKL_DOMATCOPY
}


template<>
void linalg_mkl_level2<float>::copy_ij_ji_x(
    void*,
    size_t ni, size_t nj,
    const float *a, size_t sja,
    float b,
    float *c, size_t sic) {

#ifdef HAVE_MKL_DOMATCOPY
    timings_base::start_timer("mkl_somatcopy");
    mkl_somatcopy('R', 'T', nj, ni, b, a, sja, c, sic);
    timings_base::stop_timer("mkl_somatcopy");
#else // HAVE_MKL_DOMATCOPY
    timings_base::start_timer("scopy+sscal");
    for(size_t i = 0; i < ni; i++) {
        cblas_scopy(nj, a + i, sja, c + i * sic, 1);
        cblas_sscal(nj, b, c + i * sic, 1);
    }
    timings_base::stop_timer("scopy+sscal");
#endif // HAVE_MKL_DOMATCOPY
}


template<>
void linalg_mkl_level2<float>::mul2_i_ip_p_x(
    void*,
    size_t ni, size_t np,
    const float *a, size_t sia,
    const float *b, size_t spb,
    float *c, size_t sic,
    float d) {

    timings_base::start_timer("sgemv");
    cblas_sgemv(CblasRowMajor, CblasNoTrans, ni, np, d, a, sia, b, spb, 1.0,
        c, sic);
    timings_base::stop_timer("sgemv");
}


template<>
void linalg_mkl_level2<float>::mul2_i_pi_p_x(
    void*,
    size_t ni, size_t np,
    const float *a, size_t spa,
    const float *b, size_t spb,
    float *c, size_t sic,
    float d) {

    timings_base::start_timer("sgemv");
    cblas_sgemv(CblasRowMajor, CblasTrans, np, ni, d, a, spa, b, spb, 1.0,
        c, sic);
    timings_base::stop_timer("sgemv");
}


template<>
void linalg_mkl_level2<float>::mul2_ij_i_j_x(
    void*,
    size_t ni, size_t nj,
    const float *a, size_t sia,
    const float *b, size_t sjb,
    float *c, size_t sic,
    float d) {

    timings_base::start_timer("sger");
    cblas_sger(CblasRowMajor, ni, nj, d, a, sia, b, sjb, c, sic);
    timings_base::stop_timer("sger");
}


template<>
float linalg_mkl_level2<float>::mul2_x_pq_pq(
    void*,
    size_t np, size_t nq,
    const float *a, size_t spa,
    const float *b, size_t spb) {

    timings_base::start_timer("sdot");
    float c = 0.0;
    if(nq > 1) {
        for(size_t p = 0; p < np; p++) {
            c += cblas_sdot(nq, a + p * spa, 1, b + p * spb, 1);
        }
    } else {
        for(size_t p = 0; p < np; p++) {
            const float *a1 = a + p * spa, *b1 = b + p * spb;
            for(size_t q = 0; q < nq; q++) {
                c += a1[q] * b1[q];
            }
        }
    }
    timings_base::stop_timer("sdot");
    return c;
}



} // namespace libtensor

