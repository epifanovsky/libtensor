#include <cstring>
#include <mkl.h>
#ifdef HAVE_MKL_VML
#include <mkl_vml_functions.h>
#endif // HAVE_MKL_VML
#include <libutil/threads/spinlock.h>
#include <libutil/threads/tls.h>
#include "linalg_mkl_level1.h"

namespace libtensor {


const char *linalg_mkl_level1::k_clazz = "mkl";


void linalg_mkl_level1::add_i_i_x_x(
    void*,
    size_t ni,
    const double *a, size_t sia, double ka,
    double b, double kb,
    double *c, size_t sic,
    double d) {

    timings_base::start_timer("daxpy");
    cblas_daxpy(ni, d * ka, a, sia, c, sic);
    double db = d * kb * b;
    if(sic == 1) {
        for(size_t i = 0; i < ni; i++) c[i] += db;
    } else {
        for(size_t i = 0; i < ni; i++) c[i * sic] += db;
    }
    timings_base::stop_timer("daxpy");
}


void linalg_mkl_level1::copy_i_i(
    void*,
    size_t ni,
    const double *a, size_t sia,
    double *c, size_t sic) {

    if(sia == 1 && sic == 1) {
        timings_base::start_timer("memcpy");
        ::memcpy(c, a, ni * sizeof(double));
        timings_base::stop_timer("memcpy");
    } else {
        timings_base::start_timer("dcopy");
        cblas_dcopy(ni, a, sia, c, sic);
        timings_base::stop_timer("dcopy");
    }
}


void linalg_mkl_level1::div1_i_i(
    void *,
    size_t ni,
    const double *a, size_t sia,
    double *c, size_t sic) {

#if defined(HAVE_MKL_VML)
    if(sia == 1 && sic == 1) {
        timings_base::start_timer("vddiv");
        double buf[256];
        size_t len = 256;
        while(ni > 0) {
            if(ni < len) len = ni;
            vdDiv(len, c, a, buf);
            ::memcpy(c, buf, len * sizeof(double));
            ni -= len;
            a += len;
            c += len;
        }
        timings_base::stop_timer("vddiv");
    } else
#endif
    {
        linalg_generic_level1::div1_i_i(0, ni, a, sia, c, sic);
    }
}


void linalg_mkl_level1::mul1_i_x(
    void*,
    size_t ni,
    double a,
    double *c, size_t sic) {

    timings_base::start_timer("dscal");
    cblas_dscal(ni, a, c, sic);
    timings_base::stop_timer("dscal");
}


double linalg_mkl_level1::mul2_x_p_p(
    void*,
    size_t np,
    const double *a, size_t spa,
    const double *b, size_t spb) {

    timings_base::start_timer("ddot");
    double d = cblas_ddot(np, a, spa, b, spb);
    timings_base::stop_timer("ddot");
    return d;
}


void linalg_mkl_level1::mul2_i_i_x(
    void*,
    size_t ni,
    const double *a, size_t sia,
    double b,
    double *c, size_t sic) {

    timings_base::start_timer("daxpy");
    cblas_daxpy(ni, b, a, sia, c, sic);
    timings_base::stop_timer("daxpy");
}


void linalg_mkl_level1::mul2_i_i_i_x(
    void*,
    size_t ni,
    const double *a, size_t sia,
    const double *b, size_t sib,
    double *c, size_t sic,
    double d) {

#if defined(HAVE_MKL_VML)
    if(sia == 1 && sib == 1) {
        timings_base::start_timer("vdmul+daxpy");
        double buf[256];
        size_t len = 256;
        while(ni > 0) {
            if(ni < len) len = ni;
            vdMul(len, a, b, buf);
            cblas_daxpy(len, d, buf, 1, c, sic);
            ni -= len;
            a += len;
            b += len;
            c += len * sic;
        }
        timings_base::stop_timer("vdmul+daxpy");
    } else
#endif
    {
        timings_base::start_timer("nonblas");
        for(size_t i = 0; i < ni; i++) {
            c[i * sic] += d * a[i * sia] * b[i * sib];
        }
        timings_base::stop_timer("nonblas");
    }
}


namespace {

static struct {
    libutil::spinlock lock;
    unsigned n;
} rng_stream_count;

struct rng_stream {
    bool init;
    VSLStreamStatePtr stream;
    rng_stream() : init(false) { }
};

} // unnamed namespace


void linalg_mkl_level1::rng_setup(
    void*) {

    rng_stream_count.n = 0;
}


void linalg_mkl_level1::rng_set_i_x(
    void*,
    size_t ni,
    double *a, size_t sia,
    double c) {

#ifdef HAVE_MKL_VSL

    rng_stream &rs = libutil::tls<rng_stream>::get_instance().get();
    if(!rs.init) {
        unsigned count = 0;
        rng_stream_count.lock.lock();
        count = rng_stream_count.n++;
        rng_stream_count.lock.unlock();
        vslNewStream(&rs.stream, VSL_BRNG_MT2203 + count, 162);
        rs.init = true;
    }

    if(sia == 1) {
        if(vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rs.stream,
            ni, a, 0.0, c) != VSL_STATUS_OK) {
            throw 0;
        }
    } else {
        double buf[256];
        size_t ni1 = ni, off = 0;
        while(ni1 > 0) {
            size_t batsz = std::min(ni1, size_t(256));
            if(vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rs.stream,
                batsz, buf, 0.0, c) != VSL_STATUS_OK) {
                throw 0;
            }
            for(size_t i = 0; i < batsz; i++) a[(off + i) * sia] = buf[i];
            ni1 -= batsz;
        }
    }

#else // HAVE_MKL_VSL

    linalg_generic_level1::rng_set_i_x(0, ni, a, sia, c);

#endif // HAVE_MKL_VSL
}


void linalg_mkl_level1::rng_add_i_x(
    void*,
    size_t ni,
    double *a, size_t sia,
    double c) {

#ifdef HAVE_MKL_VSL

    rng_stream &rs = libutil::tls<rng_stream>::get_instance().get();
    if(!rs.init) {
        unsigned count = 0;
        rng_stream_count.lock.lock();
        count = rng_stream_count.n++;
        rng_stream_count.lock.unlock();
        vslNewStream(&rs.stream, VSL_BRNG_MT2203 + count, 162);
        rs.init = true;
    }

    double buf[256];
    size_t ni1 = ni, off = 0;
    while(ni1 > 0) {
        size_t batsz = std::min(ni1, size_t(256));
        if(vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rs.stream,
            batsz, buf, 0.0, c) != VSL_STATUS_OK) {
            throw 0;
        }
        if(sia == 1) {
            for(size_t i = 0; i < batsz; i++) a[off + i] += buf[i];
        } else {
            for(size_t i = 0; i < batsz; i++) a[(off + i) * sia] += buf[i];
        }
        ni1 -= batsz;
    }

#else // HAVE_MKL_VSL

    linalg_generic_level1::rng_add_i_x(0, ni, a, sia, c);

#endif // HAVE_MKL_VSL
}


} // namespace libtensor

