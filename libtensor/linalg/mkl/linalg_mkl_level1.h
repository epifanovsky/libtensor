#ifndef LIBTENSOR_LINALG_MKL_LEVEL1_H
#define LIBTENSOR_LINALG_MKL_LEVEL1_H

#include "../linalg_timings.h"
#include "../generic/linalg_generic_level1.h"

namespace libtensor {


/** \brief Level-1 linear algebra operations (MKL)

    \ingroup libtensor_linalg
 **/
class linalg_mkl_level1 :
    public linalg_generic_level1,
    public linalg_timings<linalg_mkl_level1> {

public:
    static const char *k_clazz; //!< Class name

private:
    typedef linalg_timings<linalg_mkl_level1> timings_base;

public:
    static void add_i_i_x_x(
        void*,
        size_t ni,
        const double *a, size_t sia, double ka,
        double b, double kb,
        double *c, size_t sic,
        double d);

    static void copy_i_i(
        void*,
        size_t ni,
        const double *a, size_t sia,
        double *c, size_t sic);

    static void div1_i_i_x(
        void *ctx,
        size_t ni,
        const double *a, size_t sia,
        double *c, size_t sic,
        double d);

    static void mul1_i_x(
        void*,
        size_t ni,
        double a,
        double *c, size_t sic);

    static double mul2_x_p_p(
        void*,
        size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb);

    static void mul2_i_i_x(
        void*,
        size_t ni,
        const double *a, size_t sia,
        double b,
        double *c, size_t sic);

    static void mul2_i_i_i_x(
        void*,
        size_t ni,
        const double *a, size_t sia,
        const double *b, size_t sib,
        double *c, size_t sic,
        double d);

    static void rng_setup(
        void*);

    static void rng_set_i_x(
        void*,
        size_t ni,
        double *a, size_t sia,
        double c);

    static void rng_add_i_x(
        void*,
        size_t ni,
        double *a, size_t sia,
        double c);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_MKL_LEVEL1_H
