#ifndef LIBTENSOR_LINALG_BASE_LEVEL1_MKL_H
#define LIBTENSOR_LINALG_BASE_LEVEL1_MKL_H

#include <libtensor/timings.h>
#include "../generic/linalg_base_level1_generic.h"

namespace libtensor {


/** \brief Level-1 linear algebra operations (MKL)

    \ingroup libtensor_linalg
 **/
class linalg_base_level1_mkl :
    public linalg_base_level1_generic,
    public timings<linalg_base_level1_mkl> {

public:
    static const char *k_clazz; //!< Class name

public:
    static void add_i_i_x_x(
        size_t ni,
        const double *a, size_t sia, double ka,
        double b, double kb,
        double *c, size_t sic,
        double d);

    static void i_i(
        size_t ni,
        const double *a, size_t sia,
        double *c, size_t sic);

    static void i_x(
        size_t ni,
        double a,
        double *c, size_t sic);

    static double x_p_p(
        size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb);

    static void i_i_x(
        size_t ni,
        const double *a, size_t sia,
        double b,
        double *c, size_t sic);

    static void i_i_i_x(
        size_t ni,
        const double *a, size_t sia,
        const double *b, size_t sib,
        double *c, size_t sic,
        double d);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL1_MKL_H
