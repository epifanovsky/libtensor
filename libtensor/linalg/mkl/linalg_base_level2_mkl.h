#ifndef LIBTENSOR_LINALG_BASE_LEVEL2_MKL_H
#define LIBTENSOR_LINALG_BASE_LEVEL2_MKL_H

#include <libtensor/timings.h>
#include "../generic/linalg_base_level2_generic.h"

namespace libtensor {


/** \brief Level-2 linear algebra operations (MKL)

    \ingroup libtensor_linalg
 **/
class linalg_base_level2_mkl :
    public linalg_base_level2_generic,
    public timings<linalg_base_level2_mkl> {

public:
    static const char *k_clazz; //!< Class name

public:
    static void add1_ij_ij_x(
        size_t ni, size_t nj,
        const double *a, size_t sia,
        double b,
        double *c, size_t sic);

    static void add1_ij_ji_x(
        size_t ni, size_t nj,
        const double *a, size_t sja,
        double b,
        double *c, size_t sic);

    static void i_ip_p_x(
        size_t ni, size_t np,
        const double *a, size_t sia,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d);

    static void i_pi_p_x(
        size_t ni, size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d);

    static void ij_i_j_x(
        size_t ni, size_t nj,
        const double *a, size_t sia,
        const double *b, size_t sjb,
        double *c, size_t sic,
        double d);

    static void ij_ij_x(
        size_t ni, size_t nj,
        const double *a, size_t sia,
        double b,
        double *c, size_t sic);

    static void ij_ji(
        size_t ni, size_t nj,
        const double *a, size_t sja,
        double *c, size_t sic);

    static void ij_ji_x(
        size_t ni, size_t nj,
        const double *a, size_t sja,
        double b,
        double *c, size_t sic);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL2_MKL_H
