#ifndef LIBTENSOR_LINALG_CBLAS_LEVEL2_H
#define LIBTENSOR_LINALG_CBLAS_LEVEL2_H

#include "../generic/linalg_generic_level2.h"

namespace libtensor {


/** \brief Level-2 linear algebra operations (CBLAS)

    \ingroup libtensor_linalg
 **/
class linalg_cblas_level2 : public linalg_generic_level2 {
public:
    static const char *k_clazz; //!< Class name

public:
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

    static void ij_ji(
        size_t ni, size_t nj,
        const double *a, size_t sja,
        double *c, size_t sic);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_CBLAS_LEVEL2_H
