#ifndef LIBTENSOR_LINALG_BASE_LEVEL3_ACML_H
#define LIBTENSOR_LINALG_BASE_LEVEL3_ACML_H

#include <libtensor/timings.h>
#include "../generic/linalg_base_level3_generic.h"

namespace libtensor {


/** \brief Level-3 linear algebra operations (ACML)

    \ingroup libtensor_linalg
 **/
class linalg_base_level3_acml :
    public linalg_base_level3_generic,
    public timings<linalg_base_level3_acml> {

public:
    static const char *k_clazz; //!< Class name

public:
    static void ij_ip_jp_x(
        size_t ni, size_t nj, size_t np,
        const double *a, size_t sia,
        const double *b, size_t sjb,
        double *c, size_t sic,
        double d);

    static void ij_ip_pj_x(
        size_t ni, size_t nj, size_t np,
        const double *a, size_t sia,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d);

    static void ij_pi_jp_x(
        size_t ni, size_t nj, size_t np,
        const double *a, size_t spa,
        const double *b, size_t sjb,
        double *c, size_t sic,
        double d);

    static void ij_pi_pj_x(
        size_t ni, size_t nj, size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL3_ACML_H
