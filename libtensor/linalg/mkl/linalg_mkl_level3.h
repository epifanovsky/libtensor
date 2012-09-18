#ifndef LIBTENSOR_LINALG_MKL_LEVEL3_H
#define LIBTENSOR_LINALG_MKL_LEVEL3_H

#include "../linalg_timings.h"
#include "../generic/linalg_generic_level3.h"

namespace libtensor {


/** \brief Level-3 linear algebra operations (MKL)

    \ingroup libtensor_linalg
 **/
class linalg_mkl_level3 :
    public linalg_generic_level3,
    public linalg_timings<linalg_mkl_level3> {

public:
    static const char *k_clazz; //!< Class name

private:
    typedef linalg_timings<linalg_mkl_level3> timings_base;

public:
    static void mul2_ij_ip_jp_x(
        void*,
        size_t ni, size_t nj, size_t np,
        const double *a, size_t sia,
        const double *b, size_t sjb,
        double *c, size_t sic,
        double d);

    static void mul2_ij_ip_pj_x(
        void*,
        size_t ni, size_t nj, size_t np,
        const double *a, size_t sia,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d);

    static void mul2_ij_pi_jp_x(
        void*,
        size_t ni, size_t nj, size_t np,
        const double *a, size_t spa,
        const double *b, size_t sjb,
        double *c, size_t sic,
        double d);

    static void mul2_ij_pi_pj_x(
        void*,
        size_t ni, size_t nj, size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_MKL_LEVEL3_H
