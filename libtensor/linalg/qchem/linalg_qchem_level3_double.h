#ifndef LIBTENSOR_LINALG_QCHEM_LEVEL3_DOUBLE_H
#define LIBTENSOR_LINALG_QCHEM_LEVEL3_DOUBLE_H

#include "../linalg_timings.h"
#include "../generic/linalg_generic_level3.h"

namespace libtensor {


/** \brief Level-3 linear algebra operations (Q-Chem)

    \ingroup libtensor_linalg
 **/
class linalg_qchem_level3_double :
    public linalg_generic_level3<double>,
    public linalg_timings<linalg_qchem_level3_double> {

public:
    static const char k_clazz[]; //!< Class name

private:
    typedef linalg_timings<linalg_qchem_level3_double> timings_base;

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

#endif // LIBTENSOR_LINALG_QCHEM_LEVEL3_DOUBLE_H
