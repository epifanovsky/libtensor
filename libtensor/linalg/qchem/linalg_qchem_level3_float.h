#ifndef LIBTENSOR_LINALG_QCHEM_LEVEL3_FLOAT_H
#define LIBTENSOR_LINALG_QCHEM_LEVEL3_FLOAT_H

#include "../linalg_timings.h"
#include "../generic/linalg_generic_level3.h"

namespace libtensor {


/** \brief Level-3 linear algebra operations (Q-Chem)

    \ingroup libtensor_linalg
 **/
class linalg_qchem_level3_float :
    public linalg_generic_level3<float>,
    public linalg_timings<linalg_qchem_level3_float> {

public:
    static const char k_clazz[]; //!< Class name

private:
    typedef linalg_timings<linalg_qchem_level3_float> timings_base;

public:
    static void mul2_ij_ip_jp_x(
        void*,
        size_t ni, size_t nj, size_t np,
        const float *a, size_t sia,
        const float *b, size_t sjb,
        float *c, size_t sic,
        float d);

    static void mul2_ij_ip_pj_x(
        void*,
        size_t ni, size_t nj, size_t np,
        const float *a, size_t sia,
        const float *b, size_t spb,
        float *c, size_t sic,
        float d);

    static void mul2_ij_pi_jp_x(
        void*,
        size_t ni, size_t nj, size_t np,
        const float *a, size_t spa,
        const float *b, size_t sjb,
        float *c, size_t sic,
        float d);

    static void mul2_ij_pi_pj_x(
        void*,
        size_t ni, size_t nj, size_t np,
        const float *a, size_t spa,
        const float *b, size_t spb,
        float *c, size_t sic,
        float d);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_QCHEM_LEVEL3_FLOAT_H
