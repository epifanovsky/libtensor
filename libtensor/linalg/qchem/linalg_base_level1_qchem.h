#ifndef LIBTENSOR_LINALG_BASE_LEVEL1_QCHEM_H
#define LIBTENSOR_LINALG_BASE_LEVEL1_QCHEM_H

#include <libtensor/timings.h>
#include "../generic/linalg_base_level1_generic.h"

namespace libtensor {


/** \brief Level-1 linear algebra operations (Q-Chem)

    \ingroup libtensor_linalg
 **/
class linalg_base_level1_qchem :
    public linalg_base_level1_generic,
    public timings<linalg_base_level1_qchem> {

public:
    static const char *k_clazz; //!< Class name

public:
    static void add_i_i_x_x(
        size_t ni,
        const double *a, size_t sia, double ka,
        double b, double kb,
        double *c, size_t sic,
        double d);

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

private:
    static void mul_i_i_x_p11(size_t ni,
        const double *a, double b, double *c);

    static void mul_i_i_x_pxx(size_t ni,
        const double *a, size_t sia, double b, double *c, size_t sic);

    static void mul_i_i_x_m11(size_t ni,
        const double *a, double b, double *c);

    static void mul_i_i_x_mxx(size_t ni,
        const double *a, size_t sia, double b, double *c, size_t sic);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL1_QCHEM_H
