#ifndef LIBTENSOR_LINALG_BASE_LEVEL1_MKL_H
#define LIBTENSOR_LINALG_BASE_LEVEL1_MKL_H

#include <algorithm>

#if defined(HAVE_MKL_VML)
#include <mkl_vml_functions.h>
#endif

#include "../cblas/linalg_base_level1_cblas.h"

namespace libtensor {


/** \brief Level-1 linear algebra operations (MKL)

    \ingroup libtensor_linalg
 **/
struct linalg_base_level1_mkl : public linalg_base_level1_cblas {


    /** \brief \f$ c_i = c_i + a_i b_i \f$
        \param ni Number of elements i.
        \param a Pointer to a.
        \param sia Step of i in a.
        \param b Pointer to b.
        \param sib Step of i in b.
        \param c Pointer to c.
        \param sic Step of i in c.
        \param d Scalar d.
     **/
    static void i_i_i_x(
        size_t ni,
        const double *a, size_t sia,
        const double *b, size_t sib,
        double *c, size_t sic,
        double d) {

#if defined(HAVE_MKL_VML)
        if(sia == 1 && sib == 1) {
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
        } else
#endif
        {
            for(size_t i = 0; i < ni; i++) {
                c[i * sic] += d * a[i * sia] * b[i * sib];
            }
        }
    }

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL1_MKL_H
