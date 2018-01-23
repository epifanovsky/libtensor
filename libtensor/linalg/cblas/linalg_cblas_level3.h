#ifndef LIBTENSOR_LINALG_CBLAS_LEVEL3_H
#define LIBTENSOR_LINALG_CBLAS_LEVEL3_H

#include "../generic/linalg_generic_level3.h"

namespace libtensor {


/** \brief Level-3 linear algebra operations (CBLAS)

    \ingroup libtensor_linalg
 **/
template<typename T>
class linalg_cblas_level3 : public linalg_generic_level3<T> {
public:
    static const char *k_clazz; //!< Class name

public:
    static void mul2_ij_ip_jp_x(
        void*,
        size_t ni, size_t nj, size_t np,
        const T *a, size_t sia,
        const T *b, size_t sjb,
        T *c, size_t sic,
        T d);

    static void mul2_ij_ip_pj_x(
        void*,
        size_t ni, size_t nj, size_t np,
        const T *a, size_t sia,
        const T *b, size_t spb,
        T *c, size_t sic,
        T d);

    static void mul2_ij_pi_jp_x(
        void*,
        size_t ni, size_t nj, size_t np,
        const T *a, size_t spa,
        const T *b, size_t sjb,
        T *c, size_t sic,
        T d);

    static void mul2_ij_pi_pj_x(
        void*,
        size_t ni, size_t nj, size_t np,
        const T *a, size_t spa,
        const T *b, size_t spb,
        T *c, size_t sic,
        T d);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_CBLAS_LEVEL3_H
