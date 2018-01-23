#ifndef LIBTENSOR_LINALG_CBLAS_LEVEL2_H
#define LIBTENSOR_LINALG_CBLAS_LEVEL2_H

#include "../generic/linalg_generic_level2.h"

namespace libtensor {


/** \brief Level-2 linear algebra operations (CBLAS)

    \ingroup libtensor_linalg
 **/
template<typename T>
class linalg_cblas_level2 : public linalg_generic_level2<T> {
public:
    static const char *k_clazz; //!< Class name

public:
    static void copy_ij_ji(
        void*,
        size_t ni, size_t nj,
        const T *a, size_t sja,
        T *c, size_t sic);

    static void mul2_i_ip_p_x(
        void*,
        size_t ni, size_t np,
        const T *a, size_t sia,
        const T *b, size_t spb,
        T *c, size_t sic,
        T d);

    static void mul2_i_pi_p_x(
        void*,
        size_t ni, size_t np,
        const T *a, size_t spa,
        const T *b, size_t spb,
        T *c, size_t sic,
        T d);

    static void mul2_ij_i_j_x(
        void*,
        size_t ni, size_t nj,
        const T *a, size_t sia,
        const T *b, size_t sjb,
        T *c, size_t sic,
        T d);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_CBLAS_LEVEL2_H
