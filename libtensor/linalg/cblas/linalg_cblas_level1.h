#ifndef LIBTENSOR_LINALG_CBLAS_LEVEL1_H
#define LIBTENSOR_LINALG_CBLAS_LEVEL1_H

#include "../generic/linalg_generic_level1.h"

namespace libtensor {


/** \brief Level-1 linear algebra operations (CBLAS)

    \ingroup libtensor_linalg
 **/
template<typename T>
class linalg_cblas_level1 : public linalg_generic_level1<T> {
public:
    static const char *k_clazz; //!< Class name

public:
    static void add_i_i_x_x(
        void*,
        size_t ni,
        const T *a, size_t sia, T ka,
        T b, T kb,
        T *c, size_t sic,
        T d);

    static void copy_i_i(
        void*,
        size_t ni,
        const T *a, size_t sia,
        T *c, size_t sic);

    static void mul1_i_x(
        void*,
        size_t ni,
        T a,
        T *c, size_t sic);

    static T mul2_x_p_p(
        void*,
        size_t np,
        const T *a, size_t spa,
        const T *b, size_t spb);

    static void mul2_i_i_x(
        void*,
        size_t ni,
        const T *a, size_t sia,
        T b,
        T *c, size_t sic);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_CBLAS_LEVEL1_H
