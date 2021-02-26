#ifndef LIBTENSOR_LINALG_MKL_LEVEL2_H
#define LIBTENSOR_LINALG_MKL_LEVEL2_H

#include "../linalg_timings.h"
#include "../generic/linalg_generic_level2.h"

namespace libtensor {


/** \brief Level-2 linear algebra operations (MKL)

    \ingroup libtensor_linalg
 **/
template<typename T>
class linalg_mkl_level2 :
    public linalg_generic_level2<double>,
    public linalg_generic_level2<float>,
    public linalg_timings<linalg_mkl_level2<T> > {

public:
    static const char k_clazz[]; //!< Class name

private:
    typedef linalg_timings<linalg_mkl_level2<T> > timings_base;

public:
using linalg_generic_level2<double>::mul2_x_pq_qp;
using linalg_generic_level2<float>::mul2_x_pq_qp;

    static void add1_ij_ij_x(
        void*,
        size_t ni, size_t nj,
        const T *a, size_t sia,
        T b,
        T *c, size_t sic);

    static void add1_ij_ji_x(
        void*,
        size_t ni, size_t nj,
        const T *a, size_t sja,
        T b,
        T *c, size_t sic);

    static void copy_ij_ij_x(
        void*,
        size_t ni, size_t nj,
        const T *a, size_t sia,
        T b,
        T *c, size_t sic);

    static void copy_ij_ji(
        void*,
        size_t ni, size_t nj,
        const T *a, size_t sja,
        T *c, size_t sic);

    static void copy_ij_ji_x(
        void*,
        size_t ni, size_t nj,
        const T *a, size_t sja,
        T b,
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

    static T mul2_x_pq_pq(
        void *ctx,
        size_t np, size_t nq,
        const T *a, size_t spa,
        const T *b, size_t spb);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_MKL_LEVEL2_H
