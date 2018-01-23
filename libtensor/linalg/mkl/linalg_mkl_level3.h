#ifndef LIBTENSOR_LINALG_MKL_LEVEL3_H
#define LIBTENSOR_LINALG_MKL_LEVEL3_H

#include "../linalg_timings.h"
#include "../generic/linalg_generic_level3.h"

namespace libtensor {


/** \brief Level-3 linear algebra operations (MKL)

    \ingroup libtensor_linalg
 **/
template<typename T>
class linalg_mkl_level3 :
    public linalg_generic_level3<double>,
    public linalg_generic_level3<float>,
    public linalg_timings<linalg_mkl_level3<T> > {

public:
    static const char k_clazz[]; //!< Class name

private:
    typedef linalg_timings<linalg_mkl_level3<T> > timings_base;

public:
    using linalg_generic_level3<double>::mul2_i_ipq_qp_x;
    using linalg_generic_level3<float>::mul2_i_ipq_qp_x;
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

#endif // LIBTENSOR_LINALG_MKL_LEVEL3_H
