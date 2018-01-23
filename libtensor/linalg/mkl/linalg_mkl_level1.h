#ifndef LIBTENSOR_LINALG_MKL_LEVEL1_H
#define LIBTENSOR_LINALG_MKL_LEVEL1_H

#include "../linalg_timings.h"
#include "../generic/linalg_generic_level1.h"

namespace libtensor {


/** \brief Level-1 linear algebra operations (MKL)

    \ingroup libtensor_linalg
 **/
template<typename T>
class linalg_mkl_level1 :
    public linalg_generic_level1<double>,
    public linalg_generic_level1<float>,
    public linalg_timings<linalg_mkl_level1<T> > {

public:
    static const char k_clazz[]; //!< Class name

private:
    typedef linalg_timings<linalg_mkl_level1<T> > timings_base;

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

    static void div1_i_i_x(
        void *ctx,
        size_t ni,
        const T *a, size_t sia,
        T *c, size_t sic,
        T d);

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

    static void mul2_i_i_i_x(
        void*,
        size_t ni,
        const T *a, size_t sia,
        const T *b, size_t sib,
        T *c, size_t sic,
        T d);

    static void rng_setup(
        void*);

    static void rng_set_i_x(
        void*,
        size_t ni,
        T *a, size_t sia,
        T c);

    static void rng_add_i_x(
        void*,
        size_t ni,
        T *a, size_t sia,
        T c);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_MKL_LEVEL1_H
