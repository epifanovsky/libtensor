#ifndef LIBTENSOR_LINALG_GENERIC_LEVEL1_H
#define LIBTENSOR_LINALG_GENERIC_LEVEL1_H

#include <cstdlib> // for size_t
#include "../linalg_timings.h"

namespace libtensor {


/** \brief Level-1 linear algebra operations (generic)

    \ingroup libtensor_linalg
 **/
template<typename T>
class linalg_generic_level1 : public linalg_timings<linalg_generic_level1<T> > {
public:
    static const char k_clazz[]; //!< Class name

private:
    typedef linalg_timings<linalg_generic_level1<T> > timings_base;

public:
    /** \brief \f$ c_i = c_i + (a_i + b) d \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param a Pointer to a.
        \param sia Step of i in a.
        \param b Scalar b.
        \param c Pointer to c.
        \param sic Step of i in c.
        \param d Scalar d.
     **/
    static void add_i_i_x_x(
        void *ctx,
        size_t ni,
        const T *a, size_t sia, T ka,
        T b, T kb,
        T *c, size_t sic,
        T d);

    /** \brief \f$ c_i = a_i \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param a Scalar a.
        \param sia Step of i in a.
        \param c Pointer to c.
        \param sic Step of i in c.
     **/
    static void copy_i_i(
        void *ctx,
        size_t ni,
        const T *a, size_t sia,
        T *c, size_t sic);

    /** \brief \f$ c_i = d c_i / a_i \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param a Pointer to a.
        \param sia Step of i in a.
        \param c Pointer to c.
        \param sic Step of i in c.
        \param d Scalar d.
     **/
    static void div1_i_i_x(
        void *ctx,
        size_t ni,
        const T *a, size_t sia,
        T *c, size_t sic,
        T d);

    /** \brief \f$ c_i = c_i a \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param a Scalar a.
        \param c Pointer to c.
        \param sic Step of i in c.
     **/
    static void mul1_i_x(
        void *ctx,
        size_t ni,
        T a,
        T *c, size_t sic);

    /** \brief \f$ c = \sum_p a_p b_p \f$
        \param ctx Context of computational device (unused for CPUs).
        \param np Number of elements p.
        \param a Pointer to a.
        \param spa Step of p in a.
        \param b Pointer to b.
        \param spb Step of p in b.
     **/
    static T mul2_x_p_p(
        void *ctx,
        size_t np,
        const T *a, size_t spa,
        const T *b, size_t spb);

    /** \brief \f$ c_i = c_i + a_i b \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param a Pointer to a.
        \param sia Step of i in a.
        \param b Scalar b.
        \param c Pointer to c.
        \param sic Step of i in c.
     **/
    static void mul2_i_i_x(
        void *ctx,
        size_t ni,
        const T *a, size_t sia,
        T b,
        T *c, size_t sic);

    /** \brief \f$ c_i = c_i + d a_i b_i \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param a Pointer to a.
        \param sia Step of i in a.
        \param b Pointer to b.
        \param sib Step of i in b.
        \param c Pointer to c.
        \param sic Step of i in c.
        \param d Scalar d
     **/
    static void mul2_i_i_i_x(
        void *ctx,
        size_t ni,
        const T *a, size_t sia,
        const T *b, size_t sib,
        T *c, size_t sic,
        T d);

    /** \brief Sets up the random number generator
        \param ctx Context of computational device (unused for CPUs).
     **/
    static void rng_setup(
        void *ctx);

    /** \brief Generates an array of random numbers in [0, 1]
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param a Pointer to a.
        \param sia Step of i in a.
        \param c Scaling coefficient.
     **/
    static void rng_set_i_x(
        void *ctx,
        size_t ni,
        T *a, size_t sia,
        T c);

    /** \brief Adds an array of random numbers in [0, 1]
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param a Pointer to a.
        \param sia Step of i in a.
        \param c Scaling coefficient.
     **/
    static void rng_add_i_x(
        void *ctx,
        size_t ni,
        T *a, size_t sia,
        T c);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_GENERIC_LEVEL1_H
