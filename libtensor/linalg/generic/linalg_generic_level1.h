#ifndef LIBTENSOR_LINALG_GENERIC_LEVEL1_H
#define LIBTENSOR_LINALG_GENERIC_LEVEL1_H

#include <cstdlib> // for size_t
#include "../linalg_timings.h"

namespace libtensor {


/** \brief Level-1 linear algebra operations (generic)

    \ingroup libtensor_linalg
 **/
class linalg_generic_level1 : public linalg_timings<linalg_generic_level1> {
public:
    static const char *k_clazz; //!< Class name

private:
    typedef linalg_timings<linalg_generic_level1> timings_base;

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
        const double *a, size_t sia, double ka,
        double b, double kb,
        double *c, size_t sic,
        double d);

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
        const double *a, size_t sia,
        double *c, size_t sic);

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
        double a,
        double *c, size_t sic);

    /** \brief \f$ c = \sum_p a_p b_p \f$
        \param ctx Context of computational device (unused for CPUs).
        \param np Number of elements p.
        \param a Pointer to a.
        \param spa Step of p in a.
        \param b Pointer to b.
        \param spb Step of p in b.
     **/
    static double mul2_x_p_p(
        void *ctx,
        size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb);

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
        const double *a, size_t sia,
        double b,
        double *c, size_t sic);

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
        const double *a, size_t sia,
        const double *b, size_t sib,
        double *c, size_t sic,
        double d);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_GENERIC_LEVEL1_H
