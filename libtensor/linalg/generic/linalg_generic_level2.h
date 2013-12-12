#ifndef LIBTENSOR_LINALG_GENERIC_LEVEL2_H
#define LIBTENSOR_LINALG_GENERIC_LEVEL2_H

#include <cstdlib> // for size_t
#include "../linalg_timings.h"

namespace libtensor {


/** \brief Level-2 linear algebra operations (generic)

    \ingroup libtensor_linalg
 **/
class linalg_generic_level2 : public linalg_timings<linalg_generic_level2> {
public:
    static const char k_clazz[]; //!< Class name

private:
    typedef linalg_timings<linalg_generic_level2> timings_base;

public:
    /** \brief \f$ c_{ij} = c_{ij} + a_{ij} b \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param nj Number of elements j.
        \param a Pointer to a.
        \param sia Step of i in a.
        \param b Scalar b.
        \param c Pointer to c.
        \param sic Step of i in c.
     **/
    static void add1_ij_ij_x(
        void *ctx,
        size_t ni, size_t nj,
        const double *a, size_t sia,
        double b,
        double *c, size_t sic);

    /** \brief \f$ c_{ij} = c_{ij} + a_{ji} b \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param nj Number of elements j.
        \param a Pointer to a.
        \param sja Step of j in a.
        \param b Scalar b.
        \param c Pointer to c.
        \param sic Step of i in c.
     **/
    static void add1_ij_ji_x(
        void *ctx,
        size_t ni, size_t nj,
        const double *a, size_t sja,
        double b,
        double *c, size_t sic);

    /** \brief \f$ c_{ij} = a_{ij} b \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param nj Number of elements j.
        \param a Pointer to a.
        \param sia Step of i in a.
        \param b Scalar b.
        \param c Pointer to c.
        \param sic Step of i in c.
     **/
    static void copy_ij_ij_x(
        void *ctx,
        size_t ni, size_t nj,
        const double *a, size_t sia,
        double b,
        double *c, size_t sic);

    /** \brief \f$ c_{ij} = a_{ji} \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param nj Number of elements j.
        \param a Pointer to a.
        \param sja Step of j in a.
        \param c Pointer to c.
        \param sic Step of i in c.
     **/
    static void copy_ij_ji(
        void *ctx,
        size_t ni, size_t nj,
        const double *a, size_t sja,
        double *c, size_t sic);

    /** \brief \f$ c_{ij} = a_{ji} b \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param nj Number of elements j.
        \param a Pointer to a.
        \param sja Step of j in a.
        \param b Scalar b.
        \param c Pointer to c.
        \param sic Step of i in c.
     **/
    static void copy_ij_ji_x(
        void *ctx,
        size_t ni, size_t nj,
        const double *a, size_t sja,
        double b,
        double *c, size_t sic);

    /** \brief \f$ c_i = c_i + \sum_p a_{ip} b_p d \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param np Number of elements p.
        \param a Pointer to a.
        \param sia Step of i in a (sia >= np).
        \param b Pointer to b.
        \param spb Step of p in b (spb >= 1).
        \param c Pointer to c.
        \param sic Step of i in c (sic >= 1).
        \param d Scalar d.
     **/
    static void mul2_i_ip_p_x(
        void *ctx,
        size_t ni, size_t np,
        const double *a, size_t sia,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d);

    /** \brief \f$ c_i = c_i + \sum_p a_{pi} b_p d \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param np Number of elements p.
        \param a Pointer to a.
        \param spa Step of p in a (spa >= ni).
        \param b Pointer to b.
        \param spb Step of p in b (spb >= 1).
        \param c Pointer to c.
        \param sic Step of i in c (sic >= 1).
        \param d Scalar d.
     **/
    static void mul2_i_pi_p_x(
        void *ctx,
        size_t ni, size_t np,
        const double *a, size_t spa,
        const double *b, size_t spb,
        double *c, size_t sic,
        double d);

    /** \brief \f$ c_{ij} = c_{ij} + a_i b_j d \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param nj Number of elements j.
        \param a Pointer to a.
        \param sia Step of i in a (sia >= 1).
        \param b Pointer to b.
        \param sjb Step of j in b (sjb >= 1).
        \param c Pointer to c.
        \param sic Step of i in c (sic >= nj).
        \param d Scalar d.
     **/
    static void mul2_ij_i_j_x(
        void *ctx,
        size_t ni, size_t nj,
        const double *a, size_t sia,
        const double *b, size_t sjb,
        double *c, size_t sic,
        double d);

    /** \brief \f$ c = \sum_{pq} a_{pq} b_{pq} \f$
        \param ctx Context of computational device (unused for CPUs).
        \param np Number of elements p.
        \param nq Number of elements q.
        \param a Pointer to a.
        \param spa Step of p in a (spa >= nq).
        \param b Pointer to b.
        \param sqb Step of p in b (spb >= nq).
        \return c.
     **/
    static double mul2_x_pq_pq(
        void *ctx,
        size_t np, size_t nq,
        const double *a, size_t spa,
        const double *b, size_t spb);

    /** \brief \f$ c = \sum_{pq} a_{pq} b_{qp} \f$
        \param ctx Context of computational device (unused for CPUs).
        \param np Number of elements p.
        \param nq Number of elements q.
        \param a Pointer to a.
        \param spa Step of p in a (spa >= nq).
        \param b Pointer to b.
        \param sqb Step of q in b (sqb >= np).
        \return c.
     **/
    static double mul2_x_pq_qp(
        void *ctx,
        size_t np, size_t nq,
        const double *a, size_t spa,
        const double *b, size_t sqb);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_GENERIC_LEVEL2_H
