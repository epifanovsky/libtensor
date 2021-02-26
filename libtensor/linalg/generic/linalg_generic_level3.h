#ifndef LIBTENSOR_LINALG_GENERIC_LEVEL3_H
#define LIBTENSOR_LINALG_GENERIC_LEVEL3_H

#include <cstdlib> // for size_t
#include "../linalg_timings.h"

namespace libtensor {


/** \brief Level-3 linear algebra operations (generic)

    \ingroup libtensor_linalg
 **/
template<typename T>
class linalg_generic_level3 : public linalg_timings<linalg_generic_level3<T> > {
public:
    static const char k_clazz[]; //!< Class name

private:
    typedef linalg_timings<linalg_generic_level3<T> > timings_base;

public:
    /** \brief \f$ c_i = \sum_{pq} a_{ipq} b_{qp} d \f$
        \param ctx Context of computational device (unused for CPUs).
        \param a Pointer to a.
        \param b Pointer to b.
        \param c Pointer to c.
        \param d Value of d.
        \param ni Number of elements i.
        \param np Number of elements p.
        \param nq Number of elements q.
        \param sia Step of i in a (sia >= np * spa).
        \param sic Step of i in c (sic >= ni).
        \param spa Step of p in a (spa >= nq).
        \param sqb Step of q in b (sqb >= np).
     **/
    static void mul2_i_ipq_qp_x(
        void *ctx,
        size_t ni, size_t np, size_t nq,
        const T *a, size_t spa, size_t sia,
        const T *b, size_t sqb,
        T *c, size_t sic,
        T d);

    /** \brief \f$ c_{ij} = c_{ij} + \sum_p a_{ip} b_{jp} d \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param nj Number of elements j.
        \param np Number of elements p.
        \param a Pointer to a.
        \param sia Step of i in a (sia >= np).
        \param b Pointer to b.
        \param sjb Step of j in b (sjb >= np).
        \param c Pointer to c.
        \param sic Step of i in c (sic >= nj).
        \param d Scalar d.
     **/
    static void mul2_ij_ip_jp_x(
        void *ctx,
        size_t ni, size_t nj, size_t np,
        const T *a, size_t sia,
        const T *b, size_t sjb,
        T *c, size_t sic,
        T d);

    /** \brief \f$ c_{ij} = c_{ij} + \sum_p a_{ip} b_{pj} d \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param nj Number of elements j.
        \param np Number of elements p.
        \param a Pointer to a.
        \param sia Step of i in a (sia >= np).
        \param b Pointer to b.
        \param spb Step of p in b (spb >= nj);
        \param c Pointer to c.
        \param sic Step of i in c (sic >= nj).
        \param d Scalar d.
     **/
    static void mul2_ij_ip_pj_x(
        void *ctx,
        size_t ni, size_t nj, size_t np,
        const T *a, size_t sia,
        const T *b, size_t spb,
        T *c, size_t sic,
        T d);

    /** \brief \f$ c_{ij} = c_{ij} + \sum_p a_{pi} b_{jp} d \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param nj Number of elements j.
        \param np Number of elements p.
        \param a Pointer to a.
        \param spa Step of p in a (spa >= ni).
        \param b Pointer to b.
        \param sjb Step of j in b (sjb >= np).
        \param c Pointer to c.
        \param sic Step of i in c (sic >= nj).
        \param d Value of d.
     **/
    static void mul2_ij_pi_jp_x(
        void *ctx,
        size_t ni, size_t nj, size_t np,
        const T *a, size_t spa,
        const T *b, size_t sjb,
        T *c, size_t sic,
        T d);

    /** \brief \f$ c_{ij} = c_{ij} + \sum_p a_{pi} b_{pj} d \f$
        \param ctx Context of computational device (unused for CPUs).
        \param ni Number of elements i.
        \param nj Number of elements j.
        \param np Number of elements p.
        \param a Pointer to a.
        \param spa Step of p in a (spa >= ni).
        \param b Pointer to b.
        \param spb Step of p in b (spb >= nj);
        \param c Pointer to c.
        \param sic Step of i in c (sic >= nj).
        \param d Value of d.
     **/
    static void mul2_ij_pi_pj_x(
        void *ctx,
        size_t ni, size_t nj, size_t np,
        const T *a, size_t spa,
        const T *b, size_t spb,
        T *c, size_t sic,
        T d);

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_GENERIC_LEVEL3_H
