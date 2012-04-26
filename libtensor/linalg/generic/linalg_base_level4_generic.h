#ifndef LIBTENSOR_LINALG_BASE_LEVEL4_GENERIC_H
#define LIBTENSOR_LINALG_BASE_LEVEL4_GENERIC_H

#include <cstdlib> // for size_t

namespace libtensor {


/** \brief Level-4 linear algebra operations (generic)

    \ingroup libtensor_linalg
 **/
struct linalg_base_level4_generic {


    /** \brief \f$ c_{ij} = \sum_{pq} a_{ipq} b_{jqp} d \f$
        \param ni Number of elements i.
        \param nj Number of elements j.
        \param np Number of elements p.
        \param nq Number of elements q.
        \param a Pointer to a.
        \param spa Step of p in a (spa >= nq).
        \param sia Step of i in a (sia >= np * spa).
        \param b Pointer to b.
        \param sqb Step of q in b (sqb >= np).
        \param sjb Step of j in b (sjb >= nq * sqb).
        \param c Pointer to c.
        \param sic Step of i in c (sic >= ni).
        \param d Value of d.
     **/
    static void ij_ipq_jqp_x(
        size_t ni, size_t nj, size_t np, size_t nq,
        const double *a, size_t spa, size_t sia,
        const double *b, size_t sqb, size_t sjb,
        double *c, size_t sic,
        double d);


    /** \brief \f$ c_{ijk} = \sum_{p} a_{ip} b_{pkj} d \f$
        \param ni Number of elements i.
        \param nj Number of elements j.
        \param nk Number of elements k.
        \param np Number of elements p.
        \param a Pointer to a.
        \param sia Step of i in a (sia >= np).
        \param b Pointer to b.
        \param skb Step of k in b (skb >= nj).
        \param spb Step of p in b (spb >= nk * skb).
        \param c Pointer to c.
        \param sjc Step of j in c (sjc >= nk).
        \param sic Step of i in c (sic >= nj * sjc).
        \param d Value of d.
     **/
    static void ijk_ip_pkj_x(
        size_t ni, size_t nj, size_t nk, size_t np,
        const double *a, size_t sia,
        const double *b, size_t skb, size_t spb,
        double *c, size_t sjc, size_t sic,
        double d);


    /** \brief \f$ c_{ijk} = \sum_{p} a_{pi} b_{pkj} d \f$
        \param ni Number of elements i.
        \param nj Number of elements j.
        \param nk Number of elements k.
        \param np Number of elements p.
        \param a Pointer to a.
        \param spa Step of p in a (spa >= ni).
        \param b Pointer to b.
        \param skb Step of k in b (skb >= nj).
        \param spb Step of p in b (spb >= nk * skb).
        \param c Pointer to c.
        \param sjc Step of j in c (sjc >= nk).
        \param sic Step of i in c (sic >= nj * sjc).
        \param d Value of d.
     **/
    static void ijk_pi_pkj_x(
        size_t ni, size_t nj, size_t nk, size_t np,
        const double *a, size_t spa,
        const double *b, size_t skb, size_t spb,
        double *c, size_t sjc, size_t sic,
        double d);


};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL4_GENERIC_H
