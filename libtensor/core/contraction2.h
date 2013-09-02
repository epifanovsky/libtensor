#ifndef LIBTENSOR_CONTRACTION2_H
#define LIBTENSOR_CONTRACTION2_H

#include <cstdio>
#include <libtensor/defs.h>
#include <libtensor/exception.h>
#include "dimensions.h"
#include "permutation.h"
#include "permutation_builder.h"
#include "sequence.h"

namespace libtensor {


/** \brief Specifies how two tensors should be contracted
    \tparam N Order of the first %tensor (a) less the contraction degree
    \tparam M Order of the second %tensor (b) less the contraction degree
    \tparam K Contraction degree (the number of indexes over which the
        tensors are contracted)

    The contraction class provides mediation between the user of a
    contraction %tensor operation and the operation itself by providing
    convenient interfaces for both.

    The user specifies which indexes in \em a and \em b are to be contracted
    and how the uncontracted indexes should be permuted in \em c.

    Contraction operations would prefer a different interface, which
    returns an iterator over list nodes that correspond to contraction
    loops. There are two flavors of the list: fused and regular (non-fused).
    The regular list contains all contraction loops starting from those
    that correspond indexes in the resulting %tensor and ending with
    contracted indexes (those that present in \em a and \em b, but don't
    make it    to \em c). The number of nodes in the regular list equals N+M+K.
    The fused list is optimized: its nodes represent loops over combined
    indexes if those exist. The number of nodes is less or equal to N+M+K.

    For example, the contraction
    \f$ c_{ijkl} = \sum_{pq} a_{ijpq} b_{klpq} \f$ can be rewritten as
    \f$ c_{mn} = \sum_{r} a_{mr} b_{nr} \f$, where \em m represents \em i
    and \em j combined (same for \em n and \em r). The new contraction
    will only have three fused loop nodes instead of six in the original
    one.

    \ingroup libtensor_core
 **/
template<size_t N, size_t M, size_t K>
class contraction2 {
public:
    static const char k_clazz[]; //!< Class name

private:
    enum {
        k_invalid = -1,
        k_ordera = N + K, //!< Order of %tensor a
        k_orderb = M + K, //!< Order of %tensor b
        k_orderc = N + M, //!< Order of %tensor c
        k_totidx = N + M + K, //!< Total number of indexes
        k_maxconn = 2 * k_totidx
    };

private:
    permutation<k_orderc> m_permc; //!< Permutation of result indexes
    size_t m_k; //!< Number of contracted indexes specified
    sequence<k_maxconn, size_t> m_conn; //!< Index connections

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Default constructor
     **/
    contraction2();

    /** \brief Creates the contraction object
        \param perm Specifies how argument indexes should be permuted
            in the output.
     **/
    contraction2(const permutation<k_orderc> &perm);

    /** \brief Copy constructor
     **/
    contraction2(const contraction2<N, M, K> &contr);

    //@}

    //!    \name Contraction setup
    //@{

    /** \brief Returns whether this contraction is complete
     **/
    bool is_complete() const;

    /** \brief Designates a contracted index
        \param ia Index number in the first %tensor argument (a).
        \param ib Index number in the second %tensor argument (b).
        \throw exception if index numbers are invalid or this
            contraction is complete.
     **/
    void contract(size_t ia, size_t ib) throw (exception);

    /** \brief Adjusts %index numbering when the argument A comes in a
            permuted form
        \param perma Permutation of the %tensor argument A.
        \throw exception if the contraction is incomplete.
     **/
    void permute_a(const permutation<k_ordera> &perma) throw(exception);

    /** \brief Adjusts %index numbering when the argument B comes in a
            permuted form
        \param permb Permutation of the %tensor argument B.
        \throw exception if the contraction is incomplete.
     **/
    void permute_b(const permutation<k_orderb> &permb) throw(exception);

    /** \brief Adjusts %index numbering when the result comes in a
            permuted form
        \param permc Permutation of the result (c).
     **/
    void permute_c(const permutation<k_orderc> &permc) throw(exception);

    const sequence<2 * (N + M + K), size_t> &get_conn() const
        throw(exception);

    //@}

private:
    /** \brief Connects the indexes in the arguments and result
     **/
    void connect();

    void make_seqc(sequence<k_orderc, size_t> &seqc);

    /** \brief Adjusts the %permutation of C when the %permutation of
            A or B is changed
     **/
    void adjust_permc(sequence<k_orderc, size_t> &seqc1,
        sequence<k_orderc, size_t> &seqc2);

};


template<size_t NM, size_t K>
class contraction2_connector {
private:
    enum {
        k_invalid = -1,
        k_orderc = NM,
        k_totidx = NM + K,
        k_maxconn = 2 * k_totidx
    };

public:
    static void connect(sequence<k_maxconn, size_t> &conn,
        const permutation<k_orderc> &permc);
};


template<size_t K>
class contraction2_connector<0, K> {
private:
    enum {
        k_orderc = 0,
        k_totidx = K,
        k_maxconn = 2 * k_totidx
    };

public:
    static void connect(sequence<k_maxconn, size_t> &conn,
        const permutation<k_orderc> &permc);
};


template<size_t N, size_t M, size_t K>
const char contraction2<N, M, K>::k_clazz[] = "contraction2<N, M, K>";


template<size_t N, size_t M, size_t K>
contraction2<N, M, K>::contraction2() :
    m_k(0), m_conn(size_t(k_invalid)) {

    if(K == 0) connect();
}

template<size_t N, size_t M, size_t K>
contraction2<N, M, K>::contraction2(const permutation<k_orderc> &perm) :
    m_permc(perm), m_k(0), m_conn(size_t(k_invalid)) {

    if(K == 0) connect();
}

template<size_t N, size_t M, size_t K>
contraction2<N, M, K>::contraction2(const contraction2<N, M, K> &contr) :
    m_permc(contr.m_permc), m_k(contr.m_k), m_conn(contr.m_conn) {

}

template<size_t N, size_t M, size_t K>
inline bool contraction2<N, M, K>::is_complete() const {

    return m_k == K;
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::contract(size_t ia, size_t ib) throw (exception) {

    static const char *method = "contract(size_t, size_t)";

    if(is_complete()) {
        throw_exc("contraction2<N, M, K>", "contract()",
            "Contraction is complete");
    }
    if(ia >= k_ordera) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Contraction index A is out of bounds.");
    }
    if(ib >= k_orderb) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Contraction index B is out of bounds.");
    }

    size_t ja = k_orderc + ia;
    size_t jb = k_orderc + k_ordera + ib;

    if(m_conn[ja] != size_t(k_invalid)) {
        throw_exc("contraction2<N, M, K>", "contract()",
            "Index ia is already contracted");
    }
    if(m_conn[jb] != size_t(k_invalid)) {
        throw_exc("contraction2<N, M, K>", "contract()",
            "Index ib is already contracted");
    }

    m_conn[ja] = jb;
    m_conn[jb] = ja;

    if(++m_k == K) {
        // Once contracted indexes are specified, collect all the
        // remaining ones, permute them properly, and put them in place
        connect();
    }
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::permute_a(const permutation<k_ordera> &perma)
    throw(exception) {

    static const char *method = "permute_a(const permutation<N + K>&)";

    if(!is_complete()) {
        throw_exc(k_clazz, method, "Contraction is incomplete");
    }

    if(perma.is_identity()) return;

    sequence<k_ordera, size_t> seqa(0);
    sequence<k_orderc, size_t> seqc1(0), seqc2(0);
    make_seqc(seqc1);
    for(register size_t i = 0; i < k_ordera; i++)
        seqa[i] = m_conn[k_orderc + i];
    perma.apply(seqa);
    for(register size_t i = 0; i < k_ordera; i++) {
        m_conn[k_orderc + i] = seqa[i];
        m_conn[seqa[i]] = k_orderc + i;
    }
    make_seqc(seqc2);

    adjust_permc(seqc1, seqc2);
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::permute_b(const permutation<k_orderb> &permb)
    throw(exception) {

    static const char *method = "permute_b(const permutation<M + K>&)";

    if(!is_complete()) {
        throw_exc(k_clazz, method, "Contraction is incomplete");
    }

    if(permb.is_identity()) return;

    sequence<k_orderb, size_t> seqb(0);
    sequence<k_orderc, size_t> seqc1(0), seqc2(0);
    make_seqc(seqc1);
    for(register size_t i = 0; i < k_orderb; i++)
        seqb[i] = m_conn[k_orderc + k_ordera + i];
    permb.apply(seqb);
    for(register size_t i = 0; i < k_orderb; i++) {
        m_conn[k_orderc + k_ordera + i] = seqb[i];
        m_conn[seqb[i]] = k_orderc + k_ordera + i;
    }
    make_seqc(seqc2);

    adjust_permc(seqc1, seqc2);
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::permute_c(const permutation<k_orderc> &permc)
    throw(exception) {

    if(!is_complete()) {
        throw_exc("contraction2<N, M, K>", "permute_c()",
            "Contraction is incomplete");
    }

    m_permc.permute(permc);
    connect();
}

template<size_t N, size_t M, size_t K>
inline const sequence<2 * (N + M + K), size_t>&
contraction2<N, M, K>::get_conn() const throw(exception) {

    if(!is_complete()) {
        throw_exc("contraction2<N, M, K>", "get_conn()",
            "Contraction is incomplete");
    }
    return m_conn;
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::connect() {

    contraction2_connector<N + M, K>::connect(m_conn, m_permc);
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::make_seqc(sequence<k_orderc, size_t> &seqc) {

    for(register size_t i = 0, j = 0; i < k_ordera + k_orderb; i++) {
        if(m_conn[k_orderc + i] < k_orderc) {
            seqc[j++] = m_conn[k_orderc + i];
        }
    }
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::adjust_permc(sequence<k_orderc, size_t> &seqc1,
    sequence<k_orderc, size_t> &seqc2) {

    size_t seqcc1[k_orderc], seqcc2[k_orderc];
    for(register size_t i = 0; i < k_orderc; i++) {
        seqcc1[i] = seqc1[i];
        seqcc2[i] = seqc2[i];
    }
    permutation_builder<k_orderc> pb(seqcc1, seqcc2);
    permutation<k_orderc> permc(m_permc), permcinv(m_permc, true);
    m_permc.permute(permcinv).permute(pb.get_perm()).permute(permc);
}

template<size_t NM, size_t K>
void contraction2_connector<NM, K>::connect(sequence<k_maxconn, size_t> &conn,
    const permutation<k_orderc> &permc) {

    sequence<k_orderc, size_t> connc(0);
    size_t iconnc = 0;
    for(size_t i = k_orderc; i < k_maxconn; i++) {
        if(conn[i] == size_t(k_invalid) || conn[i] < k_orderc)
            connc[iconnc++] = i;
    }
    permc.apply(connc);
    for(size_t i = 0; i < k_orderc; i++) {
        conn[i] = connc[i];
        conn[connc[i]] = i;
    }
}


template<size_t K>
void contraction2_connector<0, K>::connect(sequence<k_maxconn, size_t> &conn,
    const permutation<k_orderc> &permc) {

}


} // namespace libtensor

#endif // LIBTENSOR_CONTRACTION2_H

