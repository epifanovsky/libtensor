#ifndef LIBTENSOR_CONTRACTION2_LIST_BUILDER_H
#define LIBTENSOR_CONTRACTION2_LIST_BUILDER_H

#include <sstream>
#include <libtensor/defs.h>
#include <libtensor/exception.h>
#include "contraction2.h"

namespace libtensor {

/** \brief Builder for optimized lists of contraction loops
    \tparam N Order of the first %tensor (a) less the contraction degree
    \tparam M Order of the second %tensor (b) less the contraction degree
    \tparam K Contraction degree (the number of indexes over which the
        tensors are contracted)

	Given a contraction sequence the class constructs a minimum set of
	loops by fusing successive dimensions in the input and output tensors.
	This set of loops is then translated into a list of loop parameters
	using the function \c populate according to the tensor dimensions
	passed to the function. The list object has to provide the function
	\code
	void append(size_t weight, size_t inca, size_t incb, size_t incc);
	\endcode
	where \c weight is the length of the current loop while \c inca, \c incb,
	and \c incc are the increments for each tensor.

	\sa tod_contract2

	\ingroup libtensor_core
 **/
template<size_t N, size_t M, size_t K>
class contraction2_list_builder {
public:
    static const char *k_clazz; //!< Class name

private:
    static const size_t k_ordera = N + K; //!< Order of %tensor a
    static const size_t k_orderb = M + K; //!< Order of %tensor b
    static const size_t k_orderc = N + M; //!< Order of %tensor c
    static const size_t k_totidx = N + M + K; //!< Total number of indexes
    static const size_t k_maxconn = 2 * k_totidx;

private:
    const contraction2<N, M, K> &m_contr; //!< Contraction
    size_t m_num_nodes; //!< Number of fused nodes
    sequence<k_totidx, size_t> m_nodes; //!< Fused nodes
    sequence<k_totidx, size_t> m_nodesz; //!< Fused node sizes (weights)

public:
    /** \brief Constructor
    	\param contr Object describing the contraction
     **/
    contraction2_list_builder(const contraction2<N, M, K> &contr)
        throw(bad_parameter, out_of_bounds);

    /** \brief Populate the optimized list according to the tensor dimensions
    		given
        \param list Resulting list of contraction loops
        \param dima Dimension of tensor A
        \param dimb Dimension of tensor B
        \param dimc Dimension of tensor C
     **/
    template<typename ListT>
    void populate(ListT &list, const dimensions<k_ordera> &dima,
        const dimensions<k_orderb> &dimb,
        const dimensions<k_orderc> &dimc) const throw(exception);

private:
    void fuse() throw(out_of_bounds);
};


template<size_t N, size_t M, size_t K>
const char *contraction2_list_builder<N, M, K>::k_clazz =
    "contraction2_list_builder<N, M, K>";


template<size_t N, size_t M, size_t K>
contraction2_list_builder<N, M, K>::contraction2_list_builder(
    const contraction2<N, M, K> &contr) throw(bad_parameter, out_of_bounds)
: m_contr(contr), m_num_nodes(0), m_nodes(0), m_nodesz(0) {

    static const char *method =
        "contraction2_list_builder(const contraction2<N, M, K>&)";

    if(!m_contr.is_complete()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Contraction specifier is incomplete.");
    }

    fuse();
}


template<size_t N, size_t M, size_t K>
template<typename ListT>
void contraction2_list_builder<N, M, K>::populate<ListT>(ListT &list,
    const dimensions<k_ordera> &dima, const dimensions<k_orderb> &dimb,
    const dimensions<k_orderc> &dimc) const throw (exception) {

    const sequence<k_maxconn, size_t> &conn = m_contr.get_conn();
    sequence<k_orderc, size_t> dimc1(0);

    for(size_t i = k_orderc; i < k_orderc + k_ordera; i++) {
        register size_t iconn = conn[i];
        if(iconn < k_orderc) dimc1[iconn] = dima[i - k_orderc];
    }
    for(size_t i = k_orderc + k_ordera; i < k_maxconn; i++) {
        register size_t iconn = conn[i];
        if(iconn < k_orderc) {
            dimc1[iconn] = dimb[i - k_orderc - k_ordera];
        } else if(dima[iconn - k_orderc] !=
            dimb[i - k_orderc - k_ordera]) {

            std::ostringstream ss;
            ss << "Dimensions of contraction index are "
                "incompatible: " << dima[iconn - k_orderc]
                << " (a) vs. " << dimb[i - k_orderc - k_ordera]
                << " (b)";
            throw_exc(k_clazz, "populate()", ss.str().c_str());
        }
    }

    for(size_t i = 0; i < k_orderc; i++) {
        if(dimc[i] != dimc1[i]) {

            std::ostringstream ss;
            ss << "Dimensions of result index are incompatible: "
                << dimc[i] << " (c) vs. " << dimc1[i]
                << " (a/b)";
            throw_exc(k_clazz, "populate()", ss.str().c_str());
        }
    }

    for(size_t inode = 0; inode < m_num_nodes; inode++) {
        size_t weight = 1, inca = 0, incb = 0, incc = 0;
        // Here the first index from inode comes from c or a
        // If the index comes from c, make ica->c iab->a or b
        // If the index comes from a, make ica->a iab->b
        register size_t ica = m_nodes[inode], iab = conn[ica];

        // Calculate node weight and increments
        if(ica < k_orderc && iab < k_orderc + k_ordera) {
            // The index comes from a and goes to c
            register size_t sz = m_nodesz[inode];
            for(register size_t j = 0; j < sz; j++)
                weight *= dima[iab + j - k_orderc];
            inca = dima.get_increment(iab + sz - k_orderc - 1);
            incb = 0;
            incc = 1;
            for(register size_t j = k_orderc - 1; j >= ica + sz; j--)
                incc *= dimc[j];
        } else if(ica < k_orderc) {
            // The index comes from b and goes to c
            register size_t sz = m_nodesz[inode];
            for(register size_t j = 0; j < sz; j++)
                weight *= dimb[iab + j - k_orderc - k_ordera];
            inca = 0;
            incb = dimb.get_increment(
                iab + sz - k_orderc - k_ordera - 1);
            incc = 1;
            for(register size_t j = k_orderc - 1; j >= ica + sz; j--)
                incc *= dimc[j];
        } else {
            // The index comes from a and b and gets contracted
            register size_t sz = m_nodesz[inode];
            for(register size_t j = 0; j < sz; j++)
                weight *= dima[ica + j - k_orderc];
            inca = dima.get_increment(ica + sz - k_orderc - 1);
            incb = dimb.get_increment(
                iab + sz - k_orderc - k_ordera - 1);
            incc = 0;
        }

        //~ if(weight > 1)
            list.append(weight, inca, incb, incc);
    }
}


template<size_t N, size_t M, size_t K>
void contraction2_list_builder<N, M, K>::fuse() throw(out_of_bounds) {

    const sequence<k_maxconn, size_t> &conn = m_contr.get_conn();

    m_num_nodes = 0;
    size_t i = 0;
    // Take care of indexes in result
    while(i < k_orderc) {
        size_t ngrp = 1;
        while(true) {
            if(conn[i + ngrp] != conn[i] + ngrp) break;
            if(i + ngrp >= k_orderc) break;
            if(conn[i] < k_orderc + k_ordera &&
                conn[i + ngrp] >= k_orderc + k_ordera) break;
            if(conn[i] >= k_orderc + k_ordera &&
                conn[i + ngrp] < k_orderc + k_ordera) break;
            ngrp++;
        }
        m_nodes[m_num_nodes] = i;
        m_nodesz[m_num_nodes] = ngrp;
        m_num_nodes++;
        i += ngrp;
    }
    // Take care of contracted indexes
    while(i < k_orderc + k_ordera) {
        size_t ngrp = 1;
        if(conn[i] > i) {
            while(conn[i + ngrp] == conn[i] + ngrp &&
                i + ngrp < k_orderc + k_ordera) ngrp++;
            m_nodes[m_num_nodes] = i;
            m_nodesz[m_num_nodes] = ngrp;
            m_num_nodes++;
        }
        i += ngrp;
    }
}


} // namespace libtensor

#endif // LIBTENSOR_CONTRACTION2_LIST_BUILDER_H
