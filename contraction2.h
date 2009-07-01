#ifndef LIBTENSOR_CONTRACTION2_H
#define	LIBTENSOR_CONTRACTION2_H

#include <cstdio>
#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "permutation.h"

namespace libtensor {

/**	\brief Specifies how two tensors should be contracted
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
	make it	to \em c). The number of nodes in the regular list equals N+M+K.
	The fused list is optimized: its nodes represent loops over combined
	indexes if those exist. The number of nodes is less or equal to N+M+K.

	For example, the contraction
	\f$ c_{ijkl} = \sum_{pq} a_{ijpq} b_{klpq} \f$ can be rewritten as
	\f$ c_{mn} = \sum_{r} a_{mr} b_{nr} \f$, where \em m represents \em i
	and \em j combined (same for \em n and \em r). The new contraction
	will only have three fused loop nodes instead of six in the original
	one.

	\ingroup libtensor
 **/
template<size_t N, size_t M, size_t K>
class contraction2 {
private:
	static const size_t k_invalid = (size_t) (-1);
	static const size_t k_ordera = N + K; //!< Order of %tensor a
	static const size_t k_orderb = M + K; //!< Order of %tensor b
	static const size_t k_orderc = N + M; //!< Order of %tensor c
	static const size_t k_totidx = N + M + K; //!< Total number of indexes
	static const size_t k_maxconn = 2 * k_totidx;

private:
	permutation<k_orderc> m_permc; //!< Permutation of result indexes
	size_t m_k; //!< Number of contracted indexes specified
	size_t m_conn[k_maxconn]; //!< Index connections
	size_t m_num_nodes; //!< Number of fused nodes
	size_t m_nodes[k_totidx]; //!< Fused nodes
	size_t m_nodesz[k_totidx]; //!< Fused node sizes (weights)

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Default constructor
	 **/
	contraction2();

	/**	\brief Creates the contraction object
		\param perm Specifies how argument indexes should be permuted
			in the output.
	 **/
	contraction2(const permutation<k_orderc> &perm);

	/**	\brief Copy constructor
	 **/
	contraction2(const contraction2<N, M, K> &contr);

	//@}

	//!	\name Contraction setup
	//@{

	/**	\brief Returns whether this contraction is complete
	 **/
	bool is_complete() const;

	/**	\brief Designates a contracted index
		\param ia Index number in the first %tensor argument (a).
		\param ib Index number in the second %tensor argument (b).
		\throw exception if index numbers are invalid or this
			contraction is complete.
	 **/
	void contract(size_t ia, size_t ib) throw (exception);

	/**	\brief Adjusts %index numbering when the arguments come in a
			permuted form

		The contraction must be specified for unpermuted arguments
		first. Calling this method on an incomplete contraction will
	 `	cause an exception.

		\param perma Permutation of the first %tensor argument (a).
		\param permb Permutation of the second %tensor argument (b).
		\throw exception if the contraction is incomplete.
	 **/
	void permute_ab(const permutation<k_ordera> &perma,
		const permutation<k_orderb> &permb) throw (exception);

	/**	\brief Adjusts %index numbering when the result comes in a
			permuted form
		\param permc Permutation of the result (c).
	 **/
	void permute_c(const permutation<k_orderc> &permc) throw(exception);

	//@}

	//!	\name Interface for the computational party
	//@{

	/**	\brief Checks the dimensions of the arguments and the result
			and populates the loop node list
		\tparam ListT List type.
	 **/
	template<typename ListT>
	void populate(ListT &list, const dimensions<k_ordera> &dima,
		const dimensions<k_orderb> &dimb,
		const dimensions<k_orderc> &dimc) const throw (exception);

	//@}

private:
	/**	\brief Connects the indexes in the arguments and result
	 **/
	void connect();

	/**	\brief Fuses the indexes
	 **/
	void fuse();
};

template<size_t N, size_t M, size_t K>
contraction2<N, M, K>::contraction2()
: m_k(0), m_num_nodes(0) {

	for(size_t i = 0; i < k_maxconn; i++) {
		m_conn[i] = k_invalid;
	}
	for(size_t i = 0; i < k_totidx; i++) {
		m_nodes[i] = 0;
		m_nodesz[i] = 0;
	}
}

template<size_t N, size_t M, size_t K>
contraction2<N, M, K>::contraction2(const permutation<k_orderc> &perm)
: m_permc(perm), m_k(0), m_num_nodes(0) {

	for(size_t i = 0; i < k_maxconn; i++) {
		m_conn[i] = k_invalid;
	}
	for(size_t i = 0; i < k_totidx; i++) {
		m_nodes[i] = 0;
		m_nodesz[i] = 0;
	}
}

template<size_t N, size_t M, size_t K>
contraction2<N, M, K>::contraction2(const contraction2<N, M, K> &contr)
: m_permc(contr.m_permc), m_k(contr.m_k), m_num_nodes(contr.m_num_nodes) {

	for(size_t i = 0; i < k_maxconn; i++) {
		m_conn[i] = contr.m_conn[i];
	}
	for(size_t i = 0; i < k_totidx; i++) {
		m_nodes[i] = contr.m_nodes[i];
		m_nodesz[i] = contr.m_nodesz[i];
	}
}

template<size_t N, size_t M, size_t K>
inline bool contraction2<N, M, K>::is_complete() const {

	return m_k == K;
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::contract(size_t ia, size_t ib) throw (exception) {

	if(is_complete()) {
		throw_exc("contraction2<N, M, K>", "contract()",
			"Contraction is complete");
	}
	if(ia >= k_ordera) {
		throw_exc("contraction2<N, M, K>", "contract()",
			"Contraction index ia is invalid");
	}
	if(ib >= k_orderb) {
		throw_exc("contraction2<N, M, K>", "contract()",
			"Contraction index ib is invalid");
	}

	size_t ja = k_orderc + ia;
	size_t jb = k_orderc + k_ordera + ib;

	if(m_conn[ja] != k_invalid) {
		throw_exc("contraction2<N, M, K>", "contract()",
			"Index ia is already contracted");
	}
	if(m_conn[jb] != k_invalid) {
		throw_exc("contraction2<N, M, K>", "contract()",
			"Index ib is already contracted");
	}

	m_conn[ja] = jb;
	m_conn[jb] = ja;

	if(++m_k == K) {
		// Once contracted indexes are specified, collect all the
		// remaining ones, permute them properly, and put them in place
		connect();
		fuse();
	}
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::permute_c(const permutation<k_orderc> &permc)
	throw(exception) {

	if(!is_complete()) {
		throw_exc("contraction2<N, M, K>", "populate()",
			"Contraction is incomplete");
	}

	m_permc.permute(permc);
	connect();
	fuse();
}

template<size_t N, size_t M, size_t K> template<typename ListT>
void contraction2<N, M, K>::populate(ListT &list,
	const dimensions<k_ordera> &dima,
	const dimensions<k_orderb> &dimb,
	const dimensions<k_orderc> &dimc) const throw (exception) {

	if(!is_complete()) {
		throw_exc("contraction2<N, M, K>", "populate()",
			"Contraction is incomplete");
	}

	size_t dimc1[k_orderc];
	for(size_t i = 0; i < k_orderc; i++) dimc1[i] = 0;

	for(size_t i = k_orderc; i < k_orderc + k_ordera; i++) {
		register size_t conn = m_conn[i];
		if(conn < k_orderc) dimc1[conn] = dima[i - k_orderc];
	}
	for(size_t i = k_orderc + k_ordera; i < k_maxconn; i++) {
		register size_t conn = m_conn[i];
		if(conn < k_orderc) {
			dimc1[conn] = dimb[i - k_orderc - k_ordera];
		} else if(dima[conn - k_orderc] !=
			dimb[i - k_orderc - k_ordera]) {
			char errmsg[128];
			snprintf(errmsg, 128,
				"Dimensions of contraction index are "
				"incompatible: %lu (a) vs. %lu (b)",
				dima[conn - k_orderc],
				dimb[i - k_orderc - k_ordera]);
			throw_exc("contraction2<N, M, K>", "populate()",
				errmsg);
		}
	}

	for(size_t i = 0; i < k_orderc; i++) {
		if(dimc[i] != dimc1[i]) {
			char errmsg[128];
			snprintf(errmsg, 128,
				"Dimensions of result index are "
				"incompatible: %lu (c) vs. %lu (a/b)",
				dimc[i], dimc1[i]);
			throw_exc("contraction2<N, M, K>", "populate()",
				errmsg);
		}
	}

	for(size_t inode = 0; inode < m_num_nodes; inode++) {
		size_t weight = 1, inca = 0, incb = 0, incc = 0;

		// Here the first index from inode comes from c or a
		// If the index comes from c, make ica->c iab->a or b
		// If the index comes from a, make ica->a iab->b
		register size_t ica = m_nodes[inode], iab = m_conn[ica];

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

		if(weight > 1)
			list.append(weight, inca, incb, incc);
	}
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::connect() {

	size_t connc[k_orderc];
	size_t iconnc = 0;
	for(size_t i = k_orderc; i < k_maxconn; i++) {
		if(m_conn[i] == k_invalid || m_conn[i] < k_orderc)
			connc[iconnc++] = i;
	}
	m_permc.apply(k_orderc, connc);
	for(size_t i = 0; i < k_orderc; i++) {
		m_conn[i] = connc[i];
		m_conn[connc[i]] = i;
	}
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::fuse() {

	m_num_nodes = 0;
	size_t i = 0;
	// Take care of indexes in result
	while(i < k_orderc) {
		size_t ngrp = 1;
		while(m_conn[i + ngrp] == m_conn[i] + ngrp &&
			i + ngrp < k_orderc) ngrp++;
		m_nodes[m_num_nodes] = i;
		m_nodesz[m_num_nodes] = ngrp;
		m_num_nodes++;
		i += ngrp;
	}
	// Take care of contracted indexes
	while(i < k_orderc + k_ordera) {
		size_t ngrp = 1;
		if(m_conn[i] > i) {
			while(m_conn[i + ngrp] == m_conn[i] + ngrp &&
				i + ngrp < k_orderc + k_ordera) ngrp++;
			m_nodes[m_num_nodes] = i;
			m_nodesz[m_num_nodes] = ngrp;
			m_num_nodes++;
		}
		i += ngrp;
	}
}

} // namespace libtensor

#endif // LIBTENSOR_CONTRACTION2_H

