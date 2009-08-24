#ifndef LIBTENSOR_CONTRACTION2_H
#define	LIBTENSOR_CONTRACTION2_H

#include <cstdio>
#include "defs.h"
#include "exception.h"
#include "core/dimensions.h"
#include "core/permutation.h"
#include "core/sequence.h"

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

	\ingroup libtensor_tod
 **/
template<size_t N, size_t M, size_t K>
class contraction2 {
public:
	static const char *k_clazz; //!< Class name

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
	sequence<k_maxconn, size_t> m_conn; //!< Index connections

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

	const sequence<2 * (N + M + K), size_t> &get_conn() const
		throw(exception);

	//@}

private:
	/**	\brief Connects the indexes in the arguments and result
	 **/
	void connect();

};

template<size_t N, size_t M, size_t K>
const char *contraction2<N, M, K>::k_clazz = "contraction2<N, M, K>";

template<size_t N, size_t M, size_t K>
contraction2<N, M, K>::contraction2()
: m_k(0), m_conn(k_invalid) {

}

template<size_t N, size_t M, size_t K>
contraction2<N, M, K>::contraction2(
	const permutation<k_orderc> &perm)
: m_permc(perm), m_k(0), m_conn(k_invalid) {

}

template<size_t N, size_t M, size_t K>
contraction2<N, M, K>::contraction2(const contraction2<N, M, K> &contr)
: m_permc(contr.m_permc), m_k(contr.m_k), m_conn(contr.m_conn) {

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
	}
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::permute_ab(const permutation<k_ordera> &perma,
	const permutation<k_orderb> &permb) throw (exception) {

	if(!is_complete()) {
		throw_exc("contraction2<N, M, K>", "permute_ab()",
			"Contraction is incomplete");
	}

	if(perma.is_identity() && permb.is_identity()) return;

	sequence<k_ordera, size_t> seqa(0);
	sequence<k_orderb, size_t> seqb(0);
	for(register size_t i = 0; i < k_ordera; i++)
		seqa[i] = m_conn[k_orderc + i];
	for(register size_t i = 0; i < k_orderb; i++)
		seqb[i] = m_conn[k_orderc + k_ordera + i];
	seqa.permute(perma);
	seqb.permute(permb);
	for(register size_t i = 0; i < k_ordera; i++) {
		if(seqa[i] >= k_orderc) {
			for(register size_t j = 0; j < k_orderb; j++) {
				if(seqb[j] == k_orderc + i) {
					m_conn[k_orderc + i] =
						k_orderc + k_ordera + j;
					m_conn[k_orderc + k_ordera + j] =
						k_orderc + i;
					break;
				}
			}
		}
	}

	connect();
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

	size_t connc[k_orderc];
	size_t iconnc = 0;
	for(size_t i = k_orderc; i < k_maxconn; i++) {
		if(m_conn[i] == k_invalid || m_conn[i] < k_orderc)
			connc[iconnc++] = i;
	}
	m_permc.apply(connc);
	for(size_t i = 0; i < k_orderc; i++) {
		m_conn[i] = connc[i];
		m_conn[connc[i]] = i;
	}
}

} // namespace libtensor

#endif // LIBTENSOR_CONTRACTION2_H

