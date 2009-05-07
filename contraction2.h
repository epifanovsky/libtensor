#ifndef LIBTENSOR_CONTRACTION2_H
#define	LIBTENSOR_CONTRACTION2_H

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
	indexes if those exist. The number of nodex is less or equal to N+M+K.

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
	static const size_t k_maxconn = 2 * (N + M + K);

private:
	permutation<N + K> m_permc; //!< Permutation of result indexes
	size_t m_k; //!< Number of contracted indexes specified
	size_t m_conn[k_maxconn]; //!< Index connections
	size_t m_num_nodes; //!< Number of fused nodes

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the contraction object
		\param perm Specifies how argument indexes should be permuted
			in the output.
	 **/
	contraction2(const permutation<N + M> &perm);

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
	void contract(size_t ia, size_t ib) throw(exception);

	/**	\brief Adjusts %index numbering when the arguments come in a
			permuted form

		The contraction must be specified for unpermuted arguments
		first. Calling this method on an incomplete contraction will
	 `	cause an exception.

		\param perma Permutation of the first %tensor argument (a).
		\param permb Permutation of the second %tensor argument (b).
		\throw exception if the contraction is incomplete.
	 **/
	void permute(const permutation<N + K> &perma,
		const permutation<M + K> &permb) throw(exception);

	//@}

	//!	\name Interface for the computational party
	//@{

	/**	\brief Returns the number of contraction loop nodes
	 **/
	size_t get_num_nodes() const throw(exception);

	/**	\brief Returns the weight of a contraction loop node
		\param dima Dimensions of %tensor a
		\param dimb Dimensions of %tensor b
	 **/
	size_t get_weight(const dimensions<N + K> &dima,
		const dimensions<M + K> &dimb) const throw(exception);

	/**	\brief Returns the linear %index increment in a
			(first argument)
		\param dima Dimensions of %tensor a
		\param dimb Dimensions of %tensor b
	 **/
	size_t get_increment_a(const dimensions<N + K> &dima,
		const dimensions<M + K> &dimb) const throw(exception);

	/**	\brief Returns the linear %index increment in b
			(second argument)
		\param dima Dimensions of %tensor a
		\param dimb Dimensions of %tensor b
	 **/
	size_t get_increment_b(const dimensions<N + K> &dima,
		const dimensions<M + K> &dimb) const throw(exception);

	/**	\brief Returns the linear %index increment in c (result)
		\param dima Dimensions of %tensor a
		\param dimb Dimensions of %tensor b
	 **/
	size_t get_increment_c(const dimensions<N + K> &dima,
		const dimensions<M + K> &dimb) const throw(exception);

	//@}

private:
	/**	\brief Fuses the indexes
	 **/
	void reduce();
};

template<size_t N, size_t M, size_t K>
contraction2<N, M, K>::contraction2(const permutation<N + M> &perm) :
m_permc(perm), m_k(0), m_num_nodes(0) {
	for(size_t i = 0; i < k_maxconn; i++) m_conn[i] = k_invalid;
}

template<size_t N, size_t M, size_t K>
contraction2<N, M, K>::contraction2(const contraction2<N, M, K> &contr) :
m_permc(contr.m_permc), m_k(contr.m_k), m_num_nodes(contr.m_num_nodes) {
	for(size_t i = 0; i < k_maxconn; i++) m_conn[i] = contr.m_conn[i];
}

template<size_t N, size_t M, size_t K>
inline bool contraction2<N, M, K>::is_complete() const {
	return m_k == K;
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::contract(size_t ia, size_t ib) throw(exception) {
	if(is_complete()) {
		throw_exc("contraction2<N, M, K>", "contract()",
			"Contraction is complete");
	}
	if(ia >= N + K) {
		throw_exc("contraction2<N, M, K>", "contract()",
			"Contraction index ia is invalid");
	}
	if(ib >= M + K) {
		throw_exc("contraction2<N, M, K>", "contract()",
			"Contraction index ib is invalid");
	}

	size_t ja = N + M + ia;
	size_t jb = 2 * N + M + K + ib;

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
		size_t connc[N + M];
		size_t iconnc = 0;
		for(size_t i = N + M; i < k_maxconn; i++)
			if(m_conn[i] == k_invalid) connc[iconnc++] = i;
		m_permc.apply(N + M, connc);
		for(size_t i = 0; i < N + M; i++) {
			m_conn[i] = connc[i];
			m_conn[connc[i]] = i;
		}
		//reduce();
	}
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::reduce() {
	size_t i = 0;
	while(i < N + M) {
		i++;
	}
}

} // namespace libtensor

#endif // LIBTENSOR_CONTRACTION2_H

