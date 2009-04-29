#ifndef LIBTENSOR_CONTRACTION2_H
#define	LIBTENSOR_CONTRACTION2_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

/**	\brief Specifies how two tensors should be contracted
	\tparam N Order of the first %tensor (a) less the contraction degree
	\tparam M Order of the second %tensor (b) less the contraction degree
	\tparam K Contraction degree (the number of indexes over which the
		tensors are contracted)

	The contraction class provides mediation between the user of a
	contraction %tensor operation and the operation itself by providing
	convenient interfaces for both.

	\ingroup libtensor
 **/
template<size_t N, size_t M, size_t K>
class contraction2 {
private:
	size_t m_k; //<! Number of contracted indexes specified
	size_t m_contr1[N + K]; //!< Contracted indexes in the first tensor
	size_t m_contr2[M + K]; //!< Contracted indexes in the second tensor
	bool m_is_contr1[N + K]; //!< Whether an index is contracted in 1
	bool m_is_contr2[M + K]; //!< Whether an index is contracted in 2

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the contraction object
		\param perm Specifies how argument indexes should be permuted
			in the output.
	 **/
	contraction2(const permutation<N + M> &perm);

	//@}

	//!	\name Contraction setup
	//@{

	/**	\brief Returns whether this contraction is complete
	 **/
	bool is_complete() const;

	/**	\brief Designates a contracted index
		\param i1 Index number in the first tensor.
		\param i2 Index number in the second tensor.
		\throw exception if index numbers are invalid or this
			contraction is complete.
	 **/
	void contract(size_t i1, size_t i2) throw (exception);

	//@}

	//!	\name Interface for the computational party
	//@{

	//@}
};

template<size_t N, size_t M, size_t K>
contraction2<N, M, K>::contraction2(const permutation& perm) : m_k(0) {
	for(size_t i = 0; i < N + K; i++) m_is_contr1[i] = false;
	for(size_t i = 0; k < M + K; i++) m_is_contr2[i] = false;
}

template<size_t N, size_t M, size_t K>
inline bool contraction2<N, M, L>::is_complete() const {
	return m_k == K;
}

template<size_t N, size_t M, size_t K>
void contraction2<N, M, K>::contract(size_t i1, size_t i2) throw (exception) {
	if(is_complete()) {
		throw_exc("contraction2<N, M, K>", "contract()",
			"Contraction is complete");
	}
	if(i1 >= N + K) {
		throw_exc("contraction2<N, M, K>", "contract()",
			"Contraction index i1 is invalid");
	}
	if(m_is_contr1[i1]) {
		throw_exc("contraction2<N, M, K>", "contract()",
			"Index i1 is already contracted");
	}
	if(i2 >= M + K) {
		throw_exc("contraction2<N, M, K>", "contract()",
			"Contraction index i2 is invalid");
	}
	if(m_is_contr2[i2]) {
		throw_exc("contraction2<N, M, K>", "contract()",
			"Index i2 is already contracted");
	}
	m_contr1[m_k] = i1;
	m_is_contr1[i1] = true;
	m_contr2[m_k] = i2;
	m_is_contr2[i2] = true;
	m_k++;
}

} // namespace libtensor

#endif // LIBTENSOR_CONTRACTION2_H

