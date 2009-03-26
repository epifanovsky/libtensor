#ifndef LIBTENSOR_TOD_CONTRACT2_H
#define LIBTENSOR_TOD_CONTRACT2_H

#include "defs.h"
#include "exception.h"
#include "tod_additive.h"

namespace libtensor {

/**	\brief Contracts two tensors (double)

	\param N Order of the first %tensor (a) less the contraction degree
	\param M Order of the second %tensor (b) less the contraction degree
	\param K Contraction degree (the number of indexes over which the
		tensors are contracted)

	This operation contracts %tensor T1 permuted as P1 with %tensor T2
	permuted as P2 over n last indexes. The result is permuted as Pres
	and written or added to the resulting %tensor.

	Although it is convenient to define a contraction through permutations,
	it is not the most efficient way of calculating it. This class seeks
	to use algorithms tailored for different tensors to get the best
	performance. For more information, read the wiki section on %tensor
	contractions.

	\ingroup libtensor_tod
**/
template<size_t N, size_t M, size_t K>
class tod_contract2 : public tod_additive<N+M> {
private:
	tensor_i<N+K,double> &m_ta; //!< First tensor (a)
	tensor_i<M+K,double> &m_tb; //!< Second tensor (b)
	permutation<N+K> m_pa; //!< Permutation of the first %tensor (a)
	permutation<M+K> m_pb; //!< Permutation of the second %tensor (b)
	permutation<N+M> m_pc; //!< Permutation of the result (c)

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the operation

		\param ta Tensor a
		\param pa Permutation of %tensor a
		\param tb Tensor b
		\param pb Permutation of %tensor b
		\param pc Permutation of the resulting %tensor c
	**/
	tod_contract2(tensor_i<N+K,double> &ta, const permutation<N+K> &pa,
		tensor_i<M+K,double> &tb, const permutation<M+K> &pb,
		const permutation<N+M> &pc) throw(exception);

	/**	\brief Virtual destructor
	**/
	virtual ~tod_contract2();

	//@}

	//!	\name Implementation of direct_tensor_operation<T>
	//@{
	virtual void prefetch() throw(exception);
	//@}

	//!	\name Implementation of tod_additive
	//@{
	virtual void perform(tensor_i<N+M,double> &t) throw(exception);
	virtual void perform(tensor_i<N+M,double> &t, double c)
		throw(exception);
	//@}
};

template<size_t N, size_t M, size_t K>
tod_contract2<N,M,K>::tod_contract2(
	tensor_i<N+K,double> &ta, const permutation<N+K> &pa,
	tensor_i<M+K,double> &tb, const permutation<M+K> &pb,
	const permutation<N+M> &pc) throw(exception) :
		m_ta(ta), m_pa(pa), m_tb(tb), m_pb(pb), m_pc(pc) {
}

template<size_t N, size_t M, size_t K>
tod_contract2<N,M,K>::~tod_contract2() {
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N,M,K>::prefetch() throw(exception) {
	tensor_ctrl<N+K,double> ctrl_ta(m_ta);
	tensor_ctrl<M+K,double> ctrl_tb(m_tb);
	ctrl_ta.req_prefetch();
	ctrl_tb.req_prefetch();
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N,M,K>::perform(tensor_i<N+M,double> &t) throw(exception) {
	dimensions<N+K> dims_ta(m_ta.get_dims());
	dimensions<M+K> dims_tb(m_tb.get_dims());
	dims_ta.permute(m_pa);
	dims_tb.permute(m_pb);
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N,M,K>::perform(tensor_i<N+M,double> &t, const double c)
	throw(exception) {
	char cls[32], meth[128];
	snprintf(cls, 32, "tod_contract2<%lu,%lu,%lu>", N, M, K);
	snprintf(meth, 128, "perform(tensor_i<%lu,double>&, const double)",
		N+M);
	throw_exc(cls, meth, "Contraction not implemented");
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_H

