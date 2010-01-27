#ifndef LIBTENSOR_TOD_SYMCONTRACT2_H
#define LIBTENSOR_TOD_SYMCONTRACT2_H

#include <libvmm/std_allocator.h>
#include "../defs.h"
#include "../exception.h"
#include "tod_additive.h"
#include "tod_contract2.h"

namespace libtensor {

/**	\brief Contracts two tensors and symmetrizes the result (double)

	Symmetrized contraction of two tensors:
	\f[
		B = c_B \left[ A_1 \times A_2 + c \mathcal{P} \left( A_1 \times A_2 \right) \right]
	\f]
	where \f$ \times \f$ represents an arbitrary contraction

	\ingroup libtensor_tod
**/
template<size_t N, size_t M, size_t K>
class tod_symcontract2 : public tod_additive<M+N> {
private:
	tod_contract2<N, M, K> m_contr; //!< Contraction
	permutation<N+M> m_perm; //!< permutation to symmetrize
	const double m_c;  //!< prefactor

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the contraction operation

		\param contr Contraction.
		\param ta Tensor a (first argument).
		\param tb Tensor b (second argument).
		\param pc Permutation for symmetrization
		\param cc prefactor (in most cases +1/-1)
	 **/
	tod_symcontract2(const contraction2<N, M, K> &contr,
		tensor_i<N+K, double> &ta, tensor_i<M+K, double> &tb,
		permutation<N+M> &pc, const double cc=1.0);

	/**	\brief Virtual destructor
	 **/
	virtual ~tod_symcontract2();

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
tod_symcontract2<N,M,K>::tod_symcontract2(const contraction2<N,M,K> &contr,
	tensor_i<N+K,double> &ta, tensor_i<M+K,double> &tb,
	permutation<N+M> &pc, const double cc) :
	m_contr(contr,ta,tb), m_perm(pc), m_c(cc) {
}

template<size_t N, size_t M, size_t K>
tod_symcontract2<N,M,K>::~tod_symcontract2() {
}

template<size_t N, size_t M, size_t K>
void tod_symcontract2<N,M,K>::prefetch() throw(exception) {
	m_contr.prefetch();
}


template<size_t N, size_t M, size_t K>
void tod_symcontract2<N,M,K>::perform(tensor_i<N+M,double> &t)
	throw(exception) {
	// intermediate tensor
	tensor<N+M,double,libvmm::std_allocator<double> > tmp(t);

	m_contr.perform(tmp);

	tod_copy<N+M> cp(tmp);
	cp.perform(t);

	tod_add<N+M> add(tmp,m_perm,m_c);
	add.perform(t,1.0);
}

template<size_t N, size_t M, size_t K>
void tod_symcontract2<N,M,K>::perform(tensor_i<N+M,double> &t, double c)
	throw(exception) {
	tensor<N+M,double,libvmm::std_allocator<double> > tmp(t);
	m_contr.perform(tmp);

	tod_add<N+M> add(tmp,1.0);
	add.add_op(tmp,m_perm,m_c);
	add.perform(t,c);
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_SYMCONTRACT2_H

