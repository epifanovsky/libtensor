#ifndef LIBTENSOR_TOD_SYMCONTRACT2_H
#define LIBTENSOR_TOD_SYMCONTRACT2_H

#include "defs.h"
#include "exception.h"
#include "tod_additive.h"

namespace libtensor {

/**	\brief Contracts two tensors and symmetrizes the result (double)

	\ingroup libtensor_tod
**/
template<size_t N, size_t M, size_t K>
class tod_symcontract2 : public tod_additive<N+M> {
public:
	//!	\name Implementation of tod_additive
	//@{
	virtual void perform(tensor_i<N+M,double> &t) throw(exception);
	virtual void perform(tensor_i<N+M,double> &t, double c)
		throw(exception);
	//@}
};

template<size_t N, size_t M, size_t K>
void tod_symcontract2<N,M,K>::perform(tensor_i<N+M,double> &t)
	throw(exception) {
	char cls[32], meth[128];
	snprintf(cls, 32, "tod_symcontract2<%lu,%lu,%lu>", N, M, K);
	snprintf(meth, 128, "perform(tensor_i<%lu,double>&)", N+M);
	throw_exc(cls, meth, "Symmetrized contraction not implemented");
}

template<size_t N, size_t M, size_t K>
void tod_symcontract2<N,M,K>::perform(tensor_i<N+M,double> &t, double c)
	throw(exception) {
	char cls[32], meth[128];
	snprintf(cls, 32, "tod_symcontract2<%lu,%lu,%lu>", N, M, K);
	snprintf(meth, 128, "perform(tensor_i<%lu,double>&, double)",
		N+M);
	throw_exc(cls, meth, "Symmetrized contraction not implemented");
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_SYMCONTRACT2_H

