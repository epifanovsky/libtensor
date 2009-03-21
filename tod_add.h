#ifndef LIBTENSOR_TOD_ADD_H
#define LIBTENSOR_TOD_ADD_H

#include "defs.h"
#include "exception.h"
#include "tensor_i.h"
#include "tod_additive.h"

namespace libtensor {

/**	\brief Adds two or more tensors

	Tensor addition of n tensors:
	\f[ B = c_1 \mathcal{P}_1 A_1 + c_2 \mathcal{P}_2 A_2 + \cdots +
		c_n \mathcal{P}_n A_n \f]

	Each operand must have the same dimensions as the result in order
	for the operation to be successful. 

	

	\ingroup libtensor_tod
**/
template<size_t N>
class tod_add : public tod_additive<N> {
private:
	struct operand {
		tensor_i<N,double> &m_t;
		const permutation<N> &m_p;
		const double m_c;
		operand(tensor_i<N,double> &t, const permutation<N> &p,
			double c) : m_t(t), m_p(p), m_c(c) {}
	};

	//tensor_i<double> &m_out; //!< Output tensor
	//permutation &m_perm_out; //!< Output permutation
	//dimensions m_dims_out; //!< Permuted dimensions of the output

public:
	/**	\brief Adds an operand
		\param t Tensor.
		\param p Permutation of %tensor elements.
		\param c Coefficient.
	**/
	void add_op(tensor_i<N,double> &t, const permutation<N> &p,
		const double c) throw(exception);

	//!	\name Implementation of direct_tensor_operation<T>
	//@{
	virtual void prefetch() throw(exception);
	//@}

	//!	\name Implementation of tod_additive
	//@{
	virtual void perform(tensor_i<N,double> &t) throw(exception);
	virtual void perform(tensor_i<N,double> &t, double c)
		throw(exception);
	//@}
};

template<size_t N>
void tod_add<N>::add_op(tensor_i<N,double> &t, const permutation<N> &p,
	const double c) throw(exception) {
	dimensions<N> d(t.get_dims());
	d.permute(p);
}

template<size_t N>
void tod_add<N>::prefetch() throw(exception) {
}

template<size_t N>
void tod_add<N>::perform(tensor_i<N,double> &t) throw(exception) {
}

template<size_t N>
void tod_add<N>::perform(tensor_i<N,double> &t, double c) throw(exception) {
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_ADD_H

