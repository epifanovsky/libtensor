#ifndef LIBTENSOR_TOD_DOTPROD_H
#define LIBTENSOR_TOD_DOTPROD_H

#include "defs.h"
#include "exception.h"
#include "permutation.h"
#include "tensor_ctrl.h"

namespace libtensor {

/**	\brief Calculates the dot product of two tensors
	\tparam N Tensor order.

	\ingroup libtensor_tod
 **/
template<size_t N>
class tod_dotprod {
private:
	static const char *k_clazz; //!< Class name

private:
	tensor_i<N, double> &m_t1; //!< First %tensor
	tensor_i<N, double> &m_t2; //!< Second %tensor
	permutation<N> m_perm1; //!< Permutation of the first %tensor
	permutation<N> m_perm2; //!< Permutation of the second %tensor

public:
	//!	\name Construction and destruction
	//@{

	tod_dotprod(tensor_i<N, double> &t1, tensor_i<N, double> &t2)
		throw(exception);

	tod_dotprod(tensor_i<N, double> &t1, const permutation<N> &perm1,
		tensor_i<N, double> &t2, const permutation<N> &perm2)
		throw(exception);

	//@}

	/**	\brief Computes the dot product
	 **/
	double calculate() throw(exception);

private:
	bool verify_dims();
};

template<size_t N>
const char *tod_dotprod<N>::k_clazz = "tod_dotprod<N>";

template<size_t N>
tod_dotprod<N>::tod_dotprod(tensor_i<N, double> &t1, tensor_i<N, double> &t2)
	throw(exception)
	: m_t1(t1), m_t2(t2) {

	static const char *method = "tod_dotprod(tensor_i<N, double>&, "
		"tensor_i<N, double>&)";

	if(!verify_dims()) {
		throw_exc(k_clazz, method, "Incompatible tensor dimensions");
	}
}

template<size_t N>
tod_dotprod<N>::tod_dotprod(
	tensor_i<N, double> &t1, const permutation<N> &perm1,
	tensor_i<N, double> &t2, const permutation<N> &perm2)
	throw(exception)
	: m_t1(t1), m_perm1(perm1), m_t2(t2), m_perm2(perm2) {

	static const char *method = "tod_dotprod(tensor_i<N, double>&, "
		"const permutation<N>&, tensor_i<N, double>&, "
		"const permutation<N>&)";

	if(!verify_dims()) {
		throw_exc(k_clazz, method, "Incompatible tensor dimensions");
	}
}

template<size_t N>
double tod_dotprod<N>::calculate() throw(exception) {

	return 0.0;
}

template<size_t N>
bool tod_dotprod<N>::verify_dims() {
	dimensions<N> dims1(m_t1.get_dims());
	dimensions<N> dims2(m_t2.get_dims());
	dims1.permute(m_perm1);
	dims2.permute(m_perm2);
	return dims1.equals(dims2);
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_DOTPROD_H
