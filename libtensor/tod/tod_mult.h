#ifndef LIBTENSOR_TOD_MULT_H
#define TOD_MULT_H

#include "../defs.h"
#include "../core/tensor_i.h"
#include "bad_dimensions.h"

namespace libtensor {


/**	\brief Element-wise multiplication and division
	\tparam N Tensor order.

	The operation multiplies or divides two tensors element by element.
	Both arguments and result must have the same %dimensions or an exception
	will be thrown. When the division is requested, no checks are performed
	to ensure that the denominator is non-zero.

	\ingroup libtensor
 **/
template<size_t N>
class tod_mult {
public:
	static const char *k_clazz; //!< Class name

private:
	tensor_i<N, double> &m_ta; //!< First argument
	tensor_i<N, double> &m_tb; //!< Second argument
	bool m_recip; //!< Reciprocal (multiplication by 1/bi)

public:
	/**	\brief Creates the operation
		\param ta First argument.
		\param tb Second argument.
		\param recip \c false (default) sets up multiplication and
			\c true sets up element-wise division.
	 **/
	tod_mult(tensor_i<N, double> &ta, tensor_i<N, double> &tb,
		bool recip = false);

	/**	\brief Performs the operation, replaces the output.
		\param tc Output %tensor.
	 **/
	void perform(tensor_i<N, double> &tc);

	/**	\brief Performs the operation, adds to the output.
		\param tc Output %tensor.
		\param c Coefficient.
	 **/
	void perform(tensor_i<N, double> &tc, double c);
};


template<size_t N>
const char *tod_mult<N>::k_clazz = "tod_mult<N>";


template<size_t N>
tod_mult<N>::tod_mult(
	tensor_i<N, double> &ta, tensor_i<N, double> &tb, bool recip) :

	m_ta(ta), m_tb(tb), m_recip(recip) {

	if(!ta.get_dims().equals(tb.get_dims())) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
			"ta,tb");
	}
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_MULT_H
