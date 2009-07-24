#ifndef LIBTENSOR_BTOD_COPY_H
#define LIBTENSOR_BTOD_COPY_H

#include "defs.h"
#include "exception.h"
#include "btod/btod_additive.h"
#include "tod/tod_copy.h"

namespace libtensor {

/**	\brief Makes a copy of a block %tensor, applying a permutation or
		a scaling coefficient
	\tparam N Tensor order.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_copy : public btod_additive<N> {
private:
	block_tensor_i<N, double> &m_bt; //!< Source block %tensor
	permutation<N> m_perm; //!< Permutation
	double m_c; //!< Scaling coefficient

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the copy operation
		\param bt Source block %tensor.
		\param c Scaling coefficient.
	 **/
	btod_copy(block_tensor_i<N, double> &bt, double c = 1.0);

	/**	\brief Initializes the permuted copy operation
		\param bt Source block %tensor.
		\param p Permutation.
		\param c Scaling coefficient.
	 **/
	btod_copy(block_tensor_i<N, double> &bt, const permutation<N> &p,
		double c = 1.0);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_copy();
	//@}

	//!	\name Implementation of
	//!		libtensor::direct_block_tensor_operation<N, double>
	//@{
	virtual const block_index_space_i<N> &get_bis() const;
	virtual void perform(block_tensor_i<N, double> &bt) throw(exception);
	//@}

	//!	\name Implementation of libtensor::btod_additive<N>
	//@{
	virtual void perform(block_tensor_i<N, double> &bt, double c)
		throw(exception);
	//@}

};

template<size_t N>
btod_copy<N>::btod_copy(block_tensor_i<N, double> &bt, double c)
	: m_bt(bt), m_c(c) {

}

template<size_t N>
btod_copy<N>::btod_copy(block_tensor_i<N, double> &bt, const permutation<N> &p,
	double c)
	: m_bt(bt), m_perm(p), m_c(c) {
}

template<size_t N>
btod_copy<N>::~btod_copy() {
}

template<size_t N>
const block_index_space_i<N> &btod_copy<N>::get_bis() const {
	throw_exc("btod_copy<N>", "get_bis()", "Not implemented");
}

template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &bt) throw(exception) {
	block_tensor_ctrl<N, double> ctrl_src(m_bt), ctrl_dst(bt);
	index<N> i0;
	tensor_i<N, double> &tsrc(ctrl_src.req_block(i0));
	tensor_i<N, double> &tdst(ctrl_dst.req_block(i0));
	tod_copy<N> op(tsrc, m_perm, m_c);
	op.perform(tdst);
}

template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &bt, double c)
	throw(exception) {
	block_tensor_ctrl<N, double> ctrl_src(m_bt), ctrl_dst(bt);
	index<N> i0;
	tensor_i<N, double> &tsrc(ctrl_src.req_block(i0));
	tensor_i<N, double> &tdst(ctrl_dst.req_block(i0));
	tod_copy<N> op(tsrc, m_perm, m_c);
	op.perform(tdst, c);
}

} // namespace libtensor

#endif // LITENSOR_BTOD_COPY_H
