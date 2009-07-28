#ifndef LIBTENSOR_TOD_BTCONV_H
#define LIBTENSOR_TOD_BTCONV_H

#include "defs.h"
#include "exception.h"
#include "core/block_tensor_i.h"
#include "core/block_tensor_ctrl.h"
#include "core/tensor_i.h"
#include "core/tensor_ctrl.h"
#include "btod/block_symop_double.h"

namespace libtensor {

/**	\brief Unfolds a block %tensor into a simple %tensor
	\tparam N Tensor order.

	\ingroup libtensor_tod
 **/
template<size_t N>
class tod_btconv {
private:
	block_tensor_i<N, double> m_bt; //!< Source block %tensor

public:
	//!	\name Construction and destruction
	//@{

	tod_btconv(block_tensor_i<N, double> &bt);
	~tod_btconv();

	//@}

	//!	\name Tensor operation
	//@{

	void perform(tensor_i<N, double> &t) throw(exception);

	//@}
};


template<size_t N>
tod_btconv<N>::tod_btconv(block_tensor_i<N, double> &bt) : m_bt(bt) {

}

template<size_t N>
tod_btconv<N>::~tod_btconv() {

}

template<size_t N>
void tod_btconv<N>::perform(tensor_i<N, double> &t) throw(exception) {

	const block_index_space<N> &bis = m_bt.get_bis();
	if(!bis.get_dims().equals(t.get_dims())) {
		throw_exc("tod_btconv<N>", "perform()",
			"Incompatible dimensions");
	}

	block_tensor_ctrl<N, double> btctrl(m_bt);
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_BTCONV_H
