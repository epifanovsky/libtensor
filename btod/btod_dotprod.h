#ifndef LIBTENSOR_BTOD_DOTPROD_H
#define LIBTENSOR_BTOD_DOTPROD_H

#include "defs.h"
#include "exception.h"
#include "core/block_tensor_i.h"
#include "core/block_tensor_ctrl.h"
#include "core/permutation.h"
#include "tod/tod_dotprod.h"

namespace libtensor {

/**	\brief Computes the dot product of two block tensors

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_dotprod {
private:
	static const char *k_clazz; //!< Class name

private:
	block_tensor_i<N, double> &m_bt1; //!< First block %tensor
	permutation<N> m_perm1; //!< Permutation of the first block %tensor
	block_tensor_i<N, double> &m_bt2; //!< Second block %tensor
	permutation<N> m_perm2; //!< Permutation of the second block %tensor

public:
	btod_dotprod(block_tensor_i<N, double> &bt1,
		block_tensor_i<N, double> &bt2);
	btod_dotprod(block_tensor_i<N, double> &bt1,
		const permutation<N> &perm1, block_tensor_i<N, double> &bt2,
		const permutation<N> &perm2);
	double calculate() throw(exception);
};

template<size_t N>
const char *btod_dotprod<N>::k_clazz = "btod_dotprod<N>";

template<size_t N>
btod_dotprod<N>::btod_dotprod(block_tensor_i<N, double> &bt1,
	block_tensor_i<N, double> &bt2)
	: m_bt1(bt1), m_bt2(bt2) {

}

template<size_t N>
btod_dotprod<N>::btod_dotprod(block_tensor_i<N, double> &bt1,
	const permutation<N> &perm1, block_tensor_i<N, double> &bt2,
	const permutation<N> &perm2)
	: m_bt1(bt1), m_perm1(perm1), m_bt2(bt2), m_perm2(perm2) {

}

template<size_t N>
double btod_dotprod<N>::calculate() throw(exception) {

	index<N> i0;
	block_tensor_ctrl<N, double> btc1(m_bt1);
	block_tensor_ctrl<N, double> btc2(m_bt2);
	tensor_i<N, double> &t1(btc1.req_block(i0));
	tensor_i<N, double> &t2(btc2.req_block(i0));
	tod_dotprod<N> op(t1, m_perm1, t2, m_perm2);
	return op.calculate();
}

} // namespace libtensor

#endif // LIBTENSOR_BTOD_DOTPROD_H
