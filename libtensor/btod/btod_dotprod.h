#ifndef LIBTENSOR_BTOD_DOTPROD_H
#define LIBTENSOR_BTOD_DOTPROD_H

#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/abs_index.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/permutation.h"
#include "../tod/tod_dotprod.h"

namespace libtensor {


/**	\brief Computes the dot product of two block tensors

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_dotprod : public timings< btod_dotprod<N> > {
public:
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

private:
	btod_dotprod<N> &operator=(const btod_dotprod<N>&);

};


template<size_t N>
const char *btod_dotprod<N>::k_clazz = "btod_dotprod<N>";


template<size_t N>
btod_dotprod<N>::btod_dotprod(block_tensor_i<N, double> &bt1,
	block_tensor_i<N, double> &bt2)
	: m_bt1(bt1), m_bt2(bt2) {

	if(!m_bt1.get_bis().equals(m_bt2.get_bis())) {
		throw bad_parameter(g_ns, k_clazz, "btod_dotprod()", __FILE__,
			__LINE__, "Incompatible block tensors.");
	}
}


template<size_t N>
btod_dotprod<N>::btod_dotprod(block_tensor_i<N, double> &bt1,
	const permutation<N> &perm1, block_tensor_i<N, double> &bt2,
	const permutation<N> &perm2)
	: m_bt1(bt1), m_perm1(perm1), m_bt2(bt2), m_perm2(perm2) {

	block_index_space<N> bis1(m_bt1.get_bis());
	bis1.permute(m_perm1);
	block_index_space<N> bis2(m_bt2.get_bis());
	bis2.permute(m_perm2);
	if(!bis1.equals(bis2)) {
		throw bad_parameter(g_ns, k_clazz, "btod_dotprod()", __FILE__,
			__LINE__, "Incompatible block tensors.");
	}
}


template<size_t N>
double btod_dotprod<N>::calculate() throw(exception) {

	btod_dotprod<N>::start_timer();

	// No-symmetry implementation

	block_tensor_ctrl<N, double> ctrl1(m_bt1);
	block_tensor_ctrl<N, double> ctrl2(m_bt2);
	dimensions<N> bidims(m_bt1.get_bis().get_block_index_dims());

	abs_index<N> ai1(bidims);
	double d = 0.0;
	do {
		index<N> i1(ai1.get_index()), i2(ai1.get_index());
		i1.permute(m_perm1);
		i2.permute(m_perm2);

		if(!ctrl1.req_is_zero_block(i1) &&
			!ctrl2.req_is_zero_block(i2)) {

			tensor_i<N, double> &t1(ctrl1.req_block(i1));
			tensor_i<N, double> &t2(ctrl2.req_block(i2));
			d += tod_dotprod<N>(
				t1, m_perm1, t2, m_perm2).calculate();
			ctrl1.ret_block(i1);
			ctrl2.ret_block(i2);
		}
	} while(ai1.inc());

	btod_dotprod<N>::stop_timer();

	return d;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DOTPROD_H
