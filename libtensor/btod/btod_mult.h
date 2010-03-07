#ifndef LIBTENSOR_BTOD_MULT_H
#define LIBTENSOR_BTOD_MULT_H

#include "../defs.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../tod/tod_mult.h"
#include "../tod/tod_set.h"
#include "bad_block_index_space.h"

namespace libtensor {


/**	\brief Elementwise multiplication of two block tensors
	\tparam N Tensor order.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_mult {
public:
	static const char *k_clazz; //!< Class name

private:
	block_tensor_i<N, double> &m_bta; //!< First argument
	block_tensor_i<N, double> &m_btb; //!< Second argument
	bool m_recip; //!< Reciprocal

public:
	btod_mult(block_tensor_i<N, double> &bta,
		block_tensor_i<N, double> &btb, bool recip = false);

	void perform(block_tensor_i<N, double> &btc);

	void perform(block_tensor_i<N, double> &btc, double c);

private:
	void do_perform(block_tensor_i<N, double> &btc, bool zero, double c);

private:
	btod_mult(const btod_mult<N> &);
	const btod_mult<N> &operator=(const btod_mult<N> &);

};


template<size_t N>
const char *btod_mult<N>::k_clazz = "btod_mult<N>";


template<size_t N>
btod_mult<N>::btod_mult(block_tensor_i<N, double> &bta,
	block_tensor_i<N, double> &btb, bool recip) :

	m_bta(bta), m_btb(btb), m_recip(recip) {

	static const char *method = "btod_mult(block_tensor_i<N, double>&, "
		"block_tensor_i<N, double>&, bool)";

	if(!m_bta.get_bis().equals(m_btb.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "bta,btb");
	}
}


template<size_t N>
void btod_mult<N>::perform(block_tensor_i<N, double> &btc) {

	static const char *method = "perform(block_tensor_i<N, double>&)";

	if(!btc.get_bis().equals(m_bta.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "btc");
	}

	do_perform(btc, true, 1.0);
}


template<size_t N>
void btod_mult<N>::perform(block_tensor_i<N, double> &btc, double c) {

	static const char *method =
		"perform(block_tensor_i<N, double>&, double)";

	if(!btc.get_bis().equals(m_bta.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "btc");
	}

	do_perform(btc, false, c);
}


template<size_t N>
void btod_mult<N>::do_perform(
	block_tensor_i<N, double> &btc, bool zero, double c) {

	static const char *method =
		"do_perform(block_tensor_i<N, double>&, bool, double)";

	block_tensor_ctrl<N, double> ctrla(m_bta), ctrlb(m_btb), ctrlc(btc);

	//	Assuming equal symmetry in A, B, C

	orbit_list<N, double> olsta(ctrla.req_symmetry());

	for(typename orbit_list<N, double>::iterator ioa = olsta.begin();
		ioa != olsta.end(); ioa++) {

		index<N> idxa(olsta.get_index(ioa)), idxb(idxa), idxc(idxa);

		bool zeroa = ctrla.req_is_zero_block(idxa);
		bool zerob = ctrlb.req_is_zero_block(idxb);
		if(m_recip && zerob) {
			throw bad_parameter(g_ns, k_clazz, method,
				__FILE__, __LINE__, "zero in btb");
		}
		if(zero && (zeroa || zerob)) {
			ctrlc.req_zero_block(idxc);
			continue;
		}
		if(zeroa || zerob) continue;

		tensor_i<N, double> &blka = ctrla.req_block(idxa);
		tensor_i<N, double> &blkb = ctrlb.req_block(idxb);
		tensor_i<N, double> &blkc = ctrlc.req_block(idxc);

		if(zero && c == 1.0) {
			tod_mult<N>(blka, blkb, m_recip).perform(blkc);
		} else if(zero) {
			tod_set<N>().perform(blkc);
			tod_mult<N>(blka, blkb, m_recip).perform(blkc, c);
		} else {
			tod_mult<N>(blka, blkb, m_recip).perform(blkc, c);
		}

		ctrla.ret_block(idxa);
		ctrlb.ret_block(idxb);
		ctrlc.ret_block(idxc);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT_H
