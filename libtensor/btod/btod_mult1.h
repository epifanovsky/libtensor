#ifndef LIBTENSOR_BTOD_MULT1_H
#define LIBTENSOR_BTOD_MULT1_H

#include "../defs.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../tod/tod_mult1.h"
#include "../tod/tod_set.h"
#include "bad_block_index_space.h"

namespace libtensor {


/**	\brief Elementwise multiplication of two block tensors
	\tparam N Tensor order.

	\sa tod_mult1

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_mult1 {
public:
	static const char *k_clazz; //!< Class name

private:
	block_tensor_i<N, double> &m_btb; //!< Second argument
	bool m_recip; //!< Reciprocal

public:
	btod_mult1(block_tensor_i<N, double> &btb, bool recip = false);

	void perform(block_tensor_i<N, double> &btc);

	void perform(block_tensor_i<N, double> &btc, double c);

private:
	void do_perform(block_tensor_i<N, double> &btc, bool zero, double c);

private:
	btod_mult1(const btod_mult1<N> &);
	const btod_mult1<N> &operator=(const btod_mult1<N> &);

};


template<size_t N>
const char *btod_mult1<N>::k_clazz = "btod_mult1<N>";


template<size_t N>
btod_mult1<N>::btod_mult1(block_tensor_i<N, double> &btb, bool recip) :
	m_btb(btb), m_recip(recip) {
}


template<size_t N>
void btod_mult1<N>::perform(block_tensor_i<N, double> &bta) {

	static const char *method = "perform(block_tensor_i<N, double>&)";

	if(!bta.get_bis().equals(m_btb.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "bta");
	}

	do_perform(bta, true, 1.0);
}


template<size_t N>
void btod_mult1<N>::perform(block_tensor_i<N, double> &bta, double c) {

	static const char *method =
		"perform(block_tensor_i<N, double>&, double)";

	if(!bta.get_bis().equals(m_btb.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "bta");
	}

	do_perform(bta, false, c);
}


template<size_t N>
void btod_mult1<N>::do_perform(
	block_tensor_i<N, double> &bta, bool zero, double c) {

	static const char *method =
		"do_perform(block_tensor_i<N, double>&, bool, double)";

	block_tensor_ctrl<N, double> ctrla(bta), ctrlb(m_btb);

	//	Assuming equal symmetry in A, B

	orbit_list<N, double> olsta(ctrla.req_symmetry());

	for(typename orbit_list<N, double>::iterator ioa = olsta.begin();
		ioa != olsta.end(); ioa++) {

		index<N> idxa(olsta.get_index(ioa)), idxb(idxa);

		bool zeroa = ctrla.req_is_zero_block(idxa);
		bool zerob = ctrlb.req_is_zero_block(idxb);
		if(m_recip && zerob) {
			throw bad_parameter(g_ns, k_clazz, method,
				__FILE__, __LINE__, "zero in btb");
		}
		if(zero && (zeroa || zerob)) {
			ctrla.req_zero_block(idxa);
			continue;
		}
		if(zeroa || zerob) continue;

		tensor_i<N, double> &blka = ctrla.req_block(idxa);
		tensor_i<N, double> &blkb = ctrlb.req_block(idxb);

		if(zero && c == 1.0) {
			tod_mult1<N>(blkb, m_recip).perform(blka);
		} else {
			tod_mult1<N>(blkb, m_recip).perform(blka, c);
		}

		ctrla.ret_block(idxa);
		ctrlb.ret_block(idxb);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT1_H
