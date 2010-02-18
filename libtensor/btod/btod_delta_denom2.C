#include <libtensor/core/abs_index.h>
#include <libtensor/tod/tod_delta_denom2.h>
#include "btod_delta_denom2.h"

namespace libtensor {

btod_delta_denom2::btod_delta_denom2(block_tensor_i<2, double> &dov,
	double thresh) :

	m_dov(dov), m_thresh(thresh) {

}


void btod_delta_denom2::perform(block_tensor_i<4, double> &bt)
	throw(exception) {

	// check dimensions here
//	if(!bt.get_bis().equals(m_dov.get_bis())) {
//		throw bad_parameter(g_ns, "btod_delta_denom2", "perform()",
//			__FILE__, __LINE__,
//			"Incompatible tensor.");
//	}

	// No-symmetry implementation

	block_tensor_ctrl<2, double> ctrl_dov(m_dov);
	block_tensor_ctrl<4, double> ctrl_bt(bt);
	dimensions<4> bidims(bt.get_bis().get_block_index_dims());

	// Temporary way to deal with alpha-beta; to be replaced with
	// appropriate symmetry.
	size_t ni = bidims[0], na = bidims[2];
	size_t ni_ab = ni / 2, na_ab = na / 2;

	abs_index<4> ai1(bidims);
	index<2> ia, ib, ja, jb, idx1, idx2;
	do {
		const index<4> &ijab = ai1.get_index();
		ia[0] = ib[0] = ijab[0]; ja[0] = jb[0] = ijab[1];
		ia[1] = ja[1] = ijab[2]; jb[1] = ib[1] = ijab[3];

		bool zero = ctrl_bt.req_is_zero_block(ijab);

		if((ia[0] < ni_ab) == (ia[1] < na_ab) &&
			(jb[0] < ni_ab) == (jb[1] < na_ab)) {

			idx1 = ia;
			idx2 = jb;
		} else if((ib[0] < ni_ab) == (ib[1] < na_ab) &&
			(ja[0] < ni_ab) == (ja[1] < na_ab)) {

			idx1 = ib;
			idx2 = ja;
		} else {

			zero = true;
		}

		if(zero) {
			ctrl_bt.req_zero_block(ijab);

		} else {
			tensor_i<2, double> &t1_ia(ctrl_dov.req_block(idx1));
			tensor_i<2, double> &t1_jb(ctrl_dov.req_block(idx2));
			tensor_i<4, double> &t2(ctrl_bt.req_block(ijab));
			tod_delta_denom2(t1_ia, t1_jb, m_thresh).perform(t2);
			ctrl_dov.ret_block(idx1);
			ctrl_dov.ret_block(idx2);
			ctrl_bt.ret_block(ijab);
		}
	} while(ai1.inc());
}


} // namespace libtensor
