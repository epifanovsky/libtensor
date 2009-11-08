#include "btod_delta_denom1.h"
#include "core/abs_index.h"
#include "tod/tod_delta_denom1.h"

namespace libtensor {

btod_delta_denom1::btod_delta_denom1(block_tensor_i<2, double> &dov,
	double thresh) :

	m_dov(dov), m_thresh(thresh) {

}


void btod_delta_denom1::perform(block_tensor_i<2, double> &bt)
	throw(exception) {

	if(!bt.get_bis().equals(m_dov.get_bis())) {
		throw bad_parameter(g_ns, "btod_delta_denom1", "perform()",
			__FILE__, __LINE__,
			"Incompatible tensor.");
	}

	// No-symmetry implementation

	block_tensor_ctrl<2, double> ctrl_dov(m_dov);
	block_tensor_ctrl<2, double> ctrl_bt(bt);
	dimensions<2> bidims(bt.get_bis().get_block_index_dims());

	abs_index<2> ai1(bidims);
	double d = 0.0;
	do {
		if(ctrl_dov.req_is_zero_block(ai1.get_index()) ||
			ctrl_bt.req_is_zero_block(ai1.get_index())) {

			ctrl_bt.req_zero_block(ai1.get_index());

		} else {
			tensor_i<2, double> &t1(ctrl_dov.req_block(
				ai1.get_index()));
			tensor_i<2, double> &t2(ctrl_bt.req_block(
				ai1.get_index()));
			tod_delta_denom1(t1, m_thresh).perform(t2);
			ctrl_dov.ret_block(ai1.get_index());
			ctrl_bt.ret_block(ai1.get_index());
		}
	} while(ai1.inc());
}


} // namespace libtensor
