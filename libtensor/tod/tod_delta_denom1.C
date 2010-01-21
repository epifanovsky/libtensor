#include "tod_delta_denom1.h"

namespace libtensor {


tod_delta_denom1::tod_delta_denom1(tensor_i<2, double> &dov, double thresh) :

	m_dov(dov), m_thresh(thresh) {

}


void tod_delta_denom1::prefetch() throw(exception) {

	tensor_ctrl<2, double> tctrl_dov(m_dov);
	tctrl_dov.req_prefetch();
}


void tod_delta_denom1::perform(tensor_i<2, double> &t) throw(exception) {

	if(!t.get_dims().equals(m_dov.get_dims())) {
		throw bad_parameter(g_ns, "tod_delta_denom1", "perform()",
			__FILE__, __LINE__,
			"Tensor t_{ia} has incorrect dimensions");
	}

	size_t ni = m_dov.get_dims()[0];
	size_t na = m_dov.get_dims()[1];
	tensor_ctrl<2, double> tc_dov(m_dov), tc_t(t);
	const double *p_dov = tc_dov.req_const_dataptr();
	double *p_t = tc_t.req_dataptr();

	const double *p1_dov = p_dov;
	double *p1_t = p_t;

	for(size_t i = 0; i < ni; i++) {
		for(size_t a = 0; a < na; a++)
			p1_t[a] /= p1_dov[a];
		p1_dov += na;
		p1_t += na;
	}

	tc_t.ret_dataptr(p_t);
	tc_dov.ret_dataptr(p_dov);
}


} // namespace ccman2

