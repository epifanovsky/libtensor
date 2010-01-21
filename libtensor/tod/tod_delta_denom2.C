#include <cmath>
#include <libtensor/core/tensor_ctrl.h>
#include "tod_delta_denom2.h"

namespace libtensor {


tod_delta_denom2::tod_delta_denom2(tensor_i<2, double> &dov1,
	tensor_i<2, double> &dov2, double thresh) :

	m_dov1(dov1), m_dov2(dov2), m_thresh(thresh) {

}


tod_delta_denom2::~tod_delta_denom2() {

}


void tod_delta_denom2::prefetch() throw(exception) {

	tensor_ctrl<2, double> tctrl_dov1(m_dov1), tctrl_dov2(m_dov2);
	tctrl_dov1.req_prefetch();
	tctrl_dov2.req_prefetch();
}


void tod_delta_denom2::perform(tensor_i<4, double> &t) throw(exception) {

	size_t ni = m_dov1.get_dims()[0];
	size_t nj = m_dov2.get_dims()[0];
	size_t na = m_dov1.get_dims()[1];
	size_t nb = m_dov2.get_dims()[1];

	// TODO: Check dimensions here

	tensor_ctrl<2, double> tctrl_dov1(m_dov1), tctrl_dov2(m_dov2);
	tensor_ctrl<4, double> tctrl_t(t);

	const double *p_dov1 = tctrl_dov1.req_const_dataptr();
	const double *p_dov2 = tctrl_dov2.req_const_dataptr();
	double *p_t = tctrl_t.req_dataptr();

	double *p1_t = p_t;
	const double *p1_dov = p_dov1, *p2_dov = p_dov2;
	size_t nz;

	double thresh = fabs(m_thresh);
	double neg_thresh = -thresh;

	size_t inner_step_sz = na * nb;

	for(size_t i = 0; i < ni; i++) {

		p2_dov = p_dov2;

		for(size_t j = 0; j < nj; j++) {

			inner_step(i, j, na, nb, p1_dov, p2_dov, p1_t);
			p2_dov += nb;
			p1_t += inner_step_sz;

		}

		p1_dov += na;

	}

	tctrl_dov1.ret_dataptr(p_dov1);
	tctrl_dov2.ret_dataptr(p_dov2);
	tctrl_t.ret_dataptr(p_t);
}


void tod_delta_denom2::inner_step(size_t i, size_t j, size_t na, size_t nb,
	const double *p_dov1_i, const double *p_dov2_j, double *p_t) {

	double *p1_t = p_t;
	double d[nb];

	for(size_t a = 0; a < na; a++) {
		for(size_t b = 0; b < nb; b++)
			p1_t[b] /= (p_dov1_i[a] + p_dov2_j[b]);
		p1_t += nb;
	}
}


} // namespace libtensor

