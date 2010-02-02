#include "tod_mkdelta.h"

namespace libtensor {


const char *tod_mkdelta::k_clazz = "tod_mkdelta";


tod_mkdelta::tod_mkdelta(tensor_i<2, double> &fi, tensor_i<2, double> &fa) :
	m_fi(fi), m_fa(fa) {

	static const char *method = "tod_mkdelta::tod_mkdelta()";

	if(fi.get_dims()[0] != fi.get_dims()[1]) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"The fi tensor has incorrect dimensions.");
	}
	if(fa.get_dims()[0] != fa.get_dims()[1]) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"The fa tensor has incorrect dimensions.");
	}
}


tod_mkdelta::~tod_mkdelta() {

}


void tod_mkdelta::prefetch() throw(exception) {

	tensor_ctrl<2, double> ctrl_fi(m_fi), ctrl_fa(m_fa);
	ctrl_fi.req_prefetch();
	ctrl_fa.req_prefetch();
}


void tod_mkdelta::perform(tensor_i<2, double> &delta) throw(exception) {

	static const char *method = "perform(tensor_i<2, double>&)";

	size_t szi = m_fi.get_dims()[0];
	size_t sza = m_fa.get_dims()[0];
	if(delta.get_dims()[0] != szi || delta.get_dims()[1] != sza) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"The delta tensor has incorrect dimensions.");
	}

	start_timer();

	tensor_ctrl<2, double> ctrl_fi(m_fi), ctrl_fa(m_fa);
	tensor_ctrl<2, double> ctrl_d(delta);

	const double *p_fi = ctrl_fi.req_const_dataptr();
	const double *p_fa = ctrl_fa.req_const_dataptr();
	double *p_d = ctrl_d.req_dataptr();

	double diag_i[szi], diag_a[sza];
	size_t stepi = m_fi.get_dims().get_increment(0) + 1;
	size_t stepa = m_fa.get_dims().get_increment(0) + 1;

	for(size_t i = 0; i < szi; i++) {
		diag_i[i] = p_fi[i*stepi];
	}
	for(size_t a = 0; a < sza; a++) {
		diag_a[a] = p_fa[a*stepa];
	}

	double *p1_d = p_d;
	size_t p1_d_step = delta.get_dims().get_increment(0);
	for(size_t i = 0; i < szi; i++, p1_d += p1_d_step) {
		double d = diag_i[i];
		for(size_t a = 0; a < sza; a++) {
			p1_d[a] = d - diag_a[a];
		}
	}

	ctrl_d.ret_dataptr(p_d);
	ctrl_fa.ret_dataptr(p_fa);
	ctrl_fi.ret_dataptr(p_fi);

	stop_timer();
}


} // namespace libtensor
