#include "../defs.h"
#include "../exception.h"
#include "../linalg.h"
#include "loop_list_elem.h"
#include "overflow.h"

namespace libtensor {


const char *loop_list_elem::k_clazz = "loop_list_elem";


void loop_list_elem::run_loop(list_t &loop, registers &r, double c,
		bool doadd, bool recip) {

	iterator_t op;
	for (iterator_t i = loop.begin(); i != loop.end(); i++) {

		i->fn() = 0;
		if (i->stepb(0) == 1) op = i;
	}

	if (doadd && recip)
		op->fn() = &loop_list_elem::fn_div_add;
	else if (recip)
		op->fn() = &loop_list_elem::fn_div_put;
	else if (doadd)
		op->fn() = &loop_list_elem::fn_mult_add;
	else
		op->fn() = &loop_list_elem::fn_mult_put;

	m_op.m_k = c;
	m_op.m_n = op->weight();
	m_op.m_stepa = op->stepa(0);
	m_op.m_stepb = op->stepa(1);

	iterator_t begin = loop.begin(), end = loop.end();
	if(begin != end) {
		loop_list_base<2, 1, loop_list_elem>::exec(*this, begin, end, r);
	}
}




void loop_list_elem::fn_mult_add(registers &r) const {

	static const char *method = "fn_mult_add(registers&)";

	if(m_op.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
	if(r.m_ptra[0] + (m_op.m_n - 1) * m_op.m_stepa >=
		r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source1");
	}
	if(r.m_ptra[1] + (m_op.m_n - 1) * m_op.m_stepb >=
		r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source2");
	}
	if(r.m_ptrb[0] + (m_op.m_n - 1) >=
		r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source2");
	}
#endif // LIBTENSOR_DEBUG

	register size_t ia = 0, ib = 0;
	double *pc = r.m_ptrb[0];
	const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
	for (register size_t ic = 0; ic < m_op.m_n; ic++) {
		pc[ic] += pa[ia] * pb[ib] * m_op.m_k;
		ia += m_op.m_stepa;
		ib += m_op.m_stepb;
	}
}


void loop_list_elem::fn_mult_put(registers &r) const {

	static const char *method = "fn_mult_put(registers&)";

	if(m_op.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
	if(r.m_ptra[0] + (m_op.m_n - 1) * m_op.m_stepa >=
		r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source1");
	}
	if(r.m_ptra[1] + (m_op.m_n - 1) * m_op.m_stepb >=
		r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source2");
	}
	if(r.m_ptrb[0] + (m_op.m_n - 1) >=
		r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source2");
	}
#endif // LIBTENSOR_DEBUG

	register size_t ia = 0, ib = 0;
	double *pc = r.m_ptrb[0];
	const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
	for (register size_t ic = 0; ic < m_op.m_n; ic++) {
		pc[ic] = pa[ia] * pb[ib] * m_op.m_k;
		ia += m_op.m_stepa;
		ib += m_op.m_stepb;
	}
}

void loop_list_elem::fn_div_add(registers &r) const {

	static const char *method = "fn_div_add(registers&)";

	if(m_op.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
	if(r.m_ptra[0] + (m_op.m_n - 1) * m_op.m_stepa >=
		r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source1");
	}
	if(r.m_ptra[1] + (m_op.m_n - 1) * m_op.m_stepb >=
		r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source2");
	}
	if(r.m_ptrb[0] + (m_op.m_n - 1) >=
		r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source2");
	}
#endif // LIBTENSOR_DEBUG

	register size_t ia = 0, ib = 0;
	double *pc = r.m_ptrb[0];
	const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
	for (register size_t ic = 0; ic < m_op.m_n; ic++) {
		pc[ic] += m_op.m_k * pa[ia] / pb[ib];
		ia += m_op.m_stepa;
		ib += m_op.m_stepb;
	}
}

void loop_list_elem::fn_div_put(registers &r) const {

	static const char *method = "fn_div_put(registers&)";

	if(m_op.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
	if(r.m_ptra[0] + (m_op.m_n - 1) * m_op.m_stepa >=
		r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source1");
	}
	if(r.m_ptra[1] + (m_op.m_n - 1) * m_op.m_stepb >=
		r.m_ptra_end[1]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source2");
	}
	if(r.m_ptrb[0] + (m_op.m_n - 1) >=
		r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source2");
	}
#endif // LIBTENSOR_DEBUG

	register size_t ia = 0, ib = 0;
	double *pc = r.m_ptrb[0];
	const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
	for (register size_t ic = 0; ic < m_op.m_n; ic++) {
		pc[ic] = m_op.m_k * pa[ia] / pb[ib];
		ia += m_op.m_stepa;
		ib += m_op.m_stepb;
	}
}


} // namespace libtensor
