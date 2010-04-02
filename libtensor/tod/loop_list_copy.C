#include "../exception.h"
#include "../linalg.h"
#include "loop_list_copy.h"

namespace libtensor {


const char *loop_list_copy::k_clazz = "loop_list_copy";


void loop_list_copy::run_loop(list_t &loop, registers &regs, double c) {

	//
	//	Install the kernel on the fastest-running index in A
	//
	iterator_t i;
	for(i = loop.begin(); i != loop.end(); i++) i->fn() = 0;
	for(i = loop.begin(); i != loop.end() &&
		!(i->stepa() == 1 && i->stepb() == 1); i++);
	if(i == loop.end()) {
		for(i = loop.begin(); i != loop.end() && i->stepa() != 1; i++);
	}
	if(i == loop.end()) i = loop.begin();
	if(i != loop.end()) {
		i->fn() = &loop_list_copy::fn_copy;
		m_copy.m_k = c;
		m_copy.m_n = i->weight();
		m_copy.m_stepa = i->stepa();
		m_copy.m_stepb = i->stepb();
		loop.splice(loop.end(), loop, i);
	}

	//
	//	Run the nested loops
	//
	i = loop.begin();
	iterator_t iend = loop.end();
	exec(i, iend, regs);
}


void loop_list_copy::exec(iterator_t &i, iterator_t &iend, registers &r) {

	node::fnptr_t fn = i->fn();

	if(fn == 0) fn_loop(i, iend, r);
	else (this->*fn)(r);
}


void loop_list_copy::fn_loop(iterator_t &i, iterator_t &iend, registers &r) {

	iterator_t j = i; j++;
	if(j == iend) return;

	const double *ptra = r.m_ptra;
	double *ptrb = r.m_ptrb;

	for(size_t k = 0; k < i->weight(); k++) {

		exec(j, iend, r);
		ptra += i->stepa(); r.m_ptra = ptra;
		ptrb += i->stepb(); r.m_ptrb = ptrb;
	}
}


void loop_list_copy::fn_copy(registers &r) const {

	static const char *method = "fn_copy(registers&)";

	if(m_copy.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
	if(r.m_ptra + (m_copy.m_n - 1) * m_copy.m_stepa >= r.m_ptra_end) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source");
	}
	if(r.m_ptrb + (m_copy.m_n - 1) * m_copy.m_stepb >= r.m_ptrb_end) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	blas_dcopy(m_copy.m_n, r.m_ptra, m_copy.m_stepa,
		r.m_ptrb, m_copy.m_stepb);
	if(m_copy.m_k != 1.0) {
		blas_dscal(m_copy.m_n, m_copy.m_k, r.m_ptrb, m_copy.m_stepb);
	}
}


} // namespace libtensor
