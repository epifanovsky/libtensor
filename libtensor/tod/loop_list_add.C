#include "../defs.h"
#include "../exception.h"
#include "../linalg.h"
#include "loop_list_add.h"

namespace libtensor {


const char *loop_list_add::k_clazz = "loop_list_add";


void loop_list_add::run_loop(list_t &loop, registers &r, double c) {

	install_kernel(loop, c);

	iterator_t begin = loop.begin(), end = loop.end();
	if(begin != end) {
		loop_list_base<1, 1, loop_list_add>::exec(
			*this, begin, end, r);
	}
}


void loop_list_add::install_kernel(list_t &loop, double c) {

	//
	//	Try to find the kernel loop using priority:
	//
	//	step(A) = 1, step(B) = 1
	//	step(A) = 1, step(B) = any
	//	step(A) = any, step(B) = 1
	//	step(A) = any, step(B) = any
	//
	iterator_t ia = loop.end(), ib = loop.end(), iab = loop.end();
	for(iterator_t i = loop.begin(); i != loop.end(); i++) {
		i->fn() = 0;
		if(i->stepa(0) == 1) {
			ia = i;
			if(i->stepb(0) == 1) {
				ib = i;
				iab = i;
			}
		} else {
			if(i->stepb(0) == 1) {
				ib = i;
			}
		}
	}

	iterator_t ikernel;
	if(iab != loop.end()) {
		ikernel = iab;
	} else if(ia != loop.end()) {
		ikernel = ia;
	} else if(ib != loop.end()) {
		ikernel = ib;
	} else {
		ikernel = loop.begin();
	}

	if(ikernel != loop.end()) {
		ikernel->fn() = &loop_list_add::fn_daxpy;
		m_add.m_k = c;
		m_add.m_n = ikernel->weight();
		m_add.m_stepa = ikernel->stepa(0);
		m_add.m_stepb = ikernel->stepb(0);
		loop.splice(loop.end(), loop, ikernel);
	}
}


void loop_list_add::fn_daxpy(registers &r) const {

	static const char *method = "fn_daxpy(registers&)";

	if(m_add.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
	if(r.m_ptra[0] + (m_add.m_n - 1) * m_add.m_stepa >= r.m_ptra_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"source");
	}
	if(r.m_ptrb[0] + (m_add.m_n - 1) * m_add.m_stepb >= r.m_ptrb_end[0]) {
		throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
			"destination");
	}
#endif // LIBTENSOR_DEBUG

	blas_daxpy(m_add.m_n, m_add.m_k, r.m_ptra[0], m_add.m_stepa,
		r.m_ptrb[0], m_add.m_stepb);
}


} // namespace libtensor
