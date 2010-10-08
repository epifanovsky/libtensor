#include "../../linalg/linalg.h"
#include "kern_mul_ijk_pkq_ipqj.h"


namespace libtensor {


const char *kern_mul_ijk_pkq_ipqj::k_clazz = "kern_mul_ijk_pkq_ipqj";


void kern_mul_ijk_pkq_ipqj::run(const loop_registers<2, 1> &r) {

	const double *pb = r.m_ptra[1];
	double *pc = r.m_ptrb[0];

	for(size_t i = 0; i < m_ni; i++) {

		const double *pa1 = r.m_ptra[0], *pb1 = pb;

		for(size_t p = 0; p < m_np; p++) {
			linalg::ij_pi_jp_x(m_nj, m_nk, m_nq, pb1, m_sqb,
				pa1, m_ska, pc, m_sjc, m_d);
			pa1 += m_spa;
			pb1 += m_spb;
		}

		pb += m_sib;
		pc += m_sic;
	}
}


kernel_base<2, 1> *kern_mul_ijk_pkq_ipqj::match(const kern_mul_ijk_kp_ipj &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename p -> q.

	//	Minimize spb > 0:
	//	-----------------
	//	w   a    b    c
	//	nj  0    1    sjc
	//	nq  1    sqb  0
	//	nk  ska  0    1
	//	ni  0    sib  sic
	//	np  spa  spb  0    -->  c_i#j#k = a_p#k#q b_i#p#q#j
	//	-----------------       [ijk_pkq_ipqj]
	//

	iterator_t ip = in.end();
	size_t spb_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) > 0 && i->stepb(0) == 0) {
			if(i->stepa(0) % (z.m_ska * z.m_nk)) continue;
			if(i->stepa(1) % (z.m_spb * z.m_np) ||
				z.m_sib % (i->weight() * i->stepa(1)))
				continue;
			if(spb_min == 0 || spb_min > i->stepa(1)) {
				ip = i; spb_min = i->stepa(1);
			}
		}
	}
	if(ip == in.end()) return 0;

	kern_mul_ijk_pkq_ipqj zz;
	zz.m_d = z.m_d;
	zz.m_ni = z.m_ni;
	zz.m_nj = z.m_nj;
	zz.m_nk = z.m_nk;
	zz.m_np = ip->weight();
	zz.m_nq = z.m_np;
	zz.m_spa = ip->stepa(0);
	zz.m_ska = z.m_ska;
	zz.m_sib = z.m_sib;
	zz.m_spb = ip->stepa(1);
	zz.m_sqb = z.m_spb;
	zz.m_sic = z.m_sic;
	zz.m_sjc = z.m_sjc;
	in.splice(out.begin(), out, ip);

	kernel_base<2, 1> *kern = 0;

	return new kern_mul_ijk_pkq_ipqj(zz);
}


} // namespace libtensor
