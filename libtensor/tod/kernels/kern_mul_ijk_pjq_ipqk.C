#include "../../linalg/linalg.h"
#include "kern_mul_ijk_pjq_ipqk.h"
#include "kern_mul_ijkl_pkjq_ipql.h"

namespace libtensor {


const char *kern_mul_ijk_pjq_ipqk::k_clazz = "kern_mul_ijk_pjq_ipqk";


void kern_mul_ijk_pjq_ipqk::run(const loop_registers<2, 1> &r) {

	const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
	double *pc = r.m_ptrb[0];

	for(size_t p = 0; p < m_np; p++) {
		const double *pb1 = pb;
		double *pc1 = pc;
		for(size_t i = 0; i < m_ni; i++) {
			linalg::ij_ip_pj_x(m_nj, m_nk, m_nq, pa, m_sja,
				pb1, m_sqb, pc1, m_sjc, m_d);
			pb1 += m_sib;
			pc1 += m_sic;
		}
		pa += m_spa;
		pb += m_spb;
	}
}


kernel_base<2, 1> *kern_mul_ijk_pjq_ipqk::match(const kern_mul_ijk_jp_ipk &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename p -> q.

	//	Minimize spb > 0.
	//	-----------------
	//	w   a    b    c
	//	nk  0    1    1
	//	nq  1    sqb  0
	//	nj  sja  0    sjc
	//	ni  0    sib  sic
	//	np  spa  spb  0    --> c_i#j#k = a_p#j#q b_i#p#q#k
	//	-----------------      [ijk_pjq_ipqk]
	//

	iterator_t ip = in.end();
	size_t spb_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) > 0 && i->stepb(0) == 0) {
			if(i->stepa(0) % (z.m_sja * z.m_nj)) continue;
			if(i->stepa(1) % (z.m_spb * z.m_np) ||
				z.m_sib % (i->weight() * i->stepa(1)))
				continue;
			if(spb_min == 0 || spb_min > i->stepa(1)) {
				ip = i; spb_min = i->stepa(1);
			}
		}
	}
	if(ip == in.end()) return 0;

	kern_mul_ijk_pjq_ipqk zz;
	zz.m_d = z.m_d;
	zz.m_ni = z.m_ni;
	zz.m_nj = z.m_nj;
	zz.m_nk = z.m_nk;
	zz.m_np = ip->weight();
	zz.m_nq = z.m_np;
	zz.m_spa = ip->stepa(0);
	zz.m_sja = z.m_sja;
	zz.m_sib = z.m_sib;
	zz.m_spb = ip->stepa(1);
	zz.m_sqb = z.m_spb;
	zz.m_sic = z.m_sic;
	zz.m_sjc = z.m_sjc;
	in.splice(out.begin(), out, ip);

	kernel_base<2, 1> *kern = 0;

	if(kern = kern_mul_ijkl_pkjq_ipql::match(zz, in, out)) return kern;

	return new kern_mul_ijk_pjq_ipqk(zz);
}


} // namespace libtensor
