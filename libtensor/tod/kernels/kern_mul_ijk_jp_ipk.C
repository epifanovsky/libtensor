#include "../../linalg/linalg.h"
#include "kern_mul_ijk_jp_ipk.h"
#include "kern_mul_ijk_pjq_ipqk.h"
#include "kern_mul_ijk_pjq_piqk.h"

namespace libtensor {


const char *kern_mul_ijk_jp_ipk::k_clazz = "kern_mul_ijk_jp_ipk";


void kern_mul_ijk_jp_ipk::run(const loop_registers<2, 1> &r) {

	const double *pb = r.m_ptra[1];
	double *pc = r.m_ptrb[0];

	for(size_t i = 0; i < m_ni; i++) {
		linalg::ij_ip_pj_x(m_nj, m_nk, m_np, r.m_ptra[0], m_sja,
			pb, m_spb, pc, m_sjc, m_d);
		pb += m_sib;
		pc += m_sic;
	}
}


kernel_base<2, 1> *kern_mul_ijk_jp_ipk::match(const kern_mul_ij_ip_pj &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename i -> j, j -> k.

	//	Minimize sib > 0.
	//	-----------------
	//	w   a    b    c
	//	nk  0    1    1
	//	np  1    spb  0
	//	nj  sja  0    sjc
	//	ni  0    sib  sic  --> c_i#j#k = a_j#p b_i#p#k
	//	-----------------      [ijk_jp_ipk]
	//

	iterator_t ii = in.end();
	size_t sib_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
			if(i->stepa(1) % (z.m_spb * z.m_np)) continue;
			if(i->stepb(0) % (z.m_sic * z.m_ni)) continue;
			if(sib_min == 0 || sib_min > i->stepa(1)) {
				ii = i; sib_min = i->stepa(1);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_ijk_jp_ipk zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_nj = z.m_ni;
	zz.m_nk = z.m_nj;
	zz.m_np = z.m_np;
	zz.m_sja = z.m_sia;
	zz.m_sib = ii->stepa(1);
	zz.m_spb = z.m_spb;
	zz.m_sic = ii->stepb(0);
	zz.m_sjc = z.m_sic;
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	if(kern = kern_mul_ijk_pjq_ipqk::match(zz, in, out)) return kern;
	if(kern = kern_mul_ijk_pjq_piqk::match(zz, in, out)) return kern;

	return new kern_mul_ijk_jp_ipk(zz);
}


} // namespace libtensor
