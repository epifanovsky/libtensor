#include "../../linalg/linalg.h"
#include "kern_mul_ijk_pqj_iqpk.h"
#include "kern_mul_ijkl_pqkj_iqpl.h"

namespace libtensor {


const char *kern_mul_ijk_pqj_iqpk::k_clazz = "kern_mul_ijk_pqj_iqpk";


void kern_mul_ijk_pqj_iqpk::run(const loop_registers<2, 1> &r) {

	for(size_t i = 0; i < m_ni; i++)
	for(size_t q = 0; q < m_nq; q++) {
		linalg::ij_pi_pj_x(m_nj, m_nk, m_np,
			r.m_ptra[0] + q * m_sqa, m_spa,
			r.m_ptra[1] + i * m_sib + q * m_sqb, m_spb,
			r.m_ptrb[0] + i * m_sic, m_sjc, m_d);
	}
}


kernel_base<2, 1> *kern_mul_ijk_pqj_iqpk::match(const kern_mul_ijk_pj_ipk &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Minimize sqa > 0:
	//	------------------
	//	w   a    b     c
	//	nj  1    0     sjc
	//	np  spa  spb   0
	//	nk  0    1     1
	//	ni  0    sib   sic
	//	nq  sqa  sqb   0    -->  c_i#j#k = a_p#q#j b_i#q#p#k
	//	------------------       [ijk_pqj_iqpk]
	//

	iterator_t iq = in.end();
	size_t sqa_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) > 0 && i->stepb(0) == 0) {
			if(i->stepa(0) % z.m_nj) continue;
			if(z.m_spa % (i->weight() * i->stepa(0))) continue;
			if(i->stepa(1) % (z.m_np * z.m_spb)) continue;
			if(z.m_sib % (i->weight() * i->stepa(1))) continue;
			if(sqa_min == 0 || sqa_min > i->stepa(0)) {
				iq = i; sqa_min = i->stepa(0);
			}
		}
	}
	if(iq == in.end()) return 0;

	kern_mul_ijk_pqj_iqpk zz;
	zz.m_d = z.m_d;
	zz.m_ni = z.m_ni;
	zz.m_nj = z.m_nj;
	zz.m_nk = z.m_nk;
	zz.m_np = z.m_np;
	zz.m_nq = iq->weight();
	zz.m_spa = z.m_spa;
	zz.m_sqa = iq->stepa(0);
	zz.m_sib = z.m_sib;
	zz.m_sqb = iq->stepa(1);
	zz.m_spb = z.m_spb;
	zz.m_sic = z.m_sic;
	zz.m_sjc = z.m_sjc;
	in.splice(out.begin(), out, iq);

	kernel_base<2, 1> *kern = 0;

	if(kern = kern_mul_ijkl_pqkj_iqpl::match(zz, in, out)) return kern;

	return new kern_mul_ijk_pqj_iqpk(zz);
}


} // namespace libtensor
