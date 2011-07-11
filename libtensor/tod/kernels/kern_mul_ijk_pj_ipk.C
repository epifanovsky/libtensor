#include "../../linalg/linalg.h"
#include "kern_mul_ijk_pj_ipk.h"
#include "kern_mul_ijk_pqj_iqpk.h"
#include "kern_mul_ijkl_ipk_jpl.h"

namespace libtensor {


const char *kern_mul_ijk_pj_ipk::k_clazz = "kern_mul_ijk_pj_ipk";


void kern_mul_ijk_pj_ipk::run(const loop_registers<2, 1> &r) {

	for(size_t i = 0; i < m_ni; i++) {
		linalg::ij_pi_pj_x(m_nj, m_nk, m_np,
			r.m_ptra[0], m_spa,
			r.m_ptra[1] + i * m_sib, m_spb,
			r.m_ptrb[0] + i * m_sic, m_sjc, m_d);
	}
}


kernel_base<2, 1> *kern_mul_ijk_pj_ipk::match(const kern_mul_ij_pi_pj &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename i -> j, j -> k.

	//	Minimize sjb > 0:
	//	------------------
	//	w   a    b     c
	//	nj  1    0     sjc
	//	np  spa  spb   0
	//	nk  0    1     1
	//	ni  0    sib   sic  -->  c_i#j#k = a_p#j b_i#p#k
	//	------------------       [ijk_pj_ipk]
	//

	iterator_t ii = in.end();
	size_t sib_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
			if(i->stepa(1) % (z.m_np * z.m_spb)) continue;
			if(i->stepb(0) % (z.m_ni * z.m_sic)) continue;
			if(sib_min == 0 || sib_min > i->stepa(1)) {
				ii = i; sib_min = i->stepa(1);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_ijk_pj_ipk zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_nj = z.m_ni;
	zz.m_nk = z.m_nj;
	zz.m_np = z.m_np;
	zz.m_spa = z.m_spa;
	zz.m_sib = ii->stepa(1);
	zz.m_spb = z.m_spb;
	zz.m_sic = ii->stepb(0);
	zz.m_sjc = z.m_sic;
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	if(kern = kern_mul_ijk_pqj_iqpk::match(zz, in, out)) return kern;
	if(kern = kern_mul_ijkl_ipk_jpl::match(zz, in, out)) return kern;

	return new kern_mul_ijk_pj_ipk(zz);
}


} // namespace libtensor
