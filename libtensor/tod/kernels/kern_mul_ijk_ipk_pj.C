#include "../../linalg/linalg.h"
#include "kern_mul_ijk_ipk_pj.h"
#include "kern_mul_ijkl_ipl_jpk.h"

namespace libtensor {


const char *kern_mul_ijk_ipk_pj::k_clazz = "kern_mul_ijk_ipk_pj";


void kern_mul_ijk_ipk_pj::run(const loop_registers<2, 1> &r) {

	const double *pa = r.m_ptra[0];
	double *pc = r.m_ptrb[0];

	for(size_t i = 0; i < m_ni; i++) {
		linalg::ij_pi_pj_x(m_nj, m_nk, m_np, r.m_ptra[1], m_spb, pa,
			m_spa, pc, m_sjc, m_d);
		pa += m_sia;
		pc += m_sic;
	}
}


kernel_base<2, 1> *kern_mul_ijk_ipk_pj::match(const kern_mul_ij_pj_pi &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename i -> j, j -> k

	//	Minimize sic > 0:
	//	------------------
	//	w   a    b    c
	//	nk  1    0    1
	//	np  spa  spb  0
	//	nj  0    1    sjc
	//	ni  sia  0    sic  --> c_i#j#k = a_i#p#k b_p#j
	//	-----------------      [ijk_ipk_pj]
	//

	iterator_t ii = in.end();
	size_t sic_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
			if(i->stepa(0) % z.m_spa) continue;
			if(i->stepb(0) % z.m_sic) continue;
			if(sic_min == 0 || sic_min > i->stepb(0)) {
				ii = i; sic_min = i->stepb(0);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_ijk_ipk_pj zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_nj = z.m_ni;
	zz.m_nk = z.m_nj;
	zz.m_np = z.m_np;
	zz.m_sia = ii->stepa(0);
	zz.m_spa = z.m_spa;
	zz.m_spb = z.m_spb;
	zz.m_sic = ii->stepb(0);
	zz.m_sjc = z.m_sic;
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	if(kern = kern_mul_ijkl_ipl_jpk::match(zz, in, out)) return kern;

	return new kern_mul_ijk_ipk_pj(zz);
}


} // namespace libtensor
