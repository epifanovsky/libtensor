#include "../../linalg/linalg.h"
#include "kern_mul_ijk_pkiq_pjq.h"
#include "kern_mul_ijkl_pliq_jpkq.h"
#include "kern_mul_ijkl_pljq_ipkq.h"
#include "kern_mul_ijkl_pljq_pikq.h"

namespace libtensor {


const char *kern_mul_ijk_pkiq_pjq::k_clazz = "kern_mul_ijk_pkiq_pjq";


void kern_mul_ijk_pkiq_pjq::run(const loop_registers<2, 1> &r) {

	const double *pa = r.m_ptra[0], *pb = r.m_ptra[1];
	double *pc = r.m_ptrb[0];
	for(size_t i = 0; i < m_ni; i++) {
		const double *pa1 = pa, *pb1 = pb;
		for(size_t p = 0; p < m_np; p++) {
			linalg::ij_ip_jp_x(m_nj, m_nk, m_nq, pb1, m_sjb,
				pa1, m_ska, pc, m_sjc, m_d);
			pa1 += m_spa;
			pb1 += m_spb;
		}
		pa += m_sia;
		pc += m_sic;
	}
}


kernel_base<2, 1> *kern_mul_ijk_pkiq_pjq::match(const kern_mul_ij_pjq_piq &z,
	list_t &in, list_t &out) {

	if(in.empty()) return 0;

	//	Rename i -> j, j -> k

	//	Minimize sia > 0:
	//	-----------------
	//	w   a    b    c
	//	nq  1    1    0
	//	nk  ska  0    1
	//	nj  0    sjb  sjc
	//	np  spa  spb  0
	//	ni  sia  0    sic  -->  c_i#j#k = a_p#k#i#q b_p#j#q
	//	-----------------       [ijk_pkiq_pjq]

	iterator_t ii = in.end();
	size_t sia_min = 0;
	for(iterator_t i = in.begin(); i != in.end(); i++) {
		if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
			if(i->stepa(0) % z.m_nq ||
				z.m_sja % (i->weight() * i->stepa(0)))
				continue;
			if(i->stepb(0) % (z.m_sic * z.m_ni)) continue;
			if(sia_min == 0 || sia_min > i->stepa(0)) {
				ii = i; sia_min = i->stepa(0);
			}
		}
	}
	if(ii == in.end()) return 0;

	kern_mul_ijk_pkiq_pjq zz;
	zz.m_d = z.m_d;
	zz.m_ni = ii->weight();
	zz.m_nj = z.m_ni;
	zz.m_nk = z.m_nj;
	zz.m_np = z.m_np;
	zz.m_nq = z.m_nq;
	zz.m_spa = z.m_spa;
	zz.m_ska = z.m_sja;
	zz.m_sia = ii->stepa(0);
	zz.m_spb = z.m_spb;
	zz.m_sjb = z.m_sib;
	zz.m_sic = ii->stepb(0);
	zz.m_sjc = z.m_sic;
	in.splice(out.begin(), out, ii);

	kernel_base<2, 1> *kern = 0;

	if(kern = kern_mul_ijkl_pliq_jpkq::match(zz, in, out)) return kern;
	if(kern = kern_mul_ijkl_pljq_ipkq::match(zz, in, out)) return kern;
	if(kern = kern_mul_ijkl_pljq_pikq::match(zz, in, out)) return kern;

	return new kern_mul_ijk_pkiq_pjq(zz);
}


} // namespace libtensor
