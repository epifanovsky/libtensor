#include "../../linalg/linalg.h"
#include "kern_mul_ijkl_ijp_plk.h"
#include "kern_mul_ijklm_ikp_jpml.h"

namespace libtensor {


const char *kern_mul_ijkl_ijp_plk::k_clazz = "kern_mul_ijkl_ijp_plk";


void kern_mul_ijkl_ijp_plk::run(const loop_registers<2, 1> &r) {

    for(size_t i = 0; i < m_ni; i++) {
        linalg::ijk_ip_pkj_x(m_nj, m_nk, m_nl, m_np,
            r.m_ptra[0] + i * m_sia, m_sja,
            r.m_ptra[1], m_slb, m_spb,
            r.m_ptrb[0] + i * m_sic, m_skc, m_sjc, m_d);
    }
}


kernel_base<2, 1> *kern_mul_ijkl_ijp_plk::match(const kern_mul_ijk_ip_pkj &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename i -> j, j -> k, k -> l.

    //  Minimize sia > 0:
    //  -----------------
    //  w   a    b    c
    //  nk  0    1    skc
    //  np  1    spb  0
    //  nl  0    slb  1
    //  nj  sja  0    sjc
    //  ni  sia  0    sic  -->  c_i#j#k#l = a_i#j#p b_p#l#k
    //  -----------------       [ijkl_ijp_plk]
    //

    iterator_t ii = in.end();
    size_t sia_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
            if(i->stepa(0) % (z.m_ni * z.m_sia)) continue;
            if(i->stepb(0) % (z.m_ni * z.m_sic)) continue;
            if(sia_min == 0 || sia_min > i->stepa(0)) {
                ii = i; sia_min = i->stepa(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_mul_ijkl_ijp_plk zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_nk = z.m_nj;
    zz.m_nl = z.m_nk;
    zz.m_np = z.m_np;
    zz.m_sia = ii->stepa(0);
    zz.m_sja = z.m_sia;
    zz.m_spb = z.m_spb;
    zz.m_slb = z.m_skb;
    zz.m_sic = ii->stepb(0);
    zz.m_sjc = z.m_sic;
    zz.m_skc = z.m_sjc;
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_mul_ijklm_ikp_jpml::match(zz, in, out)) return kern;

    return new kern_mul_ijkl_ijp_plk(zz);
}


} // namespace libtensor
