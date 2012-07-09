#include "../../linalg/linalg.h"
#include "kern_mul_ijk_pi_pkj.h"

namespace libtensor {


const char *kern_mul_ijk_pi_pkj::k_clazz = "kern_mul_ijk_pi_pkj";


void kern_mul_ijk_pi_pkj::run(const loop_registers<2, 1> &r) {

    linalg::ijk_pi_pkj_x(m_ni, m_nj, m_nk, m_np, r.m_ptra[0], m_spa,
        r.m_ptra[1], m_skb, m_spb, r.m_ptrb[0], m_sjc, m_sic, m_d);
}


kernel_base<2, 1> *kern_mul_ijk_pi_pkj::match(const kern_dmul2_i_pi_p &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Minimize sjc > 0, skb > 0:
    //  -------------------
    //  w   a    b     c
    //  ni  1    0     sic
    //  np  spa  spb   0
    //  nj  0    1     sjc
    //  nk  0    skb   1    -->  c_i#j#k = a_p#i b_p#k#j
    //  -------------------       [ijk_pi_pkj]
    //

    iterator_t ij = in.end(), ik = in.end();
    size_t sjc_min = 0, skb_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) == 1 && i->stepb(0) > 0) {
            if(z.m_spb % i->weight()) continue;
            if(z.m_sic % (i->weight() * i->stepb(0))) continue;
            if(sjc_min == 0 || sjc_min > i->stepb(0)) {
                ij = i; sjc_min = i->stepb(0);
            }
        }
    }
    if(ij == in.end()) return 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i == ij) continue;
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) == 1) {
            if(z.m_spb % (i->weight() * i->stepa(1))) continue;
            if(i->stepa(1) % ij->weight()) continue;
            if(ij->stepb(0) % i->weight()) continue;
            if(skb_min == 0 || skb_min > i->stepa(1)) {
                ik = i; skb_min = i->stepa(1);
            }
        }
    }
    if(ik == in.end()) return 0;

    kern_mul_ijk_pi_pkj zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_nk = ik->weight();
    zz.m_np = z.m_np;
    zz.m_spa = z.m_spa;
    zz.m_spb = z.m_spb;
    zz.m_skb = ik->stepa(1);
    zz.m_sic = z.m_sic;
    zz.m_sjc = ij->stepb(0);
    in.splice(out.begin(), out, ij);
    in.splice(out.begin(), out, ik);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ijk_pi_pkj(zz);
}


} // namespace libtensor
