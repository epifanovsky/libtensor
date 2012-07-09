#include "../../linalg/linalg.h"
#include "kern_mul_ij_p_pji.h"
#include "kern_mul_ijk_ip_pkj.h"

namespace libtensor {


const char *kern_mul_ij_p_pji::k_clazz = "kern_mul_ij_p_pji";


void kern_mul_ij_p_pji::run(const loop_registers<2, 1> &r) {

    for(size_t j = 0; j < m_nj; j++) {
        linalg::i_pi_p_x(m_ni, m_np, r.m_ptra[1] + j * m_sjb, m_spb,
            r.m_ptra[0], m_spa, r.m_ptrb[0] + j, m_sic, m_d);
    }
}


kernel_base<2, 1> *kern_mul_ij_p_pji::match(const kern_dmul2_i_p_pi &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Minimize sjb > 0:
    //  -----------------
    //  w   a    b    c
    //  ni  0    1    sic
    //  np  spa  spb  0
    //  nj  0    sjb  1    -->  c_i#j = a_p# b_p#j#i
    //  -----------------       [ij_p_pji]
    //

    iterator_t ij = in.end();
    size_t sjb_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) == 1) {
            if(i->stepa(1) % z.m_ni) continue;
            if(z.m_spb % (i->weight() * i->stepa(1))) continue;
            if(sjb_min == 0 || sjb_min > i->stepa(1)) {
                ij = i; sjb_min = i->stepa(1);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_mul_ij_p_pji zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_np = z.m_np;
    zz.m_spa = z.m_spa;
    zz.m_sjb = ij->stepa(1);
    zz.m_spb = z.m_spb;
    zz.m_sic = z.m_sic;
    in.splice(out.begin(), out, ij);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_mul_ijk_ip_pkj::match(zz, in, out)) return kern;

    return new kern_mul_ij_p_pji(zz);
}


} // namespace libtensor
