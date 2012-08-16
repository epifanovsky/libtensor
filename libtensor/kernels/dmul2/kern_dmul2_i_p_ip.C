#include <libtensor/linalg/linalg.h>
#include "kern_dmul2_i_p_ip.h"
#include "kern_dmul2_ij_ip_jp.h"

namespace libtensor {


const char *kern_dmul2_i_p_ip::k_clazz = "kern_dmul2_i_p_ip";


void kern_dmul2_i_p_ip::run(const loop_registers<2, 1> &r) {

    linalg::i_ip_p_x(m_ni, m_np, r.m_ptra[1], m_sib, r.m_ptra[0], m_spa,
        r.m_ptrb[0], m_sic, m_d);
}


kernel_base<2, 1> *kern_dmul2_i_p_ip::match(const kern_dmul2_x_p_p &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_spa != 1) return 0;

    //  2. If spa == 1, minimize sib > 0:
    //  ----------------
    //  w   a  b       c
    //  np  1  1       0
    //  ni  0  k2b*np  1  -->  c_i = a_p b_i%p
    //  ----------------       sz(i) = w2, sz(p) = np, sz(%) = k2b
    //                         [i_p_ip]

    iterator_t ii = in.end();
    size_t sib_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) == 1) {
            if(i->stepa(1) % z.m_np) continue;
            if(sib_min == 0 || sib_min > i->stepa(1)) {
                ii = i; sib_min = i->stepa(1);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_dmul2_i_p_ip zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_np = z.m_np;
    zz.m_spa = z.m_spa;
    zz.m_sib = ii->stepa(1);
    zz.m_sic = ii->stepb(0);
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_dmul2_ij_ip_jp::match(zz, in, out)) return kern;

    return new kern_dmul2_i_p_ip(zz);
}


} // namespace libtensor
