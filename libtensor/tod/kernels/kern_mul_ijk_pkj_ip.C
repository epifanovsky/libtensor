#include "../../linalg/linalg.h"
#include "kern_mul_ijk_pkj_ip.h"

namespace libtensor {


const char *kern_mul_ijk_pkj_ip::k_clazz = "kern_mul_ijk_pkj_ip";


void kern_mul_ijk_pkj_ip::run(const loop_registers<2, 1> &r) {

    linalg::ijk_ip_pkj_x(m_ni, m_nj, m_nk, m_np, r.m_ptra[1], m_sib,
        r.m_ptra[0], m_ska, m_spa, r.m_ptrb[0], m_sjc, m_sic, m_d);
}


kernel_base<2, 1> *kern_mul_ijk_pkj_ip::match(const kern_mul_ij_pji_p &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_spb != 1) return 0;

    //  Rename i -> j, j -> k.

    //  Minimize sib > 0.
    //  -------------------
    //  w   a    b     c
    //  nj  1    0     sjc
    //  np  spa  1     0
    //  nk  ska  0     1
    //  ni  0    sib   sic   -->  c_i#j#k = a_p#k#j b_i#p
    //  -------------------       [ijk_pkj_ip]
    //

    iterator_t ii = in.end();
    size_t sib_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
            if(i->stepa(1) % z.m_np) continue;
            if(i->stepb(0) % (z.m_sic * z.m_ni)) continue;
            if(sib_min == 0 || sib_min > i->stepa(1)) {
                ii = i; sib_min = i->stepa(1);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_mul_ijk_pkj_ip zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_nk = z.m_nj;
    zz.m_np = z.m_np;
    zz.m_spa = z.m_spa;
    zz.m_ska = z.m_sja;
    zz.m_sib = ii->stepa(1);
    zz.m_sic = ii->stepb(0);
    zz.m_sjc = z.m_sic;
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ijk_pkj_ip(zz);
}


} // namespace libtensor
