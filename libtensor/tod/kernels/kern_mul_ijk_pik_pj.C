#include "../../linalg/linalg.h"
#include "kern_mul_ijk_pik_pj.h"

namespace libtensor {


const char *kern_mul_ijk_pik_pj::k_clazz = "kern_mul_ijk_pik_pj";


void kern_mul_ijk_pik_pj::run(const loop_registers<2, 1> &r) {

    for(size_t i = 0; i < m_ni; i++) {
        linalg::mul2_ij_pi_pj_x(m_nj, m_nk, m_np,
            r.m_ptra[1], m_spb,
            r.m_ptra[0] + i * m_sia, m_spa,
            r.m_ptrb[0] + i * m_sic, m_sjc, m_d);
    }
}


kernel_base<2, 1> *kern_mul_ijk_pik_pj::match(const kern_dmul2_ij_pj_pi &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename i -> j, j -> k

    //  Minimize sia > 0:
    //  ------------------
    //  w   a    b    c
    //  nk  1    0    1
    //  np  spa  spb  0
    //  nj  0    1    sjc
    //  ni  sia  0    sic  --> c_i#j#k = a_p#i#k b_p#j
    //  -----------------      [ijk_pik_pj]
    //

    iterator_t ii = in.end();
    size_t sia_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
            if(i->stepa(0) % z.m_nj) continue;
            if(z.m_spa % (i->weight() * i->stepa(0))) continue;
            if(i->stepb(0) % (z.m_ni * z.m_sic)) continue;
            if(sia_min == 0 || sia_min > i->stepa(0)) {
                ii = i; sia_min = i->stepa(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_mul_ijk_pik_pj zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_nk = z.m_nj;
    zz.m_np = z.m_np;
    zz.m_spa = z.m_spa;
    zz.m_sia = ii->stepa(0);
    zz.m_spb = z.m_spb;
    zz.m_sic = ii->stepb(0);
    zz.m_sjc = z.m_sic;
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    return new kern_mul_ijk_pik_pj(zz);
}


} // namespace libtensor
