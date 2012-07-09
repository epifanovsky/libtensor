#include <libtensor/linalg/linalg.h>
#include "kern_dmul2_ij_jp_pi.h"
//#include "kern_mul_ijk_kp_ipj.h"
//#include "kern_mul_ijk_kp_jpi.h"

namespace libtensor {


const char *kern_dmul2_ij_jp_pi::k_clazz = "kern_dmul2_ij_jp_pi";


void kern_dmul2_ij_jp_pi::run(const loop_registers<2, 1> &r) {

    linalg::ij_pi_jp_x(m_ni, m_nj, m_np, r.m_ptra[1], m_spb, r.m_ptra[0],
        m_sja, r.m_ptrb[0], m_sic, m_d);
}


kernel_base<2, 1> *kern_dmul2_ij_jp_pi::match(const kern_dmul2_i_p_pi &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;
    if(z.m_spa != 1) return 0;

    //  Minimize sja > 0:
    //  -----------------
    //  w   a    b    c
    //  ni  0    1    sic
    //  np  1    spb  0
    //  nj  sja  0    1     -->  c_i#j = a_j#p b_p#i
    //  ------------------       [ij_jp_pi]
    //

    iterator_t ij = in.end();
    size_t sja_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) == 1) {
            if(i->stepa(0) % z.m_np) continue;
            if(z.m_sic % i->weight()) continue;
            if(sja_min == 0 || sja_min > i->stepa(0)) {
                ij = i; sja_min = i->stepa(0);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_dmul2_ij_jp_pi zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_np = z.m_np;
    zz.m_sja = ij->stepa(0);
    zz.m_spb = z.m_spb;
    zz.m_sic = z.m_sic;
    in.splice(out.begin(), out, ij);

    kernel_base<2, 1> *kern = 0;

//    if(kern = kern_mul_ijk_kp_ipj::match(zz, in, out)) return kern;
//    if(kern = kern_mul_ijk_kp_jpi::match(zz, in, out)) return kern;

    return new kern_dmul2_ij_jp_pi(zz);
}


} // namespace libtensor
