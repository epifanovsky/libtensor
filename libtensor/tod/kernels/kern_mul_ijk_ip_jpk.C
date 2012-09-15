#include "../../linalg/linalg.h"
#include "kern_mul_ijk_ip_jpk.h"
#include "kern_mul_ijk_piq_jpqk.h"
#include "kern_mul_ijk_piq_pjqk.h"

namespace libtensor {


const char *kern_mul_ijk_ip_jpk::k_clazz = "kern_mul_ijk_ip_jpk";


void kern_mul_ijk_ip_jpk::run(const loop_registers<2, 1> &r) {

    const double *pb = r.m_ptra[1];
    double *pc = r.m_ptrb[0];

    for(size_t j = 0; j < m_nj; j++) {
        linalg::mul2_ij_ip_pj_x(m_ni, m_nk, m_np, r.m_ptra[0], m_sia,
            pb, m_spb, pc, m_sic, m_d);
        pb += m_sjb;
        pc += m_sjc;
    }
}


kernel_base<2, 1> *kern_mul_ijk_ip_jpk::match(const kern_dmul2_ij_ip_pj &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename j -> k.

    //  Minimize sjc > 0.
    //  -----------------
    //  w   a    b    c
    //  nk  0    1    1
    //  np  1    spb  0
    //  ni  sia  0    sic
    //  nj  0    sjb  sjc  --> c_i#j#k = a_i#p b_j#p#k
    //  -----------------      [ijk_ip_jpk]
    //

    iterator_t ij = in.end();
    size_t sjc_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
            if(i->stepa(1) % (z.m_spb * z.m_np)) continue;
            if(i->stepb(0) % z.m_nj ||
                z.m_sic % (i->weight() * i->stepb(0)))
                continue;
            if(sjc_min == 0 || sjc_min > i->stepb(0)) {
                ij = i; sjc_min = i->stepb(0);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_mul_ijk_ip_jpk zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_nk = z.m_nj;
    zz.m_np = z.m_np;
    zz.m_sia = z.m_sia;
    zz.m_sjb = ij->stepa(1);
    zz.m_spb = z.m_spb;
    zz.m_sic = z.m_sic;
    zz.m_sjc = ij->stepb(0);
    in.splice(out.begin(), out, ij);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_mul_ijk_piq_jpqk::match(zz, in, out)) return kern;
    if(kern = kern_mul_ijk_piq_pjqk::match(zz, in, out)) return kern;

    return new kern_mul_ijk_ip_jpk(zz);
}


} // namespace libtensor
