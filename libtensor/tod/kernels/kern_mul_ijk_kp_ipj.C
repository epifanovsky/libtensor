#include "../../linalg/linalg.h"
#include "kern_mul_ijk_kp_ipj.h"
#include "kern_mul_ijk_pkq_ipqj.h"
#include "kern_mul_ijk_pkq_piqj.h"


namespace libtensor {


const char *kern_mul_ijk_kp_ipj::k_clazz = "kern_mul_ijk_kp_ipj";


void kern_mul_ijk_kp_ipj::run(const loop_registers<2, 1> &r) {

    const double *pb = r.m_ptra[1];
    double *pc = r.m_ptrb[0];

    for(size_t i = 0; i < m_ni; i++) {
        linalg::ij_pi_jp_x(m_nj, m_nk, m_np, pb, m_spb, r.m_ptra[0],
            m_ska, pc, m_sjc, m_d);
        pb += m_sib;
        pc += m_sic;
    }
}


kernel_base<2, 1> *kern_mul_ijk_kp_ipj::match(const kern_dmul2_ij_jp_pi &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename i -> j, j -> k.

    //  Minimize sjc > 0:
    //  -----------------
    //  w   a    b    c
    //  nj  0    1    sjc
    //  np  1    spb  0
    //  nk  ska  0    1
    //  ni  0    sib  sic  -->  c_i#j#k = a_k#p b_i#p#j
    //  -----------------       [ijk_kp_ipj]
    //

    iterator_t ii = in.end();
    size_t sic_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) == 0 && i->stepa(1) > 0 && i->stepb(0) > 0) {
            if(i->stepa(1) % (z.m_spb * z.m_np)) continue;
            if(i->stepb(0) % (z.m_sic * z.m_ni)) continue;
            if(sic_min == 0 || sic_min > i->stepb(0)) {
                ii = i; sic_min = i->stepb(0);
            }
        }
    }
    if(ii == in.end()) return 0;

    kern_mul_ijk_kp_ipj zz;
    zz.m_d = z.m_d;
    zz.m_ni = ii->weight();
    zz.m_nj = z.m_ni;
    zz.m_nk = z.m_nj;
    zz.m_np = z.m_np;
    zz.m_ska = z.m_sja;
    zz.m_sib = ii->stepa(1);
    zz.m_spb = z.m_spb;
    zz.m_sic = ii->stepb(0);
    zz.m_sjc = z.m_sic;
    in.splice(out.begin(), out, ii);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_mul_ijk_pkq_ipqj::match(zz, in, out)) return kern;
    if(kern = kern_mul_ijk_pkq_piqj::match(zz, in, out)) return kern;

    return new kern_mul_ijk_kp_ipj(zz);
}


} // namespace libtensor
