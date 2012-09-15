#include "../../linalg/linalg.h"
#include "kern_mul_ijk_pji_pk.h"
#include "kern_mul_ijk_pjqi_qpk.h"
#include "kern_mul_ijk_pqji_pqk.h"
#include "kern_mul_ijk_pqji_qpk.h"

namespace libtensor {


const char *kern_mul_ijk_pji_pk::k_clazz = "kern_mul_ijk_pji_pk";


void kern_mul_ijk_pji_pk::run(const loop_registers<2, 1> &r) {

    const double *pa = r.m_ptra[0];
    double *pc = r.m_ptrb[0];

    for(size_t j = 0; j < m_nj; j++) {
        linalg::mul2_ij_pi_pj_x(m_ni, m_nk, m_np, pa, m_spa, r.m_ptra[1],
            m_spb, pc, m_sic, m_d);
        pa += m_sja;
        pc += m_sjc;
    }
}


kernel_base<2, 1> *kern_mul_ijk_pji_pk::match(const kern_dmul2_ij_pi_pj &z,
    list_t &in, list_t &out) {

    if(in.empty()) return 0;

    //  Rename j -> k.

    //  Minimize sja > 0:
    //  ------------------
    //  w   a    b     c
    //  ni  1    0     sic
    //  np  spa  spb   0
    //  nk  0    1     1
    //  nj  sja  0     sjc  -->  c_i#j#k = a_p#j#i b_p#k
    //  ------------------       [ijk_pji_pk]
    //

    iterator_t ij = in.end();
    size_t sja_min = 0;
    for(iterator_t i = in.begin(); i != in.end(); i++) {
        if(i->stepa(0) > 0 && i->stepa(1) == 0 && i->stepb(0) > 0) {
            if(i->stepa(0) % z.m_ni ||
                z.m_spa % (i->weight() * i->stepa(0)))
                continue;
            if(i->stepb(0) % z.m_nj ||
                z.m_sic % (i->weight() * i->stepb(0)))
                continue;
            if(sja_min == 0 || sja_min > i->stepa(0)) {
                ij = i; sja_min = i->stepa(0);
            }
        }
    }
    if(ij == in.end()) return 0;

    kern_mul_ijk_pji_pk zz;
    zz.m_d = z.m_d;
    zz.m_ni = z.m_ni;
    zz.m_nj = ij->weight();
    zz.m_nk = z.m_nj;
    zz.m_np = z.m_np;
    zz.m_spa = z.m_spa;
    zz.m_sja = ij->stepa(0);
    zz.m_spb = z.m_spb;
    zz.m_sic = z.m_sic;
    zz.m_sjc = ij->stepb(0);
    in.splice(out.begin(), out, ij);

    kernel_base<2, 1> *kern = 0;

    if(kern = kern_mul_ijk_pjqi_qpk::match(zz, in, out)) return kern;
    if(kern = kern_mul_ijk_pqji_pqk::match(zz, in, out)) return kern;
    if(kern = kern_mul_ijk_pqji_qpk::match(zz, in, out)) return kern;

    return new kern_mul_ijk_pji_pk(zz);
}


} // namespace libtensor
