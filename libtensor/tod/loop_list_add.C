#include "../defs.h"
#include "../exception.h"
#include "../linalg/linalg.h"
#include "loop_list_add.h"
#include "overflow.h"

namespace libtensor {


const char *loop_list_add::k_clazz = "loop_list_add";


void loop_list_add::run_loop(list_t &loop, registers &r, double c) {

    match_l1(loop, c);

    iterator_t begin = loop.begin(), end = loop.end();
    if(begin != end) {
        loop_list_base<1, 1, loop_list_add>::exec(
            *this, begin, end, r);
    }
}


void loop_list_add::match_l1(list_t &loop, double c) {

    //  1. Find:
    //  --------
    //  w   a  b
    //  w1  1  1  -->  b_i += a_i
    //  --------       sz(i) = w1
    //                 [daxpy]
    //
    //  2. Minimize k1a:
    //
    //  ----------
    //  w   a  b
    //  w1  1  k1a  -->  b_i# += a_i
    //  ----------       sz(i) = w1, sz(#) = k1a
    //                   [daxpy]
    //
    //  3. Minimize k1b:
    //
    //  ----------
    //  w   a    b
    //  w1  k1b  1  -->  b_i += a_i#
    //  ----------       sz(i) = w1, sz(#) = k1b
    //                   [daxpy]
    //
    //  4. Fallback [daxpy]
    //
    iterator_t i1 = loop.end(), i2 = loop.end(), i3 = loop.end();
    size_t k1a_min = 0, k1b_min = 0;
    for(iterator_t i = loop.begin(); i != loop.end(); i++) {
        i->fn() = 0;
        if(i->stepa(0) == 1) {
            if(k1a_min == 0 || k1a_min > i->stepb(0)) {
                i2 = i; k1a_min = i->stepb(0);
            }
            if(i->stepb(0) == 1) {
                i1 = i;
            }
        } else {
            if(i->stepb(0) == 1) {
                if(k1b_min == 0 || k1b_min > i->stepa(0)) {
                    i3 = i; k1b_min = i->stepa(0);
                }
            }
        }
    }

    if(i1 != loop.end()) {
        i1->fn() = &loop_list_add::fn_daxpy;
        m_daxpy.m_k = c;
        m_daxpy.m_n = i1->weight();
        m_daxpy.m_stepa = 1;
        m_daxpy.m_stepb = 1;
        loop.splice(loop.end(), loop, i1);
        return;
    }

    if(i2 != loop.end()) {
        i2->fn() = &loop_list_add::fn_daxpy;
        m_daxpy.m_k = c;
        m_daxpy.m_n = i2->weight();
        m_daxpy.m_stepa = 1;
        m_daxpy.m_stepb = i2->stepb(0);
        //~ match_l2_a(loop, c, i2->weight());
        loop.splice(loop.end(), loop, i2);
        return;
    }

    if(i3 != loop.end()) {
        i3->fn() = &loop_list_add::fn_daxpy;
        m_daxpy.m_k = c;
        m_daxpy.m_n = i3->weight();
        m_daxpy.m_stepa = i3->stepa(0);
        m_daxpy.m_stepb = 1;
        //~ match_l2_b(loop, c, i3->weight());
        loop.splice(loop.end(), loop, i3);
        return;
    }

    iterator_t i4 = loop.begin();
    i4->fn() = &loop_list_add::fn_daxpy;
    m_daxpy.m_k = c;
    m_daxpy.m_n = i4->weight();
    m_daxpy.m_stepa = i4->stepa(0);
    m_daxpy.m_stepb = i4->stepb(0);
    loop.splice(loop.end(), loop, i4);
}


void loop_list_add::match_l2_a(list_t &loop, double c, size_t w1) {

    //  Found pattern:
    //  ---------
    //  w   a  b
    //  w1  1  k1  -->  b_i# += a_i
    //  ---------       sz(i) = w1, sz(#) = k1
    //                  [daxpy]

    //  1. Find k2a = 1:
    //  --------------
    //  w   a       b
    //  w1  1       k1
    //  w2  k2a*w1  1   -->  b_ij += a_j$i
    //  --------------       sz(i) = w1, sz(j) = w2, sz($) = k2a
    //                       [daxpby_trp]
    iterator_t i1 = loop.end();
    for(iterator_t i = loop.begin(); i != loop.end(); i++) {
        if(i->stepb(0) != 1) continue;
        if(i->stepa(0) != w1) continue;
        i1 = i;
        break;
    }

    if(i1 != loop.end()) {
        i1->fn() = &loop_list_add::fn_daxpby_trp;
        m_daxpby_trp.m_k = c;
        m_daxpby_trp.m_ni = w1;
        m_daxpby_trp.m_nj = i1->weight();
        m_daxpby_trp.m_stepa = 1;
        m_daxpby_trp.m_stepb = 1;
        loop.splice(loop.end(), loop, i1);
    }
}


void loop_list_add::match_l2_b(list_t &loop, double c, size_t w1) {

}


void loop_list_add::fn_daxpy(registers &r) const {

    static const char *method = "fn_daxpy(registers&)";

    if(m_daxpy.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
    if(r.m_ptra[0] + (m_daxpy.m_n - 1) * m_daxpy.m_stepa >=
        r.m_ptra_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "source");
    }
    if(r.m_ptrb[0] + (m_daxpy.m_n - 1) * m_daxpy.m_stepb >=
        r.m_ptrb_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "destination");
    }
#endif // LIBTENSOR_DEBUG

    linalg::mul2_i_i_x(0, m_daxpy.m_n, r.m_ptra[0], m_daxpy.m_stepa, m_daxpy.m_k,
        r.m_ptrb[0], m_daxpy.m_stepb);
}


void loop_list_add::fn_daxpby_trp(registers &r) const {

    static const char *method = "fn_daxpby_trp(registers&)";

    if(m_daxpby_trp.m_ni == 0 || m_daxpby_trp.m_nj) return;

#ifdef LIBTENSOR_DEBUG
    if(r.m_ptra[0] + (m_daxpby_trp.m_ni - 1) * m_daxpby_trp.m_stepa >=
        r.m_ptra_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "source");
    }
    if(r.m_ptrb[0] + (m_daxpby_trp.m_nj - 1) * m_daxpby_trp.m_stepb >=
        r.m_ptrb_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "destination");
    }
#endif // LIBTENSOR_DEBUG

//  blas::daxpby_trp(r.m_ptra[0], r.m_ptrb[0], m_daxpby_trp.m_ni,
//      m_daxpby_trp.m_nj, m_daxpby_trp.m_stepa, m_daxpby_trp.m_stepb,
//      m_daxpby_trp.m_k, 1.0);
}


} // namespace libtensor
