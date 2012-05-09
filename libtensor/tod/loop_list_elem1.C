#include "../defs.h"
#include "../exception.h"
#include "../linalg.h"
#include "loop_list_elem1.h"
#include "overflow.h"

namespace libtensor {


const char *loop_list_elem1::k_clazz = "loop_list_elem1";

void loop_list_elem1::run_loop(list_t &loop, registers &r, double c,
        bool doadd, bool recip) {

    iterator_t op;
    for (iterator_t i = loop.begin(); i != loop.end(); i++) {

        i->fn() = 0;
        if (i->stepb(0) == 1) op = i;
    }

    if (doadd && recip)
        op->fn() = &loop_list_elem1::fn_div_add;
    else if (recip)
        op->fn() = &loop_list_elem1::fn_div_put;
    else if (doadd)
        op->fn() = &loop_list_elem1::fn_mult_add;
    else
        op->fn() = &loop_list_elem1::fn_mult_put;

    m_op.m_k = c;
    m_op.m_n = op->weight();
    m_op.m_stepb = op->stepa(0);

    iterator_t begin = loop.begin(), end = loop.end();
    if(begin != end) {
        loop_list_base<1, 1, loop_list_elem1>::exec(*this, begin, end, r);
    }
}


void loop_list_elem1::fn_mult_add(registers &r) const {

    static const char *method = "fn_mult_add(registers&)";

    if(m_op.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
    if(r.m_ptra[0] + (m_op.m_n - 1) * m_op.m_stepb >=
        r.m_ptra_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "source1");
    }
    if(r.m_ptrb[0] + (m_op.m_n - 1) >= r.m_ptrb_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "source2");
    }
#endif // LIBTENSOR_DEBUG

    register size_t ib = 0;
    double *pa = r.m_ptrb[0];
    const double *pb = r.m_ptra[0];
    for (register size_t ia = 0; ia < m_op.m_n; ia++) {
        pa[ia] += m_op.m_k * pa[ia] * pb[ib];
        ib += m_op.m_stepb;
    }
}

void loop_list_elem1::fn_mult_put(registers &r) const {

    static const char *method = "fn_mult_put(registers&)";

    if(m_op.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
    if(r.m_ptra[0] + (m_op.m_n - 1) * m_op.m_stepb >=
        r.m_ptra_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "source1");
    }
    if(r.m_ptrb[0] + (m_op.m_n - 1) >= r.m_ptrb_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "source2");
    }
#endif // LIBTENSOR_DEBUG

    register size_t ib = 0;
    double *pa = r.m_ptrb[0];
    const double *pb = r.m_ptra[0];
    for (register size_t ia = 0; ia < m_op.m_n; ia++) {
        pa[ia] *= pb[ib] * m_op.m_k;
        ib += m_op.m_stepb;
    }
}

void loop_list_elem1::fn_div_add(registers &r) const {

    static const char *method = "fn_div_add(registers&)";

    if(m_op.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
    if(r.m_ptra[0] + (m_op.m_n - 1) * m_op.m_stepb >=
        r.m_ptra_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "source1");
    }
    if(r.m_ptrb[0] + (m_op.m_n - 1) >= r.m_ptrb_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "source2");
    }
#endif // LIBTENSOR_DEBUG

    register size_t ib = 0;
    double *pa = r.m_ptrb[0];
    const double *pb = r.m_ptra[0];
    for (register size_t ia = 0; ia < m_op.m_n; ia++) {
        pa[ia] += m_op.m_k * pa[ia] / pb[ib];
        ib += m_op.m_stepb;
    }
}

void loop_list_elem1::fn_div_put(registers &r) const {

    static const char *method = "fn_div_put(registers&)";

    if(m_op.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
    if(r.m_ptra[0] + (m_op.m_n - 1) * m_op.m_stepb >=
        r.m_ptra_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "source1");
    }
    if(r.m_ptrb[0] + (m_op.m_n - 1) >= r.m_ptrb_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "source2");
    }
#endif // LIBTENSOR_DEBUG

    register size_t ib = 0;
    double *pa = r.m_ptrb[0];
    const double *pb = r.m_ptra[0];
    for (register size_t ia = 0; ia < m_op.m_n; ia++) {
        pa[ia] *= m_op.m_k / pb[ib];
        ib += m_op.m_stepb;
    }
}



} // namespace libtensor
