#ifndef LIBTENSOR_LOOP_LIST_APPLY_IMPL_H
#define LIBTENSOR_LOOP_LIST_APPLY_IMPL_H

#include "../defs.h"
#include "overflow.h"

namespace libtensor {

template<typename Functor>
void apply_base<Functor>::set(size_t ni, functor_t &fn,
        const double *a, size_t sia, double b0, double b1,
        double *c, size_t sic) {

    for (size_t i = 0; i < ni; i++)
        c[i * sic] = b0 * fn(b1 * a[i * sia]);
}

template<typename Functor>
void apply_base<Functor>::add(size_t ni, functor_t &fn,
        const double *a, size_t sia, double b0, double b1,
        double *c, size_t sic) {

    for (size_t i = 0; i < ni; i++)
        c[i * sic] += b0 * fn(b1 * a[i * sia]);
}

template<typename Functor>
const char *loop_list_apply<Functor>::k_clazz = "loop_list_apply<Functor>";

template<typename Functor>
void loop_list_apply<Functor>::run_loop(list_t &loop, registers &r,
        functor_t &fn, double c0, double c1, bool do_add) {

    install_kernel(loop, fn, c0, c1, do_add);

    iterator_t begin = loop.begin(), end = loop.end();
    if(begin != end) {
        loop_list_base<1, 1, loop_list_apply>::exec(
            *this, begin, end, r);
    }
}

template<typename Functor>
void loop_list_apply<Functor>::install_kernel(list_t &loop,
        functor_t &fn, double c0, double c1, bool do_add) {

    //
    //  Try to find the kernel loop using priority:
    //
    //  step(A) = 1, step(B) = 1
    //  step(A) = 1, step(B) = any
    //  step(A) = any, step(B) = 1
    //  step(A) = any, step(B) = any
    //
    iterator_t ia = loop.end(), ib = loop.end(), iab = loop.end();
    for(iterator_t i = loop.begin(); i != loop.end(); i++) {
        i->fn() = 0;
        if(i->stepa(0) == 1) {
            ia = i;
            if(i->stepb(0) == 1) {
                ib = i;
                iab = i;
            }
        } else {
            if(i->stepb(0) == 1) {
                ib = i;
            }
        }
    }

    iterator_t ikernel;
    if(iab != loop.end()) {
        ikernel = iab;
    } else if(ia != loop.end()) {
        ikernel = ia;
    } else if(ib != loop.end()) {
        ikernel = ib;
    } else {
        ikernel = loop.begin();
    }

    if(ikernel != loop.end()) {
        if (do_add) ikernel->fn() = &loop_list_apply<functor_t>::fn_apply_add;
        else ikernel->fn() = &loop_list_apply<functor_t>::fn_apply_set;
        m_apply.m_c0 = c0;
        m_apply.m_c1 = c1;
        m_apply.m_n = ikernel->weight();
        m_apply.m_fn = &fn;
        m_apply.m_stepa = ikernel->stepa(0);
        m_apply.m_stepb = ikernel->stepb(0);
        loop.splice(loop.end(), loop, ikernel);
    }
}


template<typename Functor>
void loop_list_apply<Functor>::fn_apply_add(registers &r) const {

    static const char *method = "fn_apply_add(registers&)";

    if(m_apply.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
    if(r.m_ptra[0] + (m_apply.m_n - 1) * m_apply.m_stepa >= r.m_ptra_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "source");
    }
    if(r.m_ptrb[0] + (m_apply.m_n - 1) * m_apply.m_stepb >= r.m_ptrb_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "destination");
    }
#endif // LIBTENSOR_DEBUG

    apply_base<functor_t>::add(m_apply.m_n, *m_apply.m_fn,
            r.m_ptra[0], m_apply.m_stepa, m_apply.m_c0, m_apply.m_c1,
            r.m_ptrb[0], m_apply.m_stepb);
}


template<typename Functor>
void loop_list_apply<Functor>::fn_apply_set(registers &r) const {

    static const char *method = "fn_apply_set(registers&)";

    if(m_apply.m_n == 0) return;

#ifdef LIBTENSOR_DEBUG
    if(r.m_ptra[0] + (m_apply.m_n - 1) * m_apply.m_stepa >= r.m_ptra_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "source");
    }
    if(r.m_ptrb[0] + (m_apply.m_n - 1) * m_apply.m_stepb >= r.m_ptrb_end[0]) {
        throw overflow(g_ns, k_clazz, method, __FILE__, __LINE__,
            "destination");
    }
#endif // LIBTENSOR_DEBUG

    apply_base<functor_t>::set(m_apply.m_n, *m_apply.m_fn,
            r.m_ptra[0], m_apply.m_stepa, m_apply.m_c0, m_apply.m_c1,
            r.m_ptrb[0], m_apply.m_stepb);
}

} // namespace libtensor

#endif
