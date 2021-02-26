#ifndef LIBTENSOR_LOOP_LIST_RUNNER_H
#define LIBTENSOR_LOOP_LIST_RUNNER_H

#include "kernel_base.h"

namespace libtensor {


/** \brief Runs a series of nested loops

    \ingroup libtensor_kernels
 **/
template<typename LA, size_t N, size_t M, typename T>
class loop_list_runner_x {
public:
    typedef typename kernel_base<LA, N, M, T>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, N, M, T>::list_t list_t;
    typedef typename kernel_base<LA, N, M, T>::iterator_t iterator_t;
    typedef typename kernel_base<LA, N, M, T>::const_iterator_t const_iterator_t;

private:
    const list_t &m_list;

public:
    loop_list_runner_x(const list_t &list) : m_list(list) { }

    void run(
        device_context_ref ctx,
        const loop_registers_x<N, M, T> &r,
        kernel_base<LA, N, M, T> &k);

private:
    void run_loop(
        device_context_ref ctx,
        const const_iterator_t &i,
        const loop_registers_x<N, M, T> &r,
        kernel_base<LA, N, M, T> &k);

};


template<typename LA, size_t N, size_t M, typename T>
void loop_list_runner_x<LA, N, M, T>::run(
    device_context_ref ctx,
    const loop_registers_x<N, M, T> &r,
    kernel_base<LA, N, M, T> &k) {

    const_iterator_t i = m_list.begin();
    run_loop(ctx, i, r, k);
}


template<typename LA, size_t N, size_t M, typename T>
void loop_list_runner_x<LA, N, M, T>::run_loop(
    device_context_ref ctx,
    const const_iterator_t &i,
    const loop_registers_x<N, M, T> &r,
    kernel_base<LA, N, M, T> &k) {

    if(i == m_list.end()) {
        k.run(ctx, r);
        return;
    }

    loop_registers_x<N, M, T> r1(r);
    for(size_t j = 0; j < i->weight(); j++) {
        const_iterator_t ii = i; ii++;
        run_loop(ctx, ii, r1, k);
        for(size_t k = 0; k < N; k++) {
            r1.m_ptra[k] += i->stepa(k);
        }
        for(size_t k = 0; k < M; k++) {
            r1.m_ptrb[k] += i->stepb(k);
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_LOOP_LIST_RUNNER_H
