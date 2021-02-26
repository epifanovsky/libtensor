#ifndef LIBTENSOR_TO_SET_DIAG_IMPL_H
#define LIBTENSOR_TO_SET_DIAG_IMPL_H

#include <libtensor/kernels/kern_add1.h>
#include <libtensor/kernels/kern_copy.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../to_set_diag.h"

namespace libtensor {


template<size_t N, typename T>
const char *to_set_diag<N, T>::k_clazz = "to_set_diag<N, T>";


template<size_t N, typename T>
void to_set_diag<N, T>::perform(bool zero, dense_tensor_wr_i<N, T> &t) {

    static const char method[] = "perform(bool, dense_tensor_wr_i<N, T>&)";

    if (! zero && m_v == 0) return;

    const dimensions<N> &dims = t.get_dims();

    sequence<N, size_t> map(N);
    for (size_t i = 0; i < N; i++) {
        if (map[i] != N) continue;

        map[i] = i;
        if (m_msk[i] == 0) continue;

        for (size_t j = i + 1; j < N; j++) {
            if (m_msk[i] != m_msk[j]) continue;
            if (dims[i] != dims[j]) {
                throw bad_dimensions(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "t");
            }
            map[j] = i;
        }
    }

    dense_tensor_wr_ctrl<N, T> ctrl(t);
    ctrl.req_prefetch();

    std::list< loop_list_node<1, 1> > loop_in, loop_out;
    typename std::list< loop_list_node<1, 1> >::iterator inode;

    for (size_t i = 0; i < N; i++) {

        size_t inc = 0, len = 0;
        if (m_msk[i] == 0) {
            len = 1;
            while (i < N && m_msk[i] == 0) { len *= dims.get_dim(i); i++; }
            i--;
            inc = dims.get_increment(i);
        }
        else {
            if (map[i] < i) continue;

            for (size_t j = i; j < N; j++) {
                if (m_msk[j] != m_msk[i]) continue;
                inc += dims.get_increment(j);
            }
            len = dims.get_dim(i);
        }

        inode = loop_in.insert(loop_in.end(), loop_list_node<1, 1>(len));
        inode->stepa(0) = 0;
        inode->stepb(0) = inc;
    }

    T *ptr = ctrl.req_dataptr();
    loop_registers_x<1, 1, T> regs;
    regs.m_ptra[0] = &m_v;
    regs.m_ptrb[0] = ptr;
    regs.m_ptra_end[0] = &m_v + 1;
    regs.m_ptrb_end[0] = ptr + dims.get_size();
    {
        std::unique_ptr< kernel_base<linalg, 1, 1, T> > kern(nullptr);
        if (zero) {
            kern = std::unique_ptr< kernel_base<linalg, 1, 1, T> >(
                    kern_copy<linalg, T>::match(1.0, loop_in, loop_out));
        }
        else {
            kern = std::unique_ptr< kernel_base<linalg, 1, 1, T> >(
                    kern_add1<linalg, T>::match(1.0, loop_in, loop_out));
        }
        loop_list_runner_x<linalg, 1, 1, T>(loop_in).run(0, regs, *kern);
    }

    ctrl.ret_dataptr(ptr); ptr = 0;
}


} // namespace libtensor

#endif // LIBTENSOR_TO_SET_DIAG_IMPL_H
