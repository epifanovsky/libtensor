#ifndef LIBTENSOR_TOD_IMPORT_RAW_IMPL_H
#define LIBTENSOR_TOD_IMPORT_RAW_IMPL_H

#include <libtensor/kernels/kern_dcopy.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/core/abs_index.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_import_raw.h"

namespace libtensor {


template<size_t N>
const char *tod_import_raw<N>::k_clazz = "tod_import_raw<N>";


template<size_t N>
void tod_import_raw<N>::perform(dense_tensor_wr_i<N, double> &t) {

    static const char *method = "perform(tensor_i<N, double>&)";

    dimensions<N> dimsb(m_ir);
    if(!t.get_dims().equals(dimsb)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t.");
    }

    try {

        dense_tensor_wr_ctrl<N, double> cb(t);
        cb.req_prefetch();

        std::list< loop_list_node<1, 1> > loop_in, loop_out;
        typename std::list< loop_list_node<1, 1> >::iterator inode = loop_in.end();

        for (size_t i = 0; i < N; i++) {

            inode = loop_in.insert(loop_in.end(),
                    loop_list_node<1, 1>(dimsb[i]));
            inode->stepa(0) = m_dims.get_increment(i);
            inode->stepb(0) = dimsb.get_increment(i);
        }

        const double *pa = m_ptr +
                abs_index<N>::get_abs_index(m_ir.get_begin(), m_dims);
        double *pb = cb.req_dataptr();

        loop_registers<1, 1> regs;
        regs.m_ptra[0] = pa;
        regs.m_ptrb[0] = pb;
        regs.m_ptra_end[0] = pa + m_dims.get_size();
        regs.m_ptrb_end[0] = pb + dimsb.get_size();

        {
            std::auto_ptr< kernel_base<linalg, 1, 1, double> > kern(
                    kern_dcopy<linalg>::match(1.0, loop_in, loop_out));

            loop_list_runner<linalg, 1, 1>(loop_in).run(0, regs, *kern);
        }

        cb.ret_dataptr(pb);

    } catch (...) {
        throw;
    }
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_IMPORT_RAW_IMPL_H
