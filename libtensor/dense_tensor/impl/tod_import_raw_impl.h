#ifndef LIBTENSOR_TOD_IMPORT_RAW_IMPL_H
#define LIBTENSOR_TOD_IMPORT_RAW_IMPL_H

#include <libtensor/linalg/linalg.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "../tod_import_raw.h"

namespace libtensor {


template<size_t N>
const char *tod_import_raw<N>::k_clazz = "tod_import_raw<N>";


template<size_t N>
void tod_import_raw<N>::perform(dense_tensor_i<N, double> &t) {

    static const char *method = "perform(tensor_i<N, double>&)";

    dimensions<N> dimsb(m_ir);
    if(!t.get_dims().equals(dimsb)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t.");
    }

    dense_tensor_ctrl<N, double> tctrl(t);

    loop_list_t lst;
    for(size_t i = 0; i < N; i++) {
        size_t inca = m_dims.get_increment(i);
        size_t incb = dimsb.get_increment(i);
        loop_list_node node(dimsb[i], inca, incb);
        if(i == N - 1) {
            node.m_op = new op_dcopy(dimsb[i]);
        } else {
            node.m_op = new op_loop(dimsb[i], inca, incb);
        }
        lst.push_back(node);
    }

    const double *ptra = m_ptr +
        abs_index<N>(m_ir.get_begin(), m_dims).get_abs_index();
    double *ptrb = tctrl.req_dataptr();

    registers regs;
    regs.m_ptra = ptra;
    regs.m_ptrb = ptrb;
    processor_t proc(lst, regs);
    proc.process_next();

    tctrl.ret_dataptr(ptrb);

    for(typename loop_list_t::iterator i = lst.begin();
        i != lst.end(); i++) {

        delete i->m_op;
        i->m_op = NULL;
    }
}


template<size_t N>
void tod_import_raw<N>::op_loop::exec(processor_t &proc, registers &regs)
    throw(exception) {

    const double *ptra = regs.m_ptra;
    double *ptrb = regs.m_ptrb;

    for(size_t i=0; i<m_len; i++) {
        regs.m_ptra = ptra;
        regs.m_ptrb = ptrb;
        proc.process_next();
        ptra += m_inca;
        ptrb += m_incb;
    }
}


template<size_t N>
void tod_import_raw<N>::op_dcopy::exec(processor_t &proc, registers &regs)
    throw(exception) {

    linalg::copy_i_i(0, m_len, regs.m_ptra, 1, regs.m_ptrb, 1);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_IMPORT_RAW_IMPL_H
