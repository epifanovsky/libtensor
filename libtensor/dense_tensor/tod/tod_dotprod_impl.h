#ifndef LIBTENSOR_TOD_DOTPROD_IMPL_H
#define LIBTENSOR_TOD_DOTPROD_IMPL_H

#include <libtensor/mp/auto_cpu_lock.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/tod/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_dotprod.h"

namespace libtensor {


template<size_t N>
const char *tod_dotprod<N>::k_clazz = "tod_dotprod<N>";


template<size_t N>
const char *tod_dotprod<N>::op_ddot::k_clazz = "tod_dotprod<N>::op_ddot";


template<size_t N>
tod_dotprod<N>::tod_dotprod(dense_tensor_rd_i<N, double> &ta,
    dense_tensor_rd_i<N, double> &tb) :

    m_ta(ta), m_tb(tb) {

    static const char *method = "tod_dotprod(dense_tensor_rd_i<N, double>&, "
        "dense_tensor_rd_i<N, double>&)";

    if(!verify_dims()) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "ta != tb");
    }
}


template<size_t N>
tod_dotprod<N>::tod_dotprod(dense_tensor_rd_i<N, double> &ta,
    const permutation<N> &perma, dense_tensor_rd_i<N, double> &tb,
    const permutation<N> &permb) :

    m_ta(ta), m_perma(perma), m_tb(tb), m_permb(permb) {

    static const char *method = "tod_dotprod(dense_tensor_rd_i<N, double>&, "
        "const permutation<N>&, dense_tensor_rd_i<N, double>&, "
        "const permutation<N>&)";

    if(!verify_dims()) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "ta != tb");
    }
}


template<size_t N>
void tod_dotprod<N>::prefetch() {

    dense_tensor_rd_ctrl<N, double>(m_ta).req_prefetch();
    dense_tensor_rd_ctrl<N, double>(m_tb).req_prefetch();
}


template<size_t N>
double tod_dotprod<N>::calculate(cpu_pool &cpus) {

    double result = 0.0;

    tod_dotprod<N>::start_timer();

    try {

        permutation<N> perma(m_perma);
        perma.invert();
        permutation<N> permb(m_permb);
        permb.permute(perma);

        const dimensions<N> &dima(m_ta.get_dims());
        const dimensions<N> &dimb(m_tb.get_dims());

        build_list(m_list, dimb, permb, dima);

        dense_tensor_rd_ctrl<N, double> ca(m_ta), cb(m_tb);
        const double *pb = ca.req_const_dataptr();
        const double *pa = cb.req_const_dataptr();

        {
            auto_cpu_lock cpu(cpus);

            try {
                registers regs;
                regs.m_ptra = pa; regs.m_ptrb = pb; regs.m_ptrc = &result;
                processor_t proc(m_list, regs);
                proc.process_next();
            } catch(exception &e) {
                clean_list();
                throw;
            }

            clean_list();
        }

        ca.ret_const_dataptr(pb);
        cb.ret_const_dataptr(pa);

    } catch(...) {
        tod_dotprod<N>::stop_timer();
        throw;
    }

    tod_dotprod<N>::stop_timer();

    return result;
}


template<size_t N>
bool tod_dotprod<N>::verify_dims() {

    dimensions<N> dimsa(m_ta.get_dims());
    dimensions<N> dimsb(m_tb.get_dims());
    dimsa.permute(m_perma);
    dimsb.permute(m_permb);
    return dimsa.equals(dimsb);
}


template<size_t N>
void tod_dotprod<N>::build_list(loop_list_t &list, const dimensions<N> &da,
    const permutation<N> &pa, const dimensions<N> &db)
    throw(out_of_memory) {

    sequence<N, size_t> ia(0);
    for(size_t i = 0; i < N; i++) ia[i] = i;
    pa.apply(ia);

    // loop over all indices and build the list
    size_t pos = 0;
    try {
        typename loop_list_t::iterator posa = list.end(),
            posb = list.end();
        while(pos < N) {
            size_t len = 1;
            size_t iapos = ia[pos];
            while(pos < N) {
                if(ia[pos] != iapos) break;
                len *= da.get_dim(iapos);
                pos++; iapos++;
            }

            size_t inca = da.get_increment(iapos - 1);
            size_t incb = db.get_increment(pos - 1);

            typename loop_list_t::iterator it = list.insert(
                list.end(),loop_list_node(len,inca,incb));

            // we know that in the last loop incb=1 !!!
            if(inca == 1) {
                if(inca == incb) {
                    it->m_op = new op_ddot(len, inca, incb);
                } else {
                    posa = it;
                }
            } else {
                if(incb == 1) posb = it;
                else it->m_op = new op_loop(len, inca, incb);
            }
        }

        if(posa != posb) {
            if(posa->m_weight > posb->m_weight) {
                posa->m_op = new op_ddot(posa->m_weight,
                    posa->m_inca, posa->m_incb);
                posb->m_op = new op_loop(posb->m_weight,
                    posb->m_inca, posb->m_incb);
                list.splice(list.end(), list, posa);
            } else {
                posa->m_op = new op_loop(posa->m_weight,
                    posa->m_inca, posa->m_incb);
                posb->m_op = new op_ddot(posb->m_weight,
                    posb->m_inca, posb->m_incb);
                list.splice(posb, list, posa);
            }
        }

    } catch(std::bad_alloc &e) {

        clean_list();
        throw out_of_memory("libtensor",k_clazz,
            "build_list(loop_list_t&,const dimensions<N>&,"
            "const permutation<N>&,const dimensions<N>&)",
            __FILE__,__LINE__,e.what());
    }
}


template<size_t N>
void tod_dotprod<N>::clean_list() {

    for(typename loop_list_t::iterator i = m_list.begin();
        i != m_list.end(); i++) {

        delete i->m_op;
        i->m_op = NULL;
    }
}


template<size_t N>
void tod_dotprod<N>::op_loop::exec(processor_t &proc, registers &regs)
    throw(exception) {

    const double *ptra = regs.m_ptra, *ptrb = regs.m_ptrb;

    for(size_t i = 0; i < m_len; i++) {
        regs.m_ptra = ptra;
        regs.m_ptrb = ptrb;
        proc.process_next();
        ptra += m_inca;
        ptrb += m_incb;
    }
}


template<size_t N>
void tod_dotprod<N>::op_ddot::exec(processor_t &proc, registers &regs)
    throw(exception) {

    tod_dotprod<N>::op_ddot::start_timer();

    *(regs.m_ptrc) += linalg::x_p_p(m_n, regs.m_ptra, m_inca,
        regs.m_ptrb, m_incb);

    tod_dotprod<N>::op_ddot::stop_timer();

}


} // namespace libtensor

#endif // LIBTENSOR_TOD_DOTPROD_IMPL_H
