#ifndef LIBTENSOR_TOD_DIAG_IMPL_H
#define LIBTENSOR_TOD_DIAG_IMPL_H

#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/tod/bad_dimensions.h>
#include "../tod_diag.h"

namespace libtensor {


template<size_t N, size_t M>
const char *tod_diag<N, M>::k_clazz = "tod_diag<N, M>";


template<size_t N, size_t M>
const char *tod_diag<N, M>::op_dcopy::k_clazz = "tod_diag<N, M>::op_dcopy";


template<size_t N, size_t M>
const char *tod_diag<N, M>::op_daxpy::k_clazz = "tod_diag<N, M>::op_daxpy";


template<size_t N, size_t M>
tod_diag<N, M>::tod_diag(dense_tensor_rd_i<N, double> &t, const mask<N> &m,
    const tensor_transf<k_orderb, double> &tr) :

    m_t(t), m_mask(m), m_tr(tr), m_dims(mk_dims(t.get_dims(), m_mask)) {

    m_dims.permute(tr.get_perm());
}


template<size_t N, size_t M>
void tod_diag<N, M>::perform(bool zero, dense_tensor_wr_i<k_orderb, double> &tb) {

    static const char *method =
            "perform(bool, dense_tensor_wr_i<N - M + 1, double> &)";

#ifdef LIBTENSOR_DEBUG
    if(!tb.get_dims().equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }
#endif

    if(zero) tod_set<k_orderb>().perform(tb);
    if(m_tr.get_scalar_tr().get_coeff() == 0.0) return;

    do_perform<op_daxpy>(tb);
}


template<size_t N, size_t M>
dimensions<N - M + 1> tod_diag<N, M>::mk_dims(const dimensions<N> &dims,
    const mask<N> &msk) {

    static const char *method =
        "mk_dims(const dimensions<N> &, const mask<N>&)";

    //  Compute output dimensions
    //
    index<k_orderb> i1, i2;

    size_t m = 0, j = 0;
    size_t d = 0;
    bool bad_dims = false;
    for(size_t i = 0; i < N; i++) {
        if(msk[i]) {
            m++;
            if(d == 0) {
                d = dims[i];
                i2[j++] = d - 1;
            } else {
                bad_dims = bad_dims || d != dims[i];
            }
        } else {
            if(!bad_dims) i2[j++] = dims[i] - 1;
        }
    }
    if(m != M) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "m");
    }
    if(bad_dims) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }
    return dimensions<k_orderb>(index_range<k_orderb>(i1, i2));
}


template<size_t N, size_t M> template<typename CoreOp>
void tod_diag<N, M>::do_perform(dense_tensor_wr_i<k_orderb, double> &tb) {

    static const char *method =
        "do_perform(dense_tensor_wr_i<N - M + 1, double>&)";

    tod_diag<N, M>::start_timer();

    dense_tensor_rd_ctrl<k_ordera, double> ca(m_t);
    dense_tensor_wr_ctrl<k_orderb, double> cb(tb);
    const double *pa = ca.req_const_dataptr();
    double *pb = cb.req_dataptr();

    loop_list_t lst;
    build_list<CoreOp>(lst, tb);

    registers regs;
    regs.m_ptra = pa;
    regs.m_ptrb = pb;

    try {
        processor_t proc(lst, regs);
        proc.process_next();
    } catch(...) {
        clean_list(lst);
        throw;
    }

    clean_list(lst);

    cb.ret_dataptr(pb); pb = 0;
    ca.ret_const_dataptr(pa); pa = 0;

    tod_diag<N, M>::stop_timer();
}


template<size_t N, size_t M> template<typename CoreOp>
void tod_diag<N, M>::build_list(loop_list_t &list,
    dense_tensor_wr_i<k_orderb, double> &tb) {

    static const char *method = "build_list(loop_list_t&, "
        "dense_tensor_wr_i<N - M + 1, double>&)";

    const dimensions<k_ordera> &dimsa = m_t.get_dims();
    const dimensions<k_orderb> &dimsb = tb.get_dims();

    //  Mapping of unpermuted indexes in b to permuted ones
    //
    sequence<k_orderb, size_t> ib(0);
    for(size_t i = 0; i < k_orderb; i++) ib[i] = i;
    permutation<k_orderb> pinv(m_tr.get_perm(), true);
    pinv.apply(ib);

    //  Loop over the indexes and build the list
    //
    try { // bad_alloc

    typename loop_list_t::iterator poscore = list.end();
    bool diag_done = false;
    size_t iboffs = 0;
    for(size_t pos = 0; pos < N; pos++) {

            size_t inca = 0, incb = 0, len = 0;

            if(m_mask[pos]) {

                if(diag_done) {
                    iboffs++;
                    continue;
                }

                //  Compute the stride on the diagonal
                //
                for(size_t j = pos; j < N; j++)
                    if(m_mask[j]) inca += dimsa.get_increment(j);
                incb = dimsb.get_increment(ib[pos]);
                len = dimsa.get_dim(pos);
                diag_done = true;

            } else {

                //  Compute the stride off the diagonal
                //  concatenating indexes if possible
                //
                len = 1;
                size_t ibpos = ib[pos - iboffs];
                while(pos < N && !m_mask[pos] && ibpos == ib[pos - iboffs]) {

                    len *= dimsa.get_dim(pos);
                    pos++;
                    ibpos++;
                }
                pos--; ibpos--;
                inca = dimsa.get_increment(pos);
                incb = dimsb.get_increment(ibpos);
            }


            typename loop_list_t::iterator it = list.insert(
                list.end(), loop_list_node(len, inca, incb));

            //  Make the loop with incb the last
            //
            if(incb == 1 && poscore == list.end()) {
                it->m_op = new CoreOp(len, inca, incb,
                        m_tr.get_scalar_tr().get_coeff());
                poscore = it;
            } else {
                it->m_op = new op_loop(len, inca, incb);
            }
    }

    list.splice(list.end(), list, poscore);

    } catch(std::bad_alloc &e) {

        clean_list(list);
        throw out_of_memory(
            g_ns, k_clazz, method, __FILE__, __LINE__, e.what());
    }
}


template<size_t N, size_t M>
void tod_diag<N, M>::clean_list(loop_list_t& lst) {

    for(typename loop_list_t::iterator i = lst.begin();
        i != lst.end(); i++) {

        delete i->m_op; i->m_op = 0;
    }
}


template<size_t N, size_t M>
void tod_diag<N, M>::op_loop::exec(processor_t &proc, registers &regs)
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


template<size_t N, size_t M>
void tod_diag<N, M>::op_dcopy::exec(processor_t &proc, registers &regs)
    throw(exception) {

    if(m_len == 0) return;

    op_dcopy::start_timer();
    linalg::copy_i_i(0, m_len, regs.m_ptra, m_inca, regs.m_ptrb, m_incb);
    if(m_c != 1.0) linalg::mul1_i_x(0, m_len, m_c, regs.m_ptrb, m_incb);
    op_dcopy::stop_timer();
}


template<size_t N, size_t M>
void tod_diag<N, M>::op_daxpy::exec(processor_t &proc, registers &regs)
    throw(exception) {

    if(m_len == 0) return;

    op_daxpy::start_timer();
    linalg::mul2_i_i_x(0, m_len, regs.m_ptra, m_inca, m_c, regs.m_ptrb, m_incb);
    op_daxpy::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_DIAG_IMPL_H
