#ifndef LIBTENSOR_TOD_EXTRACT_IMPL_H
#define LIBTENSOR_TOD_EXTRACT_IMPL_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/tod/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_extract.h"

namespace libtensor {


template<size_t N, size_t M>
const char *tod_extract<N, M>::k_clazz = "tod_extract<N, M>";


template<size_t N, size_t M>
const char *tod_extract<N, M>::op_dcopy::k_clazz =
    "tod_extract<N, M>::op_dcopy";

template<size_t N, size_t M>
const char *tod_extract<N, M>::op_daxpy::k_clazz =
    "tod_extract<N, M>::op_daxpy";


template<size_t N, size_t M>
tod_extract<N, M>::tod_extract(
        dense_tensor_rd_i<NA, double> &t, const mask<NA> &m,
        const index<NA> &idx, const tensor_transf_type &tr) :

    m_t(t), m_mask(m), m_perm(tr.get_perm()),
    m_c(tr.get_scalar_tr().get_coeff()),
    m_dims(mk_dims(t.get_dims(), m_mask)), m_idx(idx) {

    m_dims.permute(m_perm);
}


template<size_t N, size_t M>
tod_extract<N, M>::tod_extract(
        dense_tensor_rd_i<NA, double> &t, const mask<NA> &m,
        const index<NA> &idx, double c) :

    m_t(t), m_mask(m), m_c(c), m_dims(mk_dims(t.get_dims(), m)), m_idx(idx) {

}


template<size_t N, size_t M>
tod_extract<N, M>::tod_extract(
        dense_tensor_rd_i<NA, double> &t, const mask<NA> &m,
        const index<NA> &idx, const permutation<NB> &p, double c) :

    m_t(t), m_mask(m), m_perm(p), m_c(c),
    m_dims(mk_dims(t.get_dims(), m)), m_idx(idx) {

    m_dims.permute(p);
}


template<size_t N, size_t M>
void tod_extract<N, M>::perform(bool zero, dense_tensor_wr_i<NB, double> &tb) {

    static const char *method =
            "perform(bool, dense_tensor_wr_i<N - M, double>&)";

    if(!tb.get_dims().equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }

    tod_extract<N, M>::start_timer();

    dense_tensor_rd_ctrl<NA, double> ca(m_t);
    dense_tensor_wr_ctrl<NB, double> cb(tb);
    const double *pa = ca.req_const_dataptr();
    double *pb = cb.req_dataptr();

    loop_list_t lst;
    build_list(lst, tb, zero);

    registers regs;

    //set the pointer to the right position for reading
    //
    const dimensions<NA> &dimsa = m_t.get_dims();
    size_t pa_offset = 0;
    for(size_t pos1 = 0; pos1 < NA; pos1++) {
        if(m_idx[pos1] != 0) {
            pa_offset += m_idx[pos1] * dimsa.get_increment(pos1);
        }
    }

    regs.m_ptra = pa + pa_offset;
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

    tod_extract<N, M>::stop_timer();
}


template<size_t N, size_t M>
dimensions<N - M> tod_extract<N, M>::mk_dims(const dimensions<NA> &dims,
    const mask<NA> &msk) {

    static const char *method = "mk_dims(const dimensions<N> &, "
        "const mask<N>&)";

    //  Compute output dimensions
    //
    index<NB> i1, i2;

    size_t m = 0, j = 0;
    bool bad_dims = false;
    for(size_t i = 0; i < N; i++) {
        if(msk[i]){
            i2[j++] = dims[i] - 1;
        }else{
            m++;
        }
    }
    if(m != M) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "m");
    }
    if(bad_dims) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }
    return dimensions<NB>(index_range<NB>(i1, i2));
}


template<size_t N, size_t M>
void tod_extract<N, M>::build_list(loop_list_t &list,
        dense_tensor_wr_i<NB, double> &tb, bool zero) {

    static const char *method = "build_list(loop_list_t&, "
            "dense_tensor_wr_i<N - M, double>&, bool zero)";

    const dimensions<NA> &dimsa = m_t.get_dims();
    const dimensions<NB> &dimsb = tb.get_dims();

    //  Mapping of unpermuted indexes in b to permuted ones
    //
    sequence<NB, size_t> ib(0);
    for(size_t i = 0; i < NB; i++) ib[i] = i;
    m_perm.apply(ib);

    //  Loop over the indexes and build the list
    //
    try { // bad_alloc

    typename loop_list_t::iterator poscore = list.end();
    size_t iboffs = 0;
    for(size_t pos2 = 0; pos2 < N; pos2++) {

        size_t inca = 0, incb = 0, len = 0;
        if(m_mask[pos2]) {
            //situation if the index is not fixed
            len = 1;
            size_t ibpos = ib[pos2 - iboffs];
            while(pos2 < NA && m_mask[pos2] &&
                    ibpos == ib[pos2 - iboffs]) {
                len *= dimsa.get_dim(pos2);
                pos2++;
                ibpos++;
            }
            pos2--; ibpos--;
            inca = dimsa.get_increment(pos2);
            incb = dimsb.get_increment(ibpos);

        } else {
            // if the index is constant do nothing
            iboffs++;
            continue;
        }

        typename loop_list_t::iterator it = list.insert(
                list.end(), loop_list_node(len, inca, incb));

        //  Make the loop with incb the last
        //
        if(incb == 1 && poscore == list.end()) {
            if (zero)
                it->m_op = new op_dcopy(len, inca, incb, m_c);
            else
                it->m_op = new op_daxpy(len, inca, incb, m_c);

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
void tod_extract<N, M>::clean_list(loop_list_t& lst) {

    for(typename loop_list_t::iterator i = lst.begin();
        i != lst.end(); i++) {

        delete i->m_op; i->m_op = 0;
    }
}


template<size_t N, size_t M>
void tod_extract<N, M>::op_loop::exec(processor_t &proc, registers &regs)
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
void tod_extract<N, M>::op_dcopy::exec(processor_t &proc, registers &regs)
    throw(exception) {

    if(m_len == 0) return;

    op_dcopy::start_timer();
    linalg::copy_i_i(0, m_len, regs.m_ptra, m_inca, regs.m_ptrb, m_incb);
    if(m_c != 1.0) linalg::mul1_i_x(0, m_len, m_c, regs.m_ptrb, m_incb);
    op_dcopy::stop_timer();
}


template<size_t N, size_t M>
void tod_extract<N, M>::op_daxpy::exec(processor_t &proc, registers &regs)
    throw(exception) {

    if(m_len == 0) return;

    op_daxpy::start_timer();
    linalg::mul2_i_i_x(0, m_len, regs.m_ptra, m_inca, m_c, regs.m_ptrb, m_incb);
    op_daxpy::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_EXTRACT_IMPL_H
