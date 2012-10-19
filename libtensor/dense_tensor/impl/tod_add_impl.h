#ifndef LIBTENSOR_TOD_ADD_IMPL_H
#define LIBTENSOR_TOD_ADD_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_add.h"
#include "../tod_copy.h"
#include "../tod_set.h"


namespace libtensor {


template<size_t N>
const char* tod_add<N>::k_clazz = "tod_add<N>";


template<size_t N>
tod_add<N>::tod_add(
        dense_tensor_rd_i<N, double> &t, const tensor_transf_t &tr) :
    m_dims(t.get_dims()) {

    m_dims.permute(tr.get_perm());
    add_operand(t, tr.get_perm(), tr.get_scalar_tr().get_coeff());
}


template<size_t N>
tod_add<N>::tod_add(dense_tensor_rd_i<N, double> &t, double c) :

    m_dims(t.get_dims()) {

    add_operand(t, permutation<N>(), c);
}


template<size_t N>
tod_add<N>::tod_add(dense_tensor_rd_i<N, double> &t, const permutation<N> &p,
    double c) :

    m_dims(t.get_dims()) {

    m_dims.permute(p);
    add_operand(t, p, c);
}


template<size_t N>
void tod_add<N>::add_op(dense_tensor_rd_i<N, double> &t, double c) {

    static const char *method = "add_op(dense_tensor_rd_i<N, double>&, double)";

    if(c == 0.0) return;

    if(!t.get_dims().equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    add_operand(t, permutation<N>(), c);
}


template<size_t N>
void tod_add<N>::add_op(dense_tensor_rd_i<N, double> &t,
    const permutation<N> &perm, double c) {

    static const char *method = "add_op(dense_tensor_rd_i<N, double>&, "
        "const permutation<N>&, double)";

    if(c == 0.0) return;

    dimensions<N> dims(t.get_dims());
    dims.permute(perm);
    if(!dims.equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    add_operand(t, perm, c);
}


template<size_t N>
void tod_add<N>::add_op(dense_tensor_rd_i<N, double> &t,
        const tensor_transf_t &tr) {

    static const char *method = "add_op(dense_tensor_rd_i<N, double>&, "
            "const tensor_transf<N, double> &)";

    if(tr.get_scalar_tr().get_coeff() == 0) return;

    dimensions<N> dims(t.get_dims());
    dims.permute(tr.get_perm());
    if(!dims.equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    add_operand(t, tr.get_perm(), tr.get_scalar_tr().get_coeff());
}

template<size_t N>
void tod_add<N>::add_operand(dense_tensor_rd_i<N, double> &t,
    const permutation<N> &perm, double c) {

    m_args.push_back(arg(t, perm, c));
}


template<size_t N>
void tod_add<N>::prefetch() {

    for(typename std::list<arg>::iterator i = m_args.begin();
        i != m_args.end(); ++i) {

        dense_tensor_rd_ctrl<N, double>(i->t).req_prefetch();
    }
}


template<size_t N>
void tod_add<N>::perform(bool zero, dense_tensor_wr_i<N, double> &t) {

    static const char *method = "perform(bool, dense_tensor_wr_i<N, double>&)";

    //  Check the dimensions of the output tensor
    if(!t.get_dims().equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    if(zero) tod_set<N>().perform(t);

    tod_add<N>::start_timer();

    typename std::list<arg>::iterator i = m_args.begin();
    for(; i != m_args.end(); ++i) {
        tod_copy<N>(i->t, i->perm, i->c).perform(false, t);
    }

    tod_add<N>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_ADD_IMPL_H
