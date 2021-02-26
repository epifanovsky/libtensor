#ifndef LIBTENSOR_TO_ADD_IMPL_H
#define LIBTENSOR_TO_ADD_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../to_add.h"
#include "../to_copy.h"
#include "../to_set.h"


namespace libtensor {


template<size_t N, typename T>
const char* to_add<N, T>::k_clazz = "to_add<N, T>";


template<size_t N, typename T>
to_add<N, T>::to_add(
        dense_tensor_rd_i<N, T> &t, const tensor_transf_t &tr) :
    m_dims(t.get_dims()) {

    m_dims.permute(tr.get_perm());
    add_operand(t, tr.get_perm(), tr.get_scalar_tr().get_coeff());
}


template<size_t N, typename T>
to_add<N, T>::to_add(dense_tensor_rd_i<N, T> &t, T c) :

    m_dims(t.get_dims()) {

    add_operand(t, permutation<N>(), c);
}


template<size_t N, typename T>
to_add<N, T>::to_add(dense_tensor_rd_i<N, T> &t, const permutation<N> &p,
    T c) :

    m_dims(t.get_dims()) {

    m_dims.permute(p);
    add_operand(t, p, c);
}


template<size_t N, typename T>
void to_add<N, T>::add_op(dense_tensor_rd_i<N, T> &t, T c) {

    static const char *method = "add_op(dense_tensor_rd_i<N, T>&, T)";

    if(c == 0.0) return;

    if(!t.get_dims().equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    add_operand(t, permutation<N>(), c);
}


template<size_t N, typename T>
void to_add<N, T>::add_op(dense_tensor_rd_i<N, T> &t,
    const permutation<N> &perm, T c) {

    static const char *method = "add_op(dense_tensor_rd_i<N, T>&, "
        "const permutation<N>&, T)";

    if(c == 0.0) return;

    dimensions<N> dims(t.get_dims());
    dims.permute(perm);
    if(!dims.equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    add_operand(t, perm, c);
}


template<size_t N, typename T>
void to_add<N, T>::add_op(dense_tensor_rd_i<N, T> &t,
        const tensor_transf_t &tr) {

    static const char *method = "add_op(dense_tensor_rd_i<N, T>&, "
            "const tensor_transf<N, T> &)";

    if(tr.get_scalar_tr().get_coeff() == 0) return;

    dimensions<N> dims(t.get_dims());
    dims.permute(tr.get_perm());
    if(!dims.equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    add_operand(t, tr.get_perm(), tr.get_scalar_tr().get_coeff());
}

template<size_t N, typename T>
void to_add<N, T>::add_operand(dense_tensor_rd_i<N, T> &t,
    const permutation<N> &perm, T c) {

    m_args.push_back(arg(t, perm, c));
}


template<size_t N, typename T>
void to_add<N, T>::prefetch() {

    for(typename std::list<arg>::iterator i = m_args.begin();
        i != m_args.end(); ++i) {

        dense_tensor_rd_ctrl<N, T>(i->t).req_prefetch();
    }
}


template<size_t N, typename T>
void to_add<N, T>::perform(bool zero, dense_tensor_wr_i<N, T> &t) {

    static const char *method = "perform(bool, dense_tensor_wr_i<N, T>&)";

    //  Check the dimensions of the output tensor
    if(!t.get_dims().equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    if(zero) to_set<N, T>().perform(zero, t);

    to_add<N, T>::start_timer();

    typename std::list<arg>::iterator i = m_args.begin();
    for(; i != m_args.end(); ++i) {
        to_copy<N, T>(i->t, i->perm, i->c).perform(false, t);
    }

    to_add<N, T>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TO_ADD_IMPL_H
