#ifndef LIBTENSOR_CUDA_TOD_ADD_IMPL_H
#define LIBTENSOR_CUDA_TOD_ADD_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "../cuda_tod_add.h"
#include "../cuda_tod_copy.h"
#include "../cuda_tod_set.h"

namespace libtensor {


template<size_t N>
const char* cuda_tod_add<N>::k_clazz = "cuda_tod_add<N>";


template<size_t N>
cuda_tod_add<N>::cuda_tod_add(dense_tensor_i<N, double> &t, double c) : m_dims(t.get_dims()) {

    static const char *method = "cuda_tod_add(tensor_i<N, double>&, double)";

    add_operand(t, permutation<N>(), c);
}


template<size_t N>
cuda_tod_add<N>::cuda_tod_add(dense_tensor_i<N, double> &t, const permutation<N> &p, double c) :
    m_dims(t.get_dims()) {

    static const char *method =
        "cuda_tod_add(tensor_i<N, double>&, const permutation<N>&, double)";

    m_dims.permute(p);
    add_operand(t, p, c);
}


template<size_t N>
cuda_tod_add<N>::~cuda_tod_add() {

}


template<size_t N>
void cuda_tod_add<N>::add_op(dense_tensor_i<N, double> &t, double c) {

    static const char *method = "add_op(tensor_i<N, double>&, double)";

    if(c == 0.0) return;

    if(!t.get_dims().equals(m_dims)) {
        throw bad_dimensions(
            g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    add_operand(t, permutation<N>(), c);
}


template<size_t N>
void cuda_tod_add<N>::add_op(
    dense_tensor_i<N, double> &t, const permutation<N> &p, double c) {

    static const char *method =
        "add_op(tensor_i<N, double>&, const permutation<N>&, double)";

    if(c == 0.0) return;

    dimensions<N> dims(t.get_dims());
    dims.permute(p);
    if(!dims.equals(m_dims)) {
        throw bad_dimensions(
            g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    add_operand(t, p, c);
}


template<size_t N>
void cuda_tod_add<N>::add_operand(dense_tensor_i<N, double> &t, const permutation<N> &p,
    double c) {

    static const char *method = "add_operand(tensor_i<N, double>&, "
        "const permutation<N>&, double)";

    m_args.push_back(arg(t, p, c));
}


template<size_t N>
void cuda_tod_add<N>::prefetch() {

    for(typename std::list<arg>::iterator i = m_args.begin();
        i != m_args.end(); ++i) {

        dense_tensor_ctrl<N, double>(i->t).req_prefetch();
    }
}


template<size_t N>
void cuda_tod_add<N>::perform(bool zero, double c,
    dense_tensor_i<N, double> &t) {

    static const char *method = "perform(bool, double, tensor_i<N, double>&)";

    //  Check the dimensions of the output tensor
    if(!t.get_dims().equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    if(zero) cuda_tod_set<N>().perform(t);
    if(c == 0.0) return;

    cuda_tod_add::start_timer();

    typename std::list<arg>::iterator i = m_args.begin();
    for(; i != m_args.end(); ++i) {
        cuda_tod_copy<N>(i->t, i->p, i->c).perform(t, c);
    }

    cuda_tod_add::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_CUDA_TOD_ADD_IMPL_H
