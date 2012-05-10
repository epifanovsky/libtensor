#ifndef LIBTENSOR_TOD_ADD_CUDA_IMPL_H
#define LIBTENSOR_TOD_ADD_CUDA_IMPL_H

#include "tod_set_cuda.h"
//#include "tod_set.h"

namespace libtensor {


template<size_t N>
const char* tod_add_cuda<N>::k_clazz = "tod_add_cuda<N>";


template<size_t N>
tod_add_cuda<N>::tod_add_cuda(dense_tensor_i<N, double> &t, double c) : m_dims(t.get_dims()) {

    static const char *method = "tod_add_cuda(tensor_i<N, double>&, double)";

    add_operand(t, permutation<N>(), c);
}


template<size_t N>
tod_add_cuda<N>::tod_add_cuda(dense_tensor_i<N, double> &t, const permutation<N> &p, double c) :
    m_dims(t.get_dims()) {

    static const char *method =
        "tod_add_cuda(tensor_i<N, double>&, const permutation<N>&, double)";

    m_dims.permute(p);
    add_operand(t, p, c);
}


template<size_t N>
tod_add_cuda<N>::~tod_add_cuda() {

}


template<size_t N>
void tod_add_cuda<N>::add_op(dense_tensor_i<N, double> &t, double c) {

    static const char *method = "add_op(tensor_i<N, double>&, double)";

    if(c == 0.0) return;

    if(!t.get_dims().equals(m_dims)) {
        throw bad_dimensions(
            g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    add_operand(t, permutation<N>(), c);
}


template<size_t N>
void tod_add_cuda<N>::add_op(
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
void tod_add_cuda<N>::add_operand(dense_tensor_i<N, double> &t, const permutation<N> &p,
    double c) {

    static const char *method = "add_operand(tensor_i<N, double>&, "
        "const permutation<N>&, double)";

    m_args.push_back(arg(t, p, c));
}


template<size_t N>
void tod_add_cuda<N>::prefetch() {

    for(typename std::list<arg>::iterator i = m_args.begin();
        i != m_args.end(); ++i) {

        dense_tensor_ctrl<N, double>(i->t).req_prefetch();
    }
}


template<size_t N>
void tod_add_cuda<N>::perform(bool zero, double c,
    dense_tensor_i<N, double> &t) {

    static const char *method = "perform(bool, double, tensor_i<N, double>&)";

    //  Check the dimensions of the output tensor
    if(!t.get_dims().equals(m_dims)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    if(zero) tod_set_cuda<N>().perform(t);
    if(c == 0.0) return;

    tod_add_cuda<N>::start_timer();

    typename std::list<arg>::iterator i = m_args.begin();
    for(; i != m_args.end(); ++i) {
        tod_cuda_copy<N>(i->t, i->p, i->c).perform(t, c);
    }

    tod_add_cuda<N>::stop_timer();
}

/*
template<size_t N>
void tod_add_cuda<N>::perform(dense_tensor_i<N, double> &t) {

    static const char *method = "perform(dense_tensor_i<N, double>&)";

    //    Check the dimensions of the output tensor
    if(!t.get_dims().equals(m_dims)) {
        throw bad_dimensions(
            g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    tod_add_cuda<N>::start_timer();

    typename std::list<arg>::iterator i = m_args.begin();
    tod_copy<N>(i->t, i->p, i->c).perform(t);
    i++;
    for(; i != m_args.end(); i++) {
        tod_copy<N>(i->t, i->p, i->c).perform(t, 1.0);
    }

    tod_add_cuda<N>::stop_timer();
}


template<size_t N>
void tod_add_cuda<N>::perform(dense_tensor_i<N, double> &t, double c) {

    static const char *method = "perform(dense_tensor_i<N, double>&, double)";

    //    Check the dimensions of the output tensor
    if(!t.get_dims().equals(m_dims)) {
        throw bad_dimensions(
            g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    if(c == 0.0) return;

    tod_add_cuda<N>::start_timer();

    typename std::list<arg>::iterator i = m_args.begin();
    for(; i != m_args.end(); i++) {
        tod_copy<N>(i->t, i->p, i->c).perform(t, c);
    }

    tod_add_cuda<N>::stop_timer();
}*/


} // namespace libtensor

#endif // LIBTENSOR_TOD_ADD_CUDA_IMPL_H
