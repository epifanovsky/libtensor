#ifndef LIBTENSOR_CTF_DENSE_TENSOR_IMPL_H
#define LIBTENSOR_CTF_DENSE_TENSOR_IMPL_H

#include <libtensor/exception.h>
#include "../ctf.h"
#include "../ctf_error.h"
#include "../ctf_dense_tensor.h"

namespace libtensor {


template<size_t N, typename T>
const char ctf_dense_tensor<N, T>::k_clazz[] = "ctf_dense_tensor<N, T>";


template<size_t N, typename T>
ctf_dense_tensor<N, T>::ctf_dense_tensor(const dimensions<N> &dims) :

    m_dims(dims), m_tid(0) {

    static const char method[] = "ctf_dense_tensor(const dimensions<N>&)";

#ifdef LIBTENSOR_DEBUG
    if(m_dims.get_size() == 0) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "dims");
    }
#endif // LIBTENSOR_DEBUG

    int edge_len[N], sym[N];
    for(size_t i = 0; i < N; i++) {
        edge_len[i] = m_dims[i];
        sym[i] = 0;
    }
    int res = ctf::get().define_tensor(N, edge_len, sym, &m_tid);
    if(res != DIST_TENSOR_SUCCESS) {
        throw ctf_error(g_ns, k_clazz, method, __FILE__, __LINE__, "");
    }
}


template<size_t N, typename T>
ctf_dense_tensor<N, T>::~ctf_dense_tensor() {

    try {
        ctf::get().clean_tensor(m_tid);
    } catch(...) {
    }
}


template<size_t N, typename T>
const dimensions<N> &ctf_dense_tensor<N, T>::get_dims() const {

    return m_dims;
}


template<size_t N, typename T>
int ctf_dense_tensor<N, T>::on_req_tensor_id() {

    static const char method[] = "on_req_tensor_id()";

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__, "");
    }

    return m_tid;
}


template<size_t N, typename T>
void ctf_dense_tensor<N, T>::on_set_immutable() {

}


} // namespace libtensor

#endif // LIBTENSOR_CTF_DENSE_TENSOR_IMPL_H
