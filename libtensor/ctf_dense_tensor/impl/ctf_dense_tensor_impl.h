#ifndef LIBTENSOR_CTF_DENSE_TENSOR_IMPL_H
#define LIBTENSOR_CTF_DENSE_TENSOR_IMPL_H

#include <libtensor/exception.h>
#include "../ctf_error.h"
#include "../ctf_dense_tensor.h"

namespace libtensor {


template<size_t N, typename T>
const char ctf_dense_tensor<N, T>::k_clazz[] = "ctf_dense_tensor<N, T>";


template<size_t N, typename T>
ctf_dense_tensor<N, T>::ctf_dense_tensor(const dimensions<N> &dims) :

    m_dims(dims), m_tens(0) {

    static const char method[] = "ctf_dense_tensor(const dimensions<N>&)";

#ifdef LIBTENSOR_DEBUG
    if(m_dims.get_size() == 0) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "dims");
    }
#endif // LIBTENSOR_DEBUG

    //  CTF stores tensors in the column-major format,
    //  need to use the reverse order of dimensions

    int edge_len[N], sym[N];
    for(size_t i = 0; i < N; i++) {
        edge_len[i] = m_dims[N - i - 1];
        sym[i] = NS;
    }
    m_tens = new tCTF_Tensor<T>(N, edge_len, sym, ctf::get_world());
}


template<size_t N, typename T>
ctf_dense_tensor<N, T>::~ctf_dense_tensor() {

    delete m_tens;
}


template<size_t N, typename T>
const dimensions<N> &ctf_dense_tensor<N, T>::get_dims() const {

    return m_dims;
}


template<size_t N, typename T>
tCTF_Tensor<T> &ctf_dense_tensor<N, T>::on_req_ctf_tensor() {

    return *m_tens;
}


template<size_t N, typename T>
void ctf_dense_tensor<N, T>::on_set_immutable() {

}


} // namespace libtensor

#endif // LIBTENSOR_CTF_DENSE_TENSOR_IMPL_H
