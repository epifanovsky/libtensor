#ifndef LIBTENSOR_CTF_DENSE_TENSOR_IMPL_H
#define LIBTENSOR_CTF_DENSE_TENSOR_IMPL_H

#include <algorithm>
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

    ctf_symmetry<N, T> sym;
    on_reset_symmetry(sym);
}


template<size_t N, typename T>
ctf_dense_tensor<N, T>::ctf_dense_tensor(const dimensions<N> &dims,
    const ctf_symmetry<N, T> &sym) :

    m_dims(dims), m_tens(0) {

    static const char method[] =
        "ctf_dense_tensor(const dimensions<N>&, const ctf_symmetry<N, T>&)";

#ifdef LIBTENSOR_DEBUG
    if(m_dims.get_size() == 0) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "dims");
    }
#endif // LIBTENSOR_DEBUG

    on_reset_symmetry(sym);
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
const ctf_symmetry<N, T> &ctf_dense_tensor<N, T>::on_req_symmetry() {

    return m_sym;
}


template<size_t N, typename T>
void ctf_dense_tensor<N, T>::on_reset_symmetry(const ctf_symmetry<N, T> &sym) {

    delete m_tens;
    m_sym = sym;

    //  CTF stores tensors in the column-major format,
    //  need to use the reverse order of dimensions

    int edge_len[N], edge_sym[N];
    for(size_t i = 0; i < N; i++) {
        edge_len[i] = m_dims[N - i - 1];
        edge_sym[i] = NS;
    }
    m_sym.write(edge_sym);
    m_tens = new tCTF_Tensor<T>(N, edge_len, edge_sym, ctf::get_world());
}


template<size_t N, typename T>
void ctf_dense_tensor<N, T>::on_adjust_symmetry(const ctf_symmetry<N, T> &sym) {

    static const char method[] = "on_adjust_symmetry()";

    //  Check that the new symmetry is a subgroup of the original symmetry
    //  (i.e. no data will be lost)
    if(!m_sym.is_subgroup(sym)) {
        throw symmetry_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
                "!m_sym.is_subgroup(sym)");
    }

    ctf_dense_tensor<N, T> tmp(m_dims, sym);

    char idxmap[N];
    for(size_t i = 0; i < N; i++) idxmap[i] = i;
    tmp.m_tens->sum(1.0, *m_tens, idxmap, 0.0, idxmap);
    std::swap(m_tens, tmp.m_tens);
    std::swap(m_sym, tmp.m_sym);
}


template<size_t N, typename T>
void ctf_dense_tensor<N, T>::on_set_immutable() {

}


} // namespace libtensor

#endif // LIBTENSOR_CTF_DENSE_TENSOR_IMPL_H
