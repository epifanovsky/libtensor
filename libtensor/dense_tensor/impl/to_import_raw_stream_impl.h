#ifndef LIBTENSOR_TO_IMPORT_RAW_STREAM_IMPL_H
#define LIBTENSOR_TO_IMPORT_RAW_STREAM_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../to_import_raw_stream.h"

namespace libtensor {


template<size_t N, typename T>
const char *to_import_raw_stream<N, T>::k_clazz = "to_import_raw_stream<N, T>";


template<size_t N, typename T>
void to_import_raw_stream<N, T>::perform(dense_tensor_wr_i<N, T> &ta) {

    static const char *method = "perform(tensor_i<N, T>&)";

    dimensions<N> dimsa(m_ir);
    if(!ta.get_dims().equals(dimsa)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    dense_tensor_wr_ctrl<N, T> ca(ta);
    T *pa = ca.req_dataptr();
    read_data(0, m_dims, m_ir, pa);
    ca.ret_dataptr(pa);
}


template<size_t N, typename T> template<size_t M>
void to_import_raw_stream<N, T>::read_data(size_t pos, const dimensions<M> &dims,
    const index_range<M> &ir, T *p) {

    index<M - 1> i1, i2, i3, i4;
    for(size_t i = 1; i < M; i++) {
        i2[i - 1] = dims[i] - 1;
        i3[i - 1] = ir.get_begin().at(i);
        i4[i - 1] = ir.get_end().at(i);
    }

    dimensions<M - 1> dims2(index_range<M - 1>(i1, i2));
    index_range<M - 1> ir2(i3, i4);
    size_t sz2src = dims2.get_size();
    size_t sz2dst = dimensions<M - 1>(ir2).get_size();
    size_t j0 = ir.get_begin().at(0), j1 = ir.get_end().at(0);
    for(size_t j = j0; j <= j1; j++) {
        read_data(pos + j * sz2src, dims2, ir2, p + (j - j0) * sz2dst);
    }
}


template<size_t N, typename T>
void to_import_raw_stream<N, T>::read_data(size_t pos, const dimensions<1> &dims,
    const index_range<1> &ir, T *p) {

    size_t len = ir.get_end().at(0) - ir.get_begin().at(0) + 1;
    m_is.seekg((pos + ir.get_begin().at(0)) * sizeof(T));
    m_is.read((char*)p, len * sizeof(T));
}


} // namespace libtensor

#endif // LIBTENSOR_TO_IMPORT_RAW_STREAM_IMPL_H
