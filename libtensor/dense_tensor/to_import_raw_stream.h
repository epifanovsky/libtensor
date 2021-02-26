#ifndef LIBTENSOR_TOD_IMPORT_RAW_STREAM_H
#define LIBTENSOR_TOD_IMPORT_RAW_STREAM_H

#include <iostream>
#include <libtensor/core/dimensions.h>
#include "dense_tensor_i.h"

namespace libtensor {

/** \brief Imports tensor data from a raw binary stream
    \tparam N Tensor order.

    Given an input binary stream that contains tensor data, this operation
    reads the data from the specified window and places the result in the
    output tensor.

    The stream is characterized by the dimensions of the tensor it contains,
    and the window is specified by an index range. The size of the output tensor
    must agree with the dimensions of the window.

    The input stream must be opened in the binary mode prior to invoking this
    operation and kept live until reading has finished.

    \ingroup libtensor_to
 **/
template<size_t N, typename T>
class to_import_raw_stream {
public:
    static const char *k_clazz; //!< Class name

private:
    std::istream &m_is; //!< Input stream
    dimensions<N> m_dims; //!< Dimensions of the tensor contained in the stream
    index_range<N> m_ir; //!< Index range of the window

public:
    /** \brief Initializes the operation
        \param is Input stream (must be opened in the binary mode).
        \param dims Dimensions of the tensor contained in the stream.
        \param ir Index range of the window that will be read.
     **/
    to_import_raw_stream(std::istream &is, const dimensions<N> &dims,
        const index_range<N> &ir) : m_is(is), m_dims(dims), m_ir(ir) { }

    /** \brief Performs the operation
        \param ta Output %tensor.
     **/
    void perform(dense_tensor_wr_i<N, T> &ta);

private:
    template<size_t M>
    void read_data(size_t pos, const dimensions<M> &dims,
        const index_range<M> &ir, T *p);

    void read_data(size_t pos, const dimensions<1> &dims,
        const index_range<1> &ir, T *p);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_IMPORT_RAW_STREAM_H
