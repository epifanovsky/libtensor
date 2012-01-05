#ifndef LIBTENSOR_TOD_IMPORT_RAW_STREAM_H
#define LIBTENSOR_TOD_IMPORT_RAW_STREAM_H

#include <iostream>
#include "../core/dimensions.h"
#include "../core/index_range.h"
#include "../core/tensor_ctrl.h"
#include "bad_dimensions.h"

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

    \ingroup libtensor_tod
 **/
template<size_t N>
class tod_import_raw_stream {
public:
    static const char *k_clazz; //!< Class name

private:
    std::istream &m_is; //!< Input stream
    dimensions<N> m_dims; //!< Dimensions of the tensor contained in the stream
    index_range<N> m_ir; //!< Index range of the window

public:
    /**	\brief Initializes the operation
        \param is Input stream (must be opened in the binary mode).
        \param dims Dimensions of the tensor contained in the stream.
        \param ir Index range of the window that will be read.
     **/
    tod_import_raw_stream(std::istream &is, const dimensions<N> &dims,
        const index_range<N> &ir) : m_is(is), m_dims(dims), m_ir(ir) { }

    /**	\brief Performs the operation
        \param ta Output %tensor.
     **/
    void perform(dense_tensor_i<N, double> &ta);

private:
    template<size_t M>
    void read_data(size_t pos, const dimensions<M> &dims,
        const index_range<M> &ir, double *p);

    void read_data(size_t pos, const dimensions<1> &dims,
        const index_range<1> &ir, double *p);

};


} // namespace libtensor

#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

extern template class tod_import_raw_stream<1>;
extern template class tod_import_raw_stream<2>;
extern template class tod_import_raw_stream<3>;
extern template class tod_import_raw_stream<4>;
extern template class tod_import_raw_stream<5>;
extern template class tod_import_raw_stream<6>;

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "tod_import_raw_stream_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES

#endif // LIBTENSOR_TOD_IMPORT_RAW_STREAM_H
