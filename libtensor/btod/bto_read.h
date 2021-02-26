#ifndef LIBTENSOR_BTO_READ_H
#define LIBTENSOR_BTO_READ_H

#include <istream>
#include <libtensor/defs.h>
#include <libtensor/exception.h>
#include <libtensor/timings.h>
#include <libtensor/core/allocator.h>
#include <libtensor/core/bad_dimensions.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include "bto_import_raw.h"

namespace libtensor {


/** \brief Reads block %tensors from an input stream
    \tparam N Tensor order.
    \tparam Allocator Allocator for temporary tensors.

    The operation fills a block %tensor with data read from a formatted
    text input stream. Items in the stream are separated by whitespace
    characters. The format does not treat the new line character in any
    special way, it is another whitespace character.

    The first item in the stream is an integer specifying the order of
    the %tensor followed by a series of integers that specify the number of
    elements along each dimension of the %tensor. Then follow the actual
    data, each %tensor element is a T precision floating point number.

    After reading the data from the stream, the operation looks for zero
    blocks by checking that all elements are zero within a threshold
    (default 0.0 meaning that the elements must be exactly zero).

    The %symmetry of the block %tensor is guessed from the initial %symmetry
    set by the user before calling the operation. It is verified that
    the data actually have the specified %symmetry, otherwise an exception
    is raised. The comparison is done using a %symmetry threshold, within
    which two related elements are considered equal. The default value
    for the threshold is 0.0 meaning that the elements must be equal
    exactly.

    Format of the input stream:
    \code
    N D1 D2 ... Dn
    A1 A2 A3 ...
    \endcode
    N -- number of dimensions (integer); D1, D2, ..., Dn -- size of the
    %tensor along each dimension (N integers); A1, A2, A3, ... --
    %tensor elements (Ts).

    Example of a 3 by 3 antisymmetric matrix:
    \code
    2 3 3
    0.1 0.0 2.0
    0.0 1.3 -0.1
    -2.0 0.1 5.1
    \endcode

    \sa bto_import_raw<N, Alloc>

    \ingroup libtensor_btod
 **/
template<size_t N, typename T, typename Alloc = allocator >
class bto_read : public timings< bto_read<N, T, Alloc> > {
public:
    static const char *k_clazz; //!< Class name

private:
    std::istream &m_stream; //!< Input stream
    T m_zero_thresh; //!< Zero threshold
    T m_sym_thresh; //!< Symmetry threshold

public:
    //!    \name Construction and destruction
    //@{

    bto_read(std::istream &stream, T zero_thresh, T sym_thresh) :
        m_stream(stream), m_zero_thresh(zero_thresh),
        m_sym_thresh(sym_thresh) { }

    bto_read(std::istream &stream, T thresh = 0.0) :
        m_stream(stream), m_zero_thresh(thresh), m_sym_thresh(thresh)
        { }

    //@}

    //!    \name Operation
    //@{

    void perform(block_tensor_i<N, T> &bt);

    //@}

private:
    bto_read(const bto_read<N, T, Alloc>&);
    const bto_read<N, T, Alloc> &operator=(const bto_read<N, T, Alloc>&);

};


template<size_t N, typename T, typename Alloc>
const char *bto_read<N, T, Alloc>::k_clazz = "bto_read<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
void bto_read<N, T, Alloc>::perform(block_tensor_i<N, T> &bt) {

    static const char *method = "perform(block_tensor_i<N, T>&)";

    bto_read<N, T>::start_timer();

    //
    //  Read the formatted data
    //

    if(!m_stream.good()) {
        throw bad_parameter(g_ns, k_clazz, method,
            __FILE__, __LINE__, "stream");
    }

    int order;
    index<N> i1, i2;
    size_t k = 0;
    m_stream >> order;
    if(order != N) {
        throw_exc(k_clazz, method, "Incorrect tensor order.");
    }
    while(m_stream.good() && k < N) {
        int dim;
        m_stream >> dim;
        if(dim <= 0) {
            throw_exc(k_clazz, method,
                "Incorrect tensor dimension.");
        }
        i2[k] = dim - 1;
        k++;
    }
    if(k < N) {
        throw_exc(k_clazz, method, "Unexpected end of stream.");
    }

    const block_index_space<N> &bis = bt.get_bis();
    dimensions<N> dims(index_range<N>(i1, i2));
    dimensions<N> bidims(bis.get_block_index_dims());
    if(!dims.equals(bis.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "stream");
    }

    //
    //  Read tensor elements from file into a buffer
    //

    typename Alloc::pointer_type buf_ptr = Alloc::allocate(dims.get_size());
    T *buf = (T*)Alloc::lock_rw(buf_ptr);

    for(size_t i = 0; i < dims.get_size(); i++) {
        if(!m_stream.good()) {
            Alloc::unlock_rw(buf_ptr); buf = 0;
            Alloc::deallocate(buf_ptr);

            throw_exc(k_clazz, method, "Unexpected end of stream.");
        }
        m_stream >> buf[i];
    }

    //
    //  Import from the buffer to the block tensor
    //

    bto_import_raw<N, T, Alloc>(buf, dims, m_zero_thresh, m_sym_thresh).
        perform(bt);

    Alloc::unlock_rw(buf_ptr); buf = 0;
    Alloc::deallocate(buf_ptr);

    bto_read<N, T>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_READ_H
