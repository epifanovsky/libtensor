#ifndef LIBTENSOR_TO_IMPORT_RAW_H
#define LIBTENSOR_TO_IMPORT_RAW_H

#include <list>
#include <libtensor/core/dimensions.h>
#include <libtensor/core/bad_dimensions.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Imports %tensor elements from memory
    \tparam N Tensor order.

    This operation reads %tensor elements from a given window of a block
    of memory. The elements in the memory must be in the usual %tensor
    format. The block is characterized by its %dimensions, as if it were
    a part of the usual %tensor object. The window is specified by a range
    of indexes.

    The size of the recipient (result of the operation) must agree with
    the window dimensions.

    \ingroup libtensor_to
 **/
template<size_t N, typename T>
class to_import_raw {
public:
    static const char *k_clazz; //!< Class name

private:
    const T *m_ptr; //!< Pointer to data
    dimensions<N> m_dims; //!< Dimensions of the memory block
    index_range<N> m_ir; //!< Index range of the window

public:
    /** \brief Initializes the operation
        \param ptr Pointer to data block
        \param dims Dimensions of the data block
        \param ir Index range of the window
    **/
    to_import_raw(const T *ptr, const dimensions<N> &dims,
        const index_range<N> &ir) :
        m_ptr(ptr), m_dims(dims), m_ir(ir) { }

    /** \brief Performs the operation
        \param t Output %tensor
    **/
    void perform(dense_tensor_wr_i<N, T> &t);

};


} // namespace libtensor

#endif // LIBTENSOR_TO_IMPORT_RAW_H
