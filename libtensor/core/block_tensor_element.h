#ifndef LIBTENSOR_BLOCK_TENSOR_ELEMENT_H
#define LIBTENSOR_BLOCK_TENSOR_ELEMENT_H

#include "index.h"
#include "tensor_element.h"

namespace libtensor {


/** \brief Representation of elements of block tensors

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class block_tensor_element {
public:
    typedef T element_type;
private:
    index<N> m_bidx; //!< Block index of %tensor element
    index<N> m_iblidx; //!< Index of %tensor element with in %tensor block
    element_type m_val; //!< Value of tensor element

public:
    /** \brief Create block tensor element
        \param bidx Block index
        \param iblidx Index within the block
        \param elem Element value
     **/
    block_tensor_element(const index<N> &bidx, const index<N> &iblidx,
            const element_type &val) :
        m_bidx(bidx), m_iblidx(iblidx), m_val(val) {

    }

    /** \brief Create block tensor element
        \param bidx Block index
        \param te Tensor element of tensor block
        \param iblidx Index within the block
        \param elem Element value
     **/
    block_tensor_element(const index<N> &bidx,
            const tensor_element<N, element_type> &te) :
        m_bidx(bidx), m_iblidx(te.get_index()), m_val(te.get_value()) {

    }

    //! \name Data access functions (getters)
    //@{

    /** \brief Return block index
     **/
    const index<N> &get_block_index() const { return m_bidx; }

    /** \brief Return index within the block
     **/
    const index<N> &get_in_block_index() const { return m_iblidx; }

    /** \brief Return value of block tensor element
     **/
    const element_type &get_value() const { return m_val; }

    //@}
};


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_ELEMENT_H
