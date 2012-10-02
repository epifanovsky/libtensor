#ifndef LIBTENSOR_TENSOR_ELEMENT_H
#define LIBTENSOR_TENSOR_ELEMENT_H

#include "index.h"

namespace libtensor {


/** \brief Representation of elements of simple tensors

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class tensor_element {
public:
    typedef T element_type;
private:
    index<N> m_idx; //!< Index of tensor element
    element_type m_val; //!< Value of tensor element

public:
    /** \brief Create tensor element
        \param idx Index
        \param elem Element value
     **/
    tensor_element(const index<N> &idx, const element_type &val) :
        m_idx(idx), m_val(val) {

    }

    //! \name Data access functions (getters)
    //@{

    /** \brief Return index of tensor element
     **/
    const index<N> &get_index() const { return m_idx; }

    /** \brief Return value of tensor element
     **/
    const element_type &get_value() const { return m_val; }

    //@}
};


} // namespace libtensor

#endif // LIBTENSOR_TENSOR_ELEMENT_H
