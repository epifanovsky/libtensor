#ifndef LIBTENSOR_GEN_BTO_SET_H
#define LIBTENSOR_GEN_BTO_SET_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Sets all elements of a block tensor to a value preserving
        the symmetry
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    In every block that is allowed by the symmetry of the block tensor, this
    operation sets all elements to a specified value.

    If the value is zero, all the blocks are zeroed out.

    This operation does not make sure that the symmetry of each block is
    preserved. For example, it will not zero the diagonal of a anti-symmetric
    tensor.

    <b>Traits</b>

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template to_set_type<N>::type -- Type of tensor operation to_set

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits, typename Timed>
class gen_bto_set : public timings<Timed>, public noncopyable {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

private:
    element_type m_v; //!< Value

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    gen_bto_set(const element_type &v) : m_v(v) { }

    /** \brief Performs the operation
        \param bta Output block tensor.
     **/
    void perform(gen_block_tensor_wr_i<N, bti_traits> &bta);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SET_H
