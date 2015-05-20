#ifndef LIBTENSOR_GEN_BTO_SHIFT_DIAG_H
#define LIBTENSOR_GEN_BTO_SHIFT_DIAG_H

#include <libtensor/core/noncopyable.h>
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Shifts the elements of a generalized diagonal of a block %tensor by
        a constant value
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.

    This operation shifts the elements of a generalized diagonal of a block
    %tensor by a constant value without affecting the off-diagonal elements.
    The generalized diagonal is defined using a masking sequence. Dimensions
    for which the mask is 0 are not part of the diagonal, while dimensions for
    which the mask has the same non-zero value are taken as diagonal. For
    example, using the operation on a block tensor \f$ a\f$ with mask
    \c [10221] will only affect the elements
    \f$ a_{ijkki}, \forall i, j, k \f$ .

    The block structure of the dimensions of the block tensor belonging to
    the same diagonal must be identical.

    <b>Traits</b>

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template to_shift_diag_type<N>::type -- Type of tensor operation
        to_shift_diag
    - \c template to_set_type<N>::type -- Type of tensor operation to_set

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits>
class gen_bto_shift_diag : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;

private:
    sequence<N, size_t> m_msk; //!< Diagonal mask
    element_type m_v; //!< Value

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param msk Diagonal mask
        \param v Tensor element value (default zero).
     **/
    gen_bto_shift_diag(const sequence<N, size_t> &msk,
        const element_type &v = Traits::zero());

    //@}

    //!    \name Operation
    //@{

    /** \brief Performs the operation
     **/
    void perform(gen_block_tensor_i<N, bti_traits> &bt);

    //@}
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SHIFT_DIAG_H
