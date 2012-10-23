#ifndef LIBTENSOR_GEN_BTO_SET_DIAG_H
#define LIBTENSOR_GEN_BTO_SET_DIAG_H

#include <libtensor/core/noncopyable.h>
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Assigns the diagonal elements of a block %tensor to a value
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.

    This operation sets the diagonal elements of a block %tensor to a value
    without affecting all the off-diagonal elements.

    <b>Traits</b>

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template to_set_diag_type<N>::type -- Type of tensor operation
        to_set_diag
    - \c template to_set_type<N>::type -- Type of tensor operation to_set

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits>
class gen_bto_set_diag : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;

private:
    element_type m_v; //!< Value

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param v Tensor element value (default zero).
     **/
    gen_bto_set_diag(const element_type &v = Traits::zero());

    //@}

    //!    \name Operation
    //@{

    /** \brief Performs the operation
     **/
    void perform(gen_block_tensor_i<N, bti_traits> &bt);

    //@}
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SET_DIAG_H
