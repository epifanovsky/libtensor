#ifndef LIBTENSOR_GEN_BTO_SET_DIAG_H
#define LIBTENSOR_GEN_BTO_SET_DIAG_H

#include <libtensor/core/noncopyable.h>
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Assigns the diagonal elements of a block %tensor to a value
    \tparam N Tensor order.

    This operation sets the diagonal elements of a block %tensor to a value
    without affecting all the off-diagonal elements.

    \ingroup libtensor_btod
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
