#ifndef LIBTENSOR_BTO_SHIFT_DIAG_H
#define LIBTENSOR_BTO_SHIFT_DIAG_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/bto_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_shift_diag.h>

namespace libtensor {


/** \brief Assigns the diagonal elements of a block %tensor to a value
    \tparam N Tensor order.

    This operation sets the diagonal elements of a block %tensor to a value
    without affecting all the off-diagonal elements.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, typename T>
class bto_shift_diag : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_shift_diag<N, bto_traits<T> > m_gbto;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param msk Diagonal mask
        \param v Tensor element value (default 0.0).
     **/
    bto_shift_diag(const sequence<N, size_t> &msk, T v = 0.0);

    //@}

    //!    \name Operation
    //@{

    /** \brief Performs the operation
     **/
    void perform(block_tensor_i<N, T> &bt);

    //@}
};


} // namespace libtensor

#endif // LIBTENSOR_BTO_SHIFT_DIAG_H
