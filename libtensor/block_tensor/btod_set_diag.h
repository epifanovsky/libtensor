#ifndef LIBTENSOR_BTOD_SET_DIAG_H
#define LIBTENSOR_BTOD_SET_DIAG_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_set_diag.h>

namespace libtensor {


/** \brief Assigns the diagonal elements of a block %tensor to a value
    \tparam N Tensor order.

    This operation sets the diagonal elements of a block %tensor to a value
    without affecting all the off-diagonal elements.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_set_diag : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_set_diag<N, btod_traits> m_gbto;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param v Tensor element value (default 0.0).
     **/
    btod_set_diag(double v = 0.0);
    //@}

    //!    \name Operation
    //@{

    /** \brief Performs the operation
     **/
    void perform(block_tensor_i<N, double> &bt);

    //@}
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_DIAG_H
