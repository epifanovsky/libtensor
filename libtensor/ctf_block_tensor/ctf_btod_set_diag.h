#ifndef LIBTENSOR_CTF_BTOD_SET_DIAG_H
#define LIBTENSOR_CTF_BTOD_SET_DIAG_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/gen_bto_set_diag.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief Assigns the diagonal elements of a distributed block tensor
        to a value
    \tparam N Tensor order.

    This operation sets the diagonal elements of a block %tensor to a value
    without affecting all the off-diagonal elements.

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N>
class ctf_btod_set_diag : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    gen_bto_set_diag<N, ctf_btod_traits> m_gbto;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param msk Diagonal mask
        \param v Tensor element value (default 0.0).
     **/
    ctf_btod_set_diag(const sequence<N, size_t> &msk, double v = 0.0);

    /** \brief Initializes the operation (compatability wrapper)
        \param v Tensor element value (default 0.0).
     **/
    ctf_btod_set_diag(double v = 0.0);

    //@}

    //!    \name Operation
    //@{

    /** \brief Performs the operation
     **/
    void perform(ctf_block_tensor_i<N, double> &bt);

    //@}
};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_SET_DIAG_H
