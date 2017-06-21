#ifndef LIBTENSOR_BTO_TRIDIAGONALIZE_H
#define LIBTENSOR_BTO_TRIDIAGONALIZE_H

#include <libtensor/block_tensor/block_tensor_i.h>

namespace libtensor {
/** \brief Converts a symmetric matrix to the tridiagonal matrix using
 *     Householder's reflections

    \ingroup libtensor_btod
 **/
template<typename T>
class bto_tridiagonalize {
public:
    bto_tridiagonalize(block_tensor_i<2, T> &bta);
    //!< bta - input symmetric matrix
    virtual void perform(block_tensor_i<2, T> &btb, block_tensor_i<2, T> &S);
    //!< btb - output tridiag matrix,S - matrix of transformation
    virtual void print(block_tensor_i<2, T> &btb);
    //!< (Optional) prints the matrix
private:
    block_tensor_i<2, T> &m_bta; //!< Input block %tensor
};

using btod_tridiagonalize = bto_tridiagonalize<double>;
}

#endif // LIBTENSOR_BTO_TRIDIAGONALIZE_H
