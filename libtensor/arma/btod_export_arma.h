#ifndef LIBTENSOR_BTOD_EXPORT_ARMA_H
#define LIBTENSOR_BTOD_EXPORT_ARMA_H

#include <armadillo>
#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/block_tensor_i.h>

namespace libtensor {


/** \brief Writes block tensor in a matricized form to an Armadillo matrix
    \tparam N Tensor order.

    Copies the contents of a block tensor to an Armadillo matrix. The matrix
    and the block tensor must agree in dimensions.

    Beware of the format difference between Armadillo and libtensor. Armadillo
    stores matrices in the row-major format, libtensor uses column-major
    ordering. The user is responsible for format conversion.

    \ingroup libtensor_arma
 **/
template<size_t N>
class btod_export_arma : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    block_tensor_rd_i<N, double> &m_bt;

public:
    /** \brief Initializes the operation
        \param bt Block tensor.
     **/
    btod_export_arma(block_tensor_rd_i<N, double> &bt);

    /** \brief Copies the tensor
        \param m Armadillo matrix.
     **/
    void perform(arma::Mat<double> &m);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EXPORT_ARMA_H

