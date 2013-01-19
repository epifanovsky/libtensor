#ifndef LIBTENSOR_CUDA_BTOD_COPY_H2D_H
#define LIBTENSOR_CUDA_BTOD_COPY_H2D_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include "cuda_block_tensor_i.h"

namespace libtensor {


/** \brief Copies a block tensor from host to device memory
    \tparam N Tensor order.

    \ingroup libtensor_cuda_btod
 **/
template<size_t N>
class cuda_btod_copy_h2d :
    public timings< cuda_btod_copy_h2d<N> >, public noncopyable {

private:
    block_tensor_rd_i<N, double> &m_bth; //!< Source block tensor on host

public:
    static const char k_clazz[]; //!< Class name

public:
    /** \brief Initializes the operation
        \param bth Block tensor on host
     **/
    cuda_btod_copy_h2d(block_tensor_rd_i<N, double> &bth) : m_bth(bth)  {

    }

    /** \brief Performs the copy operation
        \param btd Block tensor on CUDA device.
     **/
    void perform(cuda_block_tensor_wr_i<N, double> &btd);

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_BTOD_COPY_H2D_H
