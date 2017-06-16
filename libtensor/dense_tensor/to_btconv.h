#ifndef LIBTENSOR_TO_BTCONV_H
#define LIBTENSOR_TO_BTCONV_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include "dense_tensor_i.h"

namespace libtensor {

/** \brief Unfolds a block tensor into a simple tensor
    \tparam N Tensor order.

    \ingroup libtensor_tod
 **/
template<size_t N, typename T>
class to_btconv : public timings< to_btconv<N, T> >, public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    block_tensor_rd_i<N, T> &m_bt; //!< Source block %tensor

public:
    //!    \name Construction and destruction
    //@{

    to_btconv(block_tensor_rd_i<N, T> &bt);
    ~to_btconv();

    //@}

    //!    \name Tensor operation
    //@{

    void perform(dense_tensor_wr_i<N, T> &t);

    //@}

};

template<size_t N>
using tod_btconv = to_btconv<N, double>;

} // namespace libtensor

#endif // LIBTENSOR_TO_BTCONV_H
