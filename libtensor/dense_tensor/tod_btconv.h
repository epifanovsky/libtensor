#ifndef LIBTENSOR_TOD_BTCONV_H
#define LIBTENSOR_TOD_BTCONV_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include "dense_tensor_i.h"

namespace libtensor {

/** \brief Unfolds a block tensor into a simple tensor
    \tparam N Tensor order.

    \ingroup libtensor_tod
 **/
template<size_t N>
class tod_btconv : public timings< tod_btconv<N> >, public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    block_tensor_rd_i<N, double> &m_bt; //!< Source block %tensor

public:
    //!    \name Construction and destruction
    //@{

    tod_btconv(block_tensor_rd_i<N, double> &bt);
    ~tod_btconv();

    //@}

    //!    \name Tensor operation
    //@{

    void perform(dense_tensor_wr_i<N, double> &t);

    //@}

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_BTCONV_H
