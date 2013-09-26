#ifndef LIBTENSOR_CTF_BTOD_COLLECT_H
#define LIBTENSOR_CTF_BTOD_COLLECT_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include "ctf_block_tensor_i.h"

namespace libtensor {


/** \brief Copies a distributed CTF block tensor to a local block tensor
    \tparam N Tensor order.

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N>
class ctf_btod_collect : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    ctf_block_tensor_rd_i<N, double> &m_dbt; //!< Distributed CTF block tensor

public:
    /** \brief Initializes the operation
        \param dt Distributed CTF block tensor.
     **/
    ctf_btod_collect(ctf_block_tensor_rd_i<N, double> &dbt) :
        m_dbt(dbt)
    { }

    /** \brief Collects the data to a local block tensor
        \param bt Block tensor.
     **/
    void perform(block_tensor_wr_i<N, double> &bt);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_COLLECT_H

