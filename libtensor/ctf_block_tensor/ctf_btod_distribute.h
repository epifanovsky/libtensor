#ifndef LIBTENSOR_CTF_BTOD_DISTRIBUTE_H
#define LIBTENSOR_CTF_BTOD_DISTRIBUTE_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include "ctf_block_tensor_i.h"

namespace libtensor {


/** \brief Converts a local block tensor into a distributed CTF block tensor
    \tparam N Tensor order.

    \sa ctf_btod_collect

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N>
class ctf_btod_distribute : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    block_tensor_rd_i<N, double> &m_bt; //!< Block tensor

public:
    /** \brief Initializes the operation
        \param bt Local block tensor.
     **/
    ctf_btod_distribute(block_tensor_rd_i<N, double> &bt) :
        m_bt(bt)
    { }

    /** \brief Distributes the data to a CTF block tensor
        \param dbt Distributed block tensor.
     **/
    void perform(ctf_block_tensor_i<N, double> &dbt);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_DISTRIBUTE_H

