#ifndef LIBTENSOR_CTF_BLOCK_TENSOR_H
#define LIBTENSOR_CTF_BLOCK_TENSOR_H

#include <libtensor/core/immutable.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/gen_block_tensor.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include "ctf_block_tensor_i.h"
#include "ctf_block_tensor_traits.h"

namespace libtensor {


/** \brief Distributed CTF block tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N, typename T>
class ctf_block_tensor :
    virtual public gen_block_tensor< N, ctf_block_tensor_traits<T> >,
    virtual public ctf_block_tensor_i<N, T>,
    public noncopyable {

public:
    ctf_block_tensor(const block_index_space<N> &bis) :
        gen_block_tensor< N, ctf_block_tensor_traits<T> >(bis)
    { }

    virtual ~ctf_block_tensor();

};


template<size_t N, typename T>
ctf_block_tensor<N, T>::~ctf_block_tensor() {

}


} // namespace libtensor

#endif // LIBTENSOR_CTF_BLOCK_TENSOR_H
