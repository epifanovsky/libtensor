#ifndef LIBTENSOR_BTOD_SET_ELEM_H
#define LIBTENSOR_BTOD_SET_ELEM_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_set_elem.h>

namespace libtensor {


/** \brief Sets a single element of a block %tensor to a value
    \tparam N Tensor order.

    The operation sets one block %tensor element specified by a block
    %index and an %index within the block. The symmetry is preserved.
    If the affected block shares an orbit with other blocks, those will
    be affected accordingly.

    Normally for clarity reasons the block %index used with this operation
    should be canonical. If it is not, the canonical block is changed using
    %symmetry rules such that the specified element of the specified block
    is given the specified value.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_set_elem : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_set_elem<N, btod_traits> m_gbto;

public:
    /** \brief Default constructor
     **/
    btod_set_elem() { }

    /** \brief Performs the operation
        \param bt Block %tensor.
        \param bidx Block %index.
        \param idx Element %index within the block.
        \param d Element value.
     **/
    void perform(block_tensor_i<N, double> &bt, const index<N> &bidx,
        const index<N> &idx, double d);
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_ELEM_H
