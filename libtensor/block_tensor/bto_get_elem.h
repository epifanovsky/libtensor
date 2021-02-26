#ifndef LIBTENSOR_BTO_SET_ELEM_H
#define LIBTENSOR_BTO_SET_ELEM_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/bto_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_get_elem.h>

namespace libtensor {


/** \brief Sets a single element of a block %tensor to a value
    \tparam N Tensor order.

    The operation gets one block %tensor element specified by a block
    %index and an %index within the block. The symmetry is preserved.
    If the affected block shares an orbit with other blocks, those will
    be affected accordingly.

    Normally for clarity reasons the block %index used with this operation
    should be canonical. If it is not, the canonical block is changed using
    %symmetry rules such that the specified element of the specified block
    is given the specified value.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, typename T>
class bto_get_elem : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_get_elem<N, bto_traits<T> > m_gbto;

public:
    /** \brief Default constructor
     **/
    bto_get_elem() { }

    /** \brief Performs the operation
        \param bt Block %tensor.
        \param bidx Block %index.
        \param idx Element %index within the block.
        \param d Element value.
     **/
    void perform(block_tensor_i<N, T> &bt, const index<N> &bidx,
        const index<N> &idx, T& d);
};


} // namespace libtensor

#endif // LIBTENSOR_BTO_SET_ELEM_H
