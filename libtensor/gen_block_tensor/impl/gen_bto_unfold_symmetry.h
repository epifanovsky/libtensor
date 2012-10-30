#ifndef LIBTENSOR_GEN_BTO_UNFOLD_SYMMETRY_H
#define LIBTENSOR_GEN_BTO_UNFOLD_SYMMETRY_H

#include <libtensor/core/noncopyable.h>
#include "../gen_block_tensor_i.h"

namespace libtensor {


/** \brief Unfolds symmetry in a block tensor
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.

    This algorithm takes a block tensor with symmetry and turns it into a
    block tensor without symmetry by computing all non-canonical blocks from
    canonical blocks.

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits>
class gen_bto_unfold_symmetry : public noncopyable {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

public:
    void perform(gen_block_tensor_i<N, bti_traits> &bt);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_UNFOLD_SYMMETRY_H
