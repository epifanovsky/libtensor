#ifndef LIBTENSOR_GEN_BTO_RANDOM_H
#define LIBTENSOR_GEN_BTO_RANDOM_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Puts random data into block tensor
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    Fills a block %tensor with random data without affecting its
    symmetry.

    <b>Traits</b>

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template temp_block_type<N>::type -- Type of temporary tensor block
    - \c template to_add_type<N>::type -- Type of tensor operation to_copy
    - \c template to_copy_type<N>::type -- Type of tensor operation to_add
    - \c template to_random_type<N>::type -- Type of tensor operation to_random

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits, typename Timed>
class gen_bto_random : public timings<Timed>, public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

public:
    /** \brief Fills a block tensor with random values preserving symmetry
        \param bt Block tensor.
     **/
    void perform(gen_block_tensor_wr_i<N, bti_traits> &bt);

    /** \brief Fills one block of a block tensor with random values preserving
            symmetry
        \param bt Block tensor.
        \param idx Block index in the block tensor.
     **/
    void perform(gen_block_tensor_wr_i<N, bti_traits> &bt, const index<N> &idx);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_RANDOM_H
