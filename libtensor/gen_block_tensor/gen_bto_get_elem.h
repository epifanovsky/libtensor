#ifndef LIBTENSOR_GEN_BTO_GET_ELEM_H
#define LIBTENSOR_GEN_BTO_GET_ELEM_H

#include <list>
#include <map>
#include <libtensor/defs.h>
#include <libtensor/core/dimensions.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/symmetry.h>
#include <libtensor/core/tensor_transf.h>
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Sets a single element of a block %tensor to a value
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.

    The operation gets one block %tensor element specified by a block
    %index and an %index within the block. The symmetry is preserved.
    If the affected block shares an orbit with other blocks, those will
    be affected accordingly.

    Normally for clarity reasons the block %index used with this operation
    should be canonical. If it is not, the canonical block is changed using
    %symmetry rules such that the specified element of the specified block
    is given the specified value.

    <b>Traits</b>

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template to_get_elem_type<N>::type -- Type of tensor operation
        to_get_elem
    - \c template to_get_type<N>::type -- Type of tensor operation to_get

    \ingroup libtensor_btod
 **/
template<size_t N, typename Traits>
class gen_bto_get_elem : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of write-only block
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

private:
    typedef std::list< tensor_transf<N, element_type> > transf_list_t;
    typedef std::map<size_t, transf_list_t> transf_map_t;

public:
    /** \brief Default constructor
     **/
    gen_bto_get_elem() { }

    /** \brief Performs the operation
        \param bt Block %tensor.
        \param bidx Block %index.
        \param idx Element %index within the block.
        \param d Element value.
     **/
    void perform(gen_block_tensor_i<N, bti_traits> &bt,
            const index<N> &bidx, const index<N> &idx,
            element_type &d);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_GET_ELEM_H
