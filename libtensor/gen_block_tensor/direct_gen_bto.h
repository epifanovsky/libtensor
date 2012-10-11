#ifndef LIBTENSOR_DIRECT_GEN_BTO_H
#define LIBTENSOR_DIRECT_GEN_BTO_H

#include <libtensor/core/block_index_space.h>
#include <libtensor/core/symmetry.h>
#include "assignment_schedule.h"
#include "gen_block_tensor_i.h"
#include "gen_block_stream_i.h"

namespace libtensor {


/** \brief Underlying operation for direct block tensors

    Block %tensor operations that serve as underlying operations for
    direct block tensors take an arbitrary number of arguments, but result
    in one block %tensor.

    \ingroup libtensor_core
 **/
template<size_t N, typename BtiTraits>
class direct_gen_bto {
public:
    //! Type of tensor elements
    typedef typename BtiTraits::element_type element_type;

    //! Type of blocks
    typedef typename BtiTraits::template wr_block_type<N>::type wr_block_type;

public:
    virtual ~direct_gen_bto() { }

    /** \brief Returns the block %index space of the result
     **/
    virtual const block_index_space<N> &get_bis() const = 0;

    /** \brief Returns the symmetry of the result
     **/
    virtual const symmetry<N, element_type> &get_symmetry() const = 0;

    /** \brief Returns the assignment schedule -- the preferred order
            of computing blocks
     **/
    virtual const assignment_schedule<N, element_type> &get_schedule() const = 0;

    /** \brief Runs the operation and writes the result into the output stream
     **/
    virtual void perform(gen_block_stream_i<N, BtiTraits> &out) = 0;

    /** \brief Computes one block
     **/
    virtual void compute_block(wr_block_type &blk, const index<N> &idx) = 0;

};


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_GEN_BTO_H

