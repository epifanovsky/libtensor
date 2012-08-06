#ifndef LIBTENSOR_DIRECT_BTO_H
#define LIBTENSOR_DIRECT_BTO_H

#include <libtensor/defs.h>
#include <libtensor/exception.h>
#include <libtensor/core/block_index_space.h>
#include <libtensor/core/symmetry.h>
#include "assignment_schedule.h"
#include "bto_stream_i.h"

namespace libtensor {


/** \brief Underlying operation for direct block tensors

    Block %tensor operations that serve as underlying operations for
    direct block tensors take an arbitrary number of arguments, but result
    in one block %tensor.

    \ingroup libtensor_core
 **/
template<size_t N, typename Traits>
class direct_bto {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_t;

    //! Type of block tensors
    typedef typename Traits::template block_tensor_type<N>::type
            block_tensor_t;

    //! Type of blocks
    typedef typename Traits::template block_type<N>::type block_t;

public:
    virtual ~direct_bto() { }

    /** \brief Returns the block %index space of the result
     **/
    virtual const block_index_space<N> &get_bis() const = 0;

    /** \brief Returns the symmetry of the result
     **/
    virtual const symmetry<N, element_t> &get_symmetry() const = 0;

    /** \brief Invoked to execute the operation
     **/
    virtual void perform(block_tensor_t &bt) = 0;

    /** \brief Runs the operation and writes the result into the output stream
     **/
    virtual void perform(bto_stream_i<N, Traits> &out) = 0;

    /** \brief Returns the assignment schedule -- the preferred order
            of computing blocks
     **/
    virtual const assignment_schedule<N, element_t> &get_schedule() const = 0;

    /** \brief Computes a single block of the result
     **/
    virtual void compute_block(block_t &blk, const index<N> &i) = 0;

    /** \brief Enables the synchronization of arguments
     **/
    virtual void sync_on() = 0;

    /** \brief Disables the synchronization of arguments
     **/
    virtual void sync_off() = 0;

};


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BTO_H

