#ifndef LIBTENSOR_BTO_AUX_ADD_H
#define LIBTENSOR_BTO_AUX_ADD_H

#include <map>
#include <set>
#include <libutil/threads/mutex.h>
#include <libtensor/gen_block_tensor/addition_schedule.h>
#include "bto_stream_i.h"

namespace libtensor {


/** \brief Adds blocks to a block tensor (auxiliary operation)
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits structure.

    This auxiliary block tensor operation accepts blocks and performs the
    addition to a given block tensor. Upon calling open(), the symmetry
    of the target tensor is lowered if necessary, however the canonical blocks
    are not replicated into new orbits. The replication is done as blocks
    arrive through put(). Calling close() finalizes the replication. As such,
    the target block tensor remains incomplete (algebraically incorrect)
    between the calls to open() and close(). The blocks pushed through put()
    must be canonical in the source symmetry.

    \sa bto_stream_i

    \ingroup libtensor_block_tensor_bto
 **/
template<size_t N, typename Traits>
class bto_aux_add : public bto_stream_i<N, Traits> {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::template block_type<N>::type block_type;
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_type;
    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;
    typedef symmetry<N, element_type> symmetry_type;
    typedef tensor_transf<N, element_type> tensor_transf_type;

private:
    block_index_space<N> m_bis; //!< Block index space
    dimensions<N> m_bidims; //!< Block index dimensions
    symmetry_type m_syma; //!< Symmetry of source
    const addition_schedule<N, Traits> &m_asch; //!< Addition schedule
    block_tensor_type &m_btb; //!< Target block tensor
    element_type m_c; //!< Scaling coefficient
    block_tensor_ctrl_type m_cb; //!< Control on target block tensor
    bool m_open; //!< Open state
    size_t m_grpcount; //!< Group count
    std::map<size_t, size_t> m_grpmap; //!< Maps index in A to group number
    libutil::mutex m_mtx; //!< Global mutex
    std::vector<libutil::mutex*> m_grpmtx; //!< Per-group mutexes

public:
    /** \brief Constructs the operation
        \brief syma Symmetry of the source block tensor.
        \brief asch Addition schedule.
        \brief btb Target block tensor.
        \brief c Scaling coefficient for addition.
     **/
    bto_aux_add(
        const symmetry_type &syma,
        const addition_schedule<N, Traits> &asch,
        block_tensor_type &btb,
        const element_type &c);

    /** \brief Virtual destructor
     **/
    virtual ~bto_aux_add();

    /** \brief Implements bto_stream_i::open(). Prepares the copy operation
     **/
    virtual void open();

    /** \brief Implements bto_stream_i::close()
     **/
    virtual void close();

    /** \brief Implements bto_stream_i::put(). Saves a block in the output
            block tensor
     **/
    virtual void put(
        const index<N> &idx,
        block_type &blk,
        const tensor_transf_type &tr);

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_AUX_ADD_H
