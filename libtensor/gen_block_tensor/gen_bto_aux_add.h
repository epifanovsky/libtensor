#ifndef LIBTENSOR_GEN_BTO_AUX_ADD_H
#define LIBTENSOR_GEN_BTO_AUX_ADD_H

#include <map>
#include <vector>
#include <libutil/threads/mutex.h>
#include "addition_schedule.h"
#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"
#include "gen_block_tensor_ctrl.h"

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

    \sa gen_block_stream_i

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits>
class gen_bto_aux_add :
    public gen_block_stream_i<N, typename Traits::bti_traits> {

public:
    static const char *k_clazz; //!< Class name

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;

    typedef addition_schedule<N, Traits> schedule_type;
    typedef typename schedule_type::iterator schedule_iterator;
    typedef typename schedule_type::schedule_group schedule_group;
    typedef typename schedule_type::node schedule_node;
    typedef typename std::list<schedule_node>::const_iterator group_iterator;

private:
    block_index_space<N> m_bis; //!< Block index space
    dimensions<N> m_bidims; //!< Block index dimensions
    symmetry<N, element_type> m_syma; //!< Symmetry of source
    const addition_schedule<N, Traits> &m_asch; //!< Addition schedule
    gen_block_tensor_i<N, bti_traits> &m_btb; //!< Target block tensor
    scalar_transf<element_type> m_c; //!< Scaling coefficient
    gen_block_tensor_ctrl<N, bti_traits> m_cb; //!< Block tensor control
    bool m_open; //!< Open state
    std::map<size_t, const schedule_group*> m_schgrp; //!< Map A to sch grp
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
    gen_bto_aux_add(
        const symmetry<N, element_type> &syma,
        const addition_schedule<N, Traits> &asch,
        gen_block_tensor_i<N, bti_traits> &btb,
        const scalar_transf<element_type> &c);

    /** \brief Virtual destructor
     **/
    virtual ~gen_bto_aux_add();

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
        rd_block_type &blk,
        const tensor_transf<N, element_type> &tr);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_AUX_ADD_H
