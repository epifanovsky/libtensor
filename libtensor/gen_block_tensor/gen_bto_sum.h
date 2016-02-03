#ifndef LIBTENSOR_GEN_BTO_SUM_H
#define LIBTENSOR_GEN_BTO_SUM_H

#include <list>
#include <utility>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/tensor_transf.h>
#include "additive_gen_bto.h"
#include "assignment_schedule.h"
#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Adds together the results of a sequence of block tensor operations
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.

    This operation runs a sequence of block tensor operations and
    accumulates their results with given coefficients. All of the operations
    in the sequence shall derive from additive_gen_bto.

    The sequence must contain at least one operation, which is called the
    base operation.

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits>
class gen_bto_sum : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;

private:
    typedef std::pair< additive_gen_bto<N, bti_traits>*,
        scalar_transf<element_type> > op_type;

private:
    std::list<op_type> m_ops; //!< List of operations
    block_index_space<N> m_bis; //!< Block index space
    dimensions<N> m_bidims; //!< Block index dims
    symmetry<N, element_type> m_sym; //!< Symmetry of operation
    mutable bool m_dirty_sch; //!< Whether the assignment schedule is dirty
    mutable assignment_schedule<N, element_type> *m_sch; //!< Assignment sched

public:
    /** \brief Initializes the base operation
        \param op Operation.
        \param c Coefficient.
     **/
    gen_bto_sum(
        additive_gen_bto<N, bti_traits> &op,
        const scalar_transf<element_type> &c);

    /** \brief Destructor
     **/
    ~gen_bto_sum();

    /** \brief Returns the block index space of the result
     **/
    const block_index_space<N> &get_bis() const {

        return m_bis;
    }

    /** \brief Returns the symmetry of the result
     **/
    const symmetry<N, element_type> &get_symmetry() const {

        return m_sym;
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    const assignment_schedule<N, element_type> &get_schedule() const {

        if(m_sch == 0 || m_dirty_sch) make_schedule();
        return *m_sch;
    }

    /** \brief Adds an operation to the sequence
        \param op Operation.
        \param c Coefficient.
     **/
    void add_op(
        additive_gen_bto<N, bti_traits> &op,
        const scalar_transf<element_type> &c);

    /** \brief Writes the blocks of the result to an output stream
        \param out Output stream.
     **/
    void perform(
        gen_block_stream_i<N, bti_traits> &out);

    /** \brief Computes one block of the result
     **/
    void compute_block(
        bool zero,
        const index<N> &ib,
        const tensor_transf<N, element_type> &trb,
        wr_block_type &blkb);

private:
    void make_schedule() const;

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SUM_H

