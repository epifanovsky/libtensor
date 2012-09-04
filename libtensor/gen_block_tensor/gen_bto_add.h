#ifndef LIBTENSOR_GEN_BTO_ADD_H
#define LIBTENSOR_GEN_BTO_ADD_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/tensor_transf.h>
#include "assignment_schedule.h"
#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Linear combination of multiple block tensors
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    This block tensor operation performs the addition of block tensors:
    \f[ B = \mathcal{T}_1 A_1 + \mathcal{T}_2 A_2 + \cdots \f]

    The operation must have at least one operand provided at the time of
    construction. Other operands are added using add_op() and must agree in
    their dimensions and block structure.

    \sa gen_bto_copy

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits, typename Timed>
class gen_bto_add : public timings<Timed>, public noncopyable {
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

private:
    struct arg {
        gen_block_tensor_rd_i<N, bti_traits> &bta;
        tensor_transf<N, element_type> tra;
        arg(
            gen_block_tensor_rd_i<N, bti_traits> &bta_,
            const tensor_transf<N, element_type> &tra_) :
            bta(bta_), tra(tra_)
        { }
    };

private:
    std::list<arg> m_args; //!< List of arguments
    block_index_space<N> m_bisb; //!< Block index space of B
    symmetry<N, element_type> m_symb; //!< Symmetry of B
    assignment_schedule<N, element_type> m_schb; //!< Non-zero list of B

public:
    /** \brief Initializes the addition operation
        \param bta First block tensor in the linear combination.
        \param tra Transformation of the first tensor.
     **/
    gen_bto_add(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const tensor_transf<N, element_type> &tra);

    /** \brief Adds an operand (next tensor in the linear combination)
        \param bta Block tensor in the linear combination.
        \param tra Transformation of the tensor.
     **/
    void add_op(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const tensor_transf<N, element_type> &tra);

    /** \brief Returns the block index space of the result
     **/
    const block_index_space<N> &get_bis() const {

        return m_bisb;
    }

    /** \brief Returns the symmetry of the result
     **/
    const symmetry<N, element_type> &get_symmetry() const {

        return m_symb;
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    const assignment_schedule<N, element_type> &get_schedule() const {

        return m_schb;
    }

    /** \brief Turns on synchronization on all arguments
     **/
    void sync_on();

    /** \brief Turns off synchronization on all arguments
     **/
    void sync_off();

    /** \brief Writes the blocks of the result to an output stream
        \param out Output stream.
     **/
    void perform(gen_block_stream_i<N, bti_traits> &out);

    /** \brief Computes one block of the result
     **/
    void compute_block(
        bool zero,
        wr_block_type &blkb,
        const index<N> &ib,
        const tensor_transf<N, element_type> &trb,
        const element_type &c);

    /** \brief Same as compute_block(), except it doesn't run a timer
     **/
    void compute_block_untimed(
        bool zero,
        wr_block_type &blkb,
        const index<N> &ib,
        const tensor_transf<N, element_type> &trb,
        const element_type &c);

private:
    void add_operand(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const tensor_transf<N, element_type> &tra);

    void make_schedule();

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_ADD_H
