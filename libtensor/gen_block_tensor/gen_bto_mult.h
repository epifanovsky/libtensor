#ifndef LIBTENSOR_GEN_BTO_MULT_H
#define LIBTENSOR_GEN_BTO_MULT_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/tensor_transf.h>
#include "assignment_schedule.h"
#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Elementwise multiplication of two block tensors
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    Computes the element-wise product of two block tensors.

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template temp_block_type<N>::type -- Type of temporary tensor block
    - \c template to_set_type<N>::type -- Type of tensor operation to_set
    - \c template to_mult_type<N>::type -- Type of tensor operation to_mult

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits, typename Timed>
class gen_bto_mult : public timings<Timed>, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block
    typedef typename
            bti_traits::template rd_block_type<N>::type rd_block_type;

    //! Type of write-only block
    typedef typename
            bti_traits::template wr_block_type<N>::type wr_block_type;

    //! Type of tensor transformation
    typedef tensor_transf<N, element_type> tensor_transf_type;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bta; //!< First argument
    gen_block_tensor_rd_i<N, bti_traits> &m_btb; //!< Second argument
    tensor_transf_type m_tra; //!< Tensor transformation of bta
    tensor_transf_type m_trb; //!< Tensor transformation of btb
    bool m_recip; //!< Reciprocal
    scalar_transf<element_type> m_trc; //!< Scaling coefficient

    block_index_space<N> m_bisc; //!< Block %index space of the result
    symmetry<N, element_type> m_symc; //!< Result symmetry
    assignment_schedule<N, element_type> m_sch; //!< Schedule

public:
    //! \name Constructors / destructor
    //@{

    /** \brief Constructor
        \param bta First argument
        \param tra Tensor transformation of first argument
        \param btb Second argument
        \param trb Tensor transformation of second argument
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param trc Scalar transformation of result
     **/
    gen_bto_mult(
            gen_block_tensor_rd_i<N, bti_traits> &bta,
            const tensor_transf_type &tra,
            gen_block_tensor_rd_i<N, bti_traits> &btb,
            const tensor_transf_type &trb,
            bool recip = false, const scalar_transf<element_type> &trc =
                    scalar_transf<element_type>());

    /** \brief Virtual destructor
     **/
    virtual ~gen_bto_mult() { }

    //@}

    /** \brief Returns the block index space of the result
     **/
    const block_index_space<N> &get_bis() const {
        return m_bisc;
    }

    /** \brief Returns the symmetry of the result
     **/
    const symmetry<N, element_type> &get_symmetry() const {
        return m_symc;
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    const assignment_schedule<N, element_type> &get_schedule() const {
        return m_sch;
    }

    /** \brief Computes and writes the blocks of the result to an output stream
        \param out Output stream.
     **/
    void perform(gen_block_stream_i<N, bti_traits> &out);

    /** \brief Computes one block of the result
        \param zero Zero target block first
        \param blkc Target block
        \param ic Index of target block
        \param trc Tensor transformation
     **/
    void compute_block(
        bool zero,
        const index<N> &ic,
        const tensor_transf_type &trc,
        wr_block_type &blkc);

    /** \brief Same as compute_block(), except it doesn't run a timer
     **/
    void compute_block_untimed(
        bool zero,
        const index<N> &ic,
        const tensor_transf_type &trc,
        wr_block_type &blkc);


private:
    void make_schedule();
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT_H
