#ifndef LIBTENSOR_GEN_BTO_EWMULT2_H
#define LIBTENSOR_GEN_BTO_EWMULT2_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/tensor_transf.h>
#include "assignment_schedule.h"
#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Generalized element-wise (Hadamard) product of two block tensors
    \tparam N Order of first argument (A) less the number of shared indexes.
    \tparam M Order of second argument (B) less the number of shared
        indexes.
    \tparam K Number of shared indexes.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    This operation computes the element-wise product of two block tensor.
    Refer to tod_ewmult2<N, M, K> for setup info.

    Both arguments and result must agree on their block index spaces,
    otherwise the constructor and perform() will raise
    bad_block_index_space.

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template temp_block_type<N>::type -- Type of temporary tensor block
    - \c template to_ewmult2_type<N, M, K>::type -- Type of tensor operation
        to_ewmult2
    - \c template to_set_type<N>::type -- Type of tensor operation to_set

    \sa tod_ewmult2, gen_bto_contract2

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
class gen_bto_ewmult2 : public timings<Timed>, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M + K //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block of A
    typedef typename bti_traits::template rd_block_type<NA>::type
            rd_block_a_type;

    //! Type of read-only block of B
    typedef typename bti_traits::template rd_block_type<NB>::type
            rd_block_b_type;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<NC>::type wr_block_type;

    //! Type of tensor transformation of result
    typedef tensor_transf<NC, element_type> tensor_transf_type;

private:
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta; //!< First argument (A)
    tensor_transf<NA, element_type> m_tra; //!< Tensor transformaion of A
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb; //!< Second argument (B)
    tensor_transf<NB, element_type> m_trb; //!< Tensor transformation of B
    tensor_transf_type m_trc; //!< Tensor transformation of result (C)
    block_index_space<NC> m_bisc; //!< Block index space of result
    symmetry<NC, element_type> m_symc; //!< Symmetry of result
    assignment_schedule<NC, element_type> m_sch; //!< Assignment schedule

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param bta First argument (A).
        \param tra Tensor transformation of A.
        \param btb Second argument (B).
        \param trb Tensor transformation of B.
        \param trc Tensor transformation of result (C).
     **/
    gen_bto_ewmult2(
            gen_block_tensor_rd_i<NA, bti_traits> &bta,
            const tensor_transf<NA, element_type> &tra,
            gen_block_tensor_rd_i<NB, bti_traits> &btb,
            const tensor_transf<NB, element_type> &trb,
            const tensor_transf_type &trc = tensor_transf_type());

    //@}


    /** \brief Returns the block index space of the result
     **/
    const block_index_space<NC> &get_bis() const {
        return m_bisc;
    }

    /** \brief Returns the symmetry of the result
     **/
    const symmetry<NC, element_type> &get_symmetry() const {
        return m_symc;
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    const assignment_schedule<N + M + K, element_type> &get_schedule() const {
        return m_sch;
    }

    /** \brief Writes the blocks of the result to an output stream
        \param out Output stream.
     **/
   void perform(gen_block_stream_i<NC, bti_traits> &out);

   /** \brief Computes one block of the result and writes it to a tensor
       \param zero Whether to zero out the contents of output before adding
           the contraction
       \param idxc Index of the computed block, must be a canonical block in
           the output tensor's symmetry
       \param trc Transformation to be applied to the computed block.
       \param[out] blkc Output tensor.
    **/
   void compute_block(
            bool zero,
            const index<NC> &idxc,
            const tensor_transf<NC, element_type> &trc,
            wr_block_type &blkc);

    /** \brief Same as compute block but untimed....
     **/
   void compute_block_untimed(
            bool zero,
            const index<NC> &idxc,
            const tensor_transf<NC, element_type> &trc,
            wr_block_type &blkc);

private:
    /** \brief Computes the block index space of the result block tensor
     **/
    static block_index_space<N + M + K> make_bisc(
        const block_index_space<NA> &bisa,
        const permutation<NA> &perma,
        const block_index_space<NB> &bisb,
        const permutation<NB> &permb,
        const permutation<NC> &permc);

    /** \brief Computes the symmetry of the result block tensor
     **/
    void make_symc();

    /** \brief Prepares the assignment schedule
     **/
    void make_schedule();
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_EWMULT2_H
