#ifndef LIBTENSOR_GEN_BTO_EXTRACT_H
#define LIBTENSOR_GEN_BTO_EXTRACT_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "assignment_schedule.h"
#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Extracts a tensor with smaller dimension from the %tensor
    \tparam N Tensor order.
    \tparam M Number of fixed dimensions.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    Extracts a general block tensor with dimension N - M from the given
    block tensor.

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template temp_block_type<N>::type -- Type of temporary tensor block
    - \c template to_set_type<N>::type -- Type of tensor operation to_set
    - \c template to_extract_type<N, M>::type -- Type of tensor operation
        to_extract


    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, typename Traits, typename Timed>
class gen_bto_extract : public timings<Timed>, public noncopyable {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    //! Type of read-only block
    typedef typename bti_traits::template wr_block_type<N - M>::type
            wr_block_type;

    //! Type of tensor transformation of result
    typedef tensor_transf<N - M, element_type> tensor_transf_type;

public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        NA = N, //!< Order of the argument
        NB = N - M //!< Order of the result
    };

private:
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta; //!< Input block %tensor
    mask<NA> m_msk;//!< Mask for extraction
    tensor_transf_type m_tr; //!< Transformation of the result
    block_index_space<NB> m_bis; //!< Block %index space of the result
    index<NA> m_idxbl;//!< Index for extraction of the block
    index<NA> m_idxibl;//!< Index for extraction inside the block
    symmetry<NB, element_type> m_sym; //!< Symmetry of the result
    assignment_schedule<NB, element_type> m_sch; //!< Assignment schedule

public:
    gen_bto_extract(
            gen_block_tensor_rd_i<NA, bti_traits> &bta,
            const mask<NA> &m, const index<NA> &idxbl,
            const index<NA> &idxibl, const tensor_transf_type &tr);

    const block_index_space<NB> &get_bis() const {
        return m_bis;
    }

    const symmetry<NB, element_type> &get_symmetry() const {
        return m_sym;
    }

    const assignment_schedule<NB, element_type> &get_schedule() const {
        return m_sch;
    }

    /** \brief Computes and writes the blocks of the result to an output stream
        \param out Output stream.
     **/
    void perform(gen_block_stream_i<NB, bti_traits> &out);

    /** \brief Computes one block of the result and writes it to a tensor
        \param zero Whether to zero out the contents of output before adding
        \param idxb Index of the computed block, must be a canonical block in
            the output tensor's symmetry
        \param trb Transformation to be applied to the computed block.
        \param[out] blkb Output tensor.
     */
    virtual void compute_block(
            bool zero,
            const index<NB> &idxb,
            const tensor_transf<NB, element_type> &trb,
            wr_block_type &blkb);

    /** \brief Identical to compute_block, but untimed
     */
    virtual void compute_block_untimed(
            bool zero,
            const index<NB> &idxb,
            const tensor_transf<NB, element_type> &trb,
            wr_block_type &blkb);

private:
    /** \brief Forms the block %index space of the output or throws an
            exception if the input is incorrect
     **/
    static block_index_space<N - M> mk_bis(const block_index_space<NA> &bis,
        const mask<NA> &msk, const permutation<NB> &perm);

    /** \brief Sets up the assignment schedule for the operation.
     **/
    void make_schedule();
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_EXTRACT_H
