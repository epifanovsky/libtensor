#ifndef LIBTENSOR_GEN_BTO_DIRSUM_H
#define LIBTENSOR_GEN_BTO_DIRSUM_H

#include <map>
#include <libtensor/timings.h>
#include <libtensor/core/index.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/tensor_transf.h>
#include "assignment_schedule.h"
#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"
#include "impl/gen_bto_dirsum_sym.h"


namespace libtensor {


/** \brief Computes the direct sum of two block tensors
    \tparam N Order of the first %tensor.
    \tparam M Order of the second %tensor.

    Given two tensors \f$ a_{ij\cdots} \f$ and \f$ b_{mn\cdots} \f$,
    the operation computes
    \f$ c_{ij\cdots mn\cdots} = k_a a_{ij\cdots} + k_b b_{mn\cdots} \f$.

    The order of %tensor indexes in the result can be specified using
    a permutation.

    \ingroup libtensor_btod
 **/
template<size_t N, size_t M, typename Traits, typename Timed>
class gen_bto_dirsum : public timings<Timed>, public noncopyable {
public:
    enum {
        NA = N, //!< Order of first argument (A)
        NB = M, //!< Order of second argument (B)
        NC = N + M //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block in A
    typedef typename bti_traits::template rd_block_type<NA>::type
            rd_block_a_type;

    //! Type of read-only block in B
    typedef typename bti_traits::template rd_block_type<NB>::type
            rd_block_b_type;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<NC>::type wr_block_type;

    //! Type of scalar transformation
    typedef scalar_transf<element_type> scalar_transf_type;

    //! Type of tensor transformation of result
    typedef tensor_transf<NC, element_type> tensor_transf_type;

private:
    struct schrec {
        size_t absidxa, absidxb;
        bool zeroa, zerob;
        scalar_transf<element_type> ka, kb;
        tensor_transf_type trc;
    };
    typedef std::map<size_t, schrec> schedule_t;

private:
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta; //!< First %tensor (A)
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb; //!< Second %tensor (B)
    scalar_transf_type m_ka; //!< Coefficient A
    scalar_transf_type m_kb; //!< Coefficient B
    tensor_transf_type m_trc; //!< Tensor transformation of the result
    gen_bto_dirsum_sym<NA, NB, Traits> m_symc; //!< Symmetry of result
    dimensions<NA> m_bidimsa; //!< Block %index dims of A
    dimensions<NB> m_bidimsb; //!< Block %index dims of B
    dimensions<NC> m_bidimsc; //!< Block %index dims of the result
    schedule_t m_op_sch; //!< Direct sum schedule
    assignment_schedule<NC, element_type> m_sch; //!< Assignment schedule

public:
    /** \brief Initializes the operation
        \param bta First input block %tensor
        \param ka Scalar transformation applied to bta
        \param btb Second input block %tensor
        \param kb Scalar transformation applied to btb
        \param trc Tensor transformation of result
     **/
    gen_bto_dirsum(
            gen_block_tensor_rd_i<NA, bti_traits> &bta,
            const scalar_transf_type &ka,
            gen_block_tensor_rd_i<NB, bti_traits> &btb,
            const scalar_transf_type &kb,
            const tensor_transf_type &trc = tensor_transf_type());

    /** \brief Virtual destructor
     **/
    virtual ~gen_bto_dirsum() { }

    /** \brief Returns the block index space of the result
     **/
    const block_index_space<NC> &get_bis() const {

        return m_symc.get_bis();
    }

    /** \brief Returns the symmetry of the result
     **/
    const symmetry<NC, element_type> &get_symmetry() const {

        return m_symc.get_symmetry();
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    const assignment_schedule<NC, element_type> &get_schedule() const {

        return m_sch;
    }

    /** \brief Computes and writes the blocks of the result to an output stream
        \param out Output stream.
     **/
    void perform(gen_block_stream_i<NC, bti_traits> &out);

    /** \brief Computes one block of the result and writes it to a tensor
        \param zero Whether to zero out the contents of output before adding
        \param idxc Index of the computed block, must be a canonical block in
            the output tensor's symmetry
        \param trc Transformation to be applied to the computed block.
        \param[out] blkc Output tensor.
     */
    virtual void compute_block(
            bool zero,
            const index<NC> &idxc,
            const tensor_transf<NC, element_type> &trc,
            wr_block_type &blkc);

    /** \brief Identical to compute_block, but untimed
     */
    virtual void compute_block_untimed(
            bool zero,
            const index<NC> &idxc,
            const tensor_transf<NC, element_type> &trc,
            wr_block_type &blkc);

private:
    void make_schedule();
    void make_schedule(
            const orbit<NA, element_type> &oa, bool zeroa,
            const orbit<NB, element_type> &ob, bool zerob,
            const orbit_list<NC, element_type> &olc);
};


} // namespace libtensor

#endif // LIBTENOSR_GEN_BTO_DIRSUM_H
