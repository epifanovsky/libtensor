#ifndef LIBTENSOR_GEN_BTO_CONTRACT3_H
#define LIBTENSOR_GEN_BTO_CONTRACT3_H

#include <vector>
#include <libtensor/timings.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include "impl/gen_bto_contract2_sym.h"
#include "assignment_schedule.h"
#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Contracts a train of three tensors
    \tparam N1 Order of first tensor less first contraction degree.
    \tparam N2 Order of second tensor less total contraction degree.
    \tparam N3 Order of third tensor less second contraction degree.
    \tparam K1 First contraction degree.
    \tparam K2 Second contraction degree.

    This algorithm computes the contraction of three linearly connected tensors.

    The contraction is performed as follows. The first tensor is contracted
    with the second tensor to form an intermediate, which is then contracted
    with the third tensor to yield the final result.

    The formation of the intermediate is done in batches:
    \f[
        ABC = A(B_1 + B_2 + \dots + B_n)C = \sum_{i=1}^n (AB_i)C \qquad
        B = \sum_{i=1}^n B_i
    \f]

    \ingroup libtensor_block_tensor
 **/
template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2,
    typename Traits, typename Timed>
class gen_bto_contract3 : public timings<Timed>, public noncopyable {
public:
    enum {
        NA = N1 + K1, //!< Rank of tensor A
        NB = N2 + K1 + K2, //!< Rank of tensor B
        NAB = N1 + N2 + K2, //!< Rank of intermediate tensor (A*B)
        NC = N3 + K2, //!< Rank of tensor C
        ND = N1 + N2 + N3 //!< Rank of result tensor (D)
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

    //! Type of read-only block of C
    typedef typename bti_traits::template rd_block_type<NC>::type
            rd_block_c_type;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<ND>::type wr_block_type;

private:
    contraction2<N1, N2 + K2, K1> m_contr1; //!< First contraction
    contraction2<N1 + N2, N3, K2> m_contr2; //!< Second contraction
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta; //!< First tensor (A)
    scalar_transf<element_type> m_ka; //!< Scalar transformation of A
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb; //!< Second tensor (B)
    scalar_transf<element_type> m_kb; //!< Scalar transformation of B
    gen_block_tensor_rd_i<NC, bti_traits> &m_btc; //!< Third tensor (C)
    scalar_transf<element_type> m_kc; //!< Scalar transformation of C
    scalar_transf<element_type> m_kd; //!< Scalar transformation of result (D)

    gen_bto_contract2_sym<N1, N2 + K2, K1, Traits> m_symab; //!< Symmetry of intermediate (AB)
    gen_bto_contract2_sym<N1 + N2, N3, K2, Traits> m_symd; //!< Symmetry of result (D)

    assignment_schedule<NAB, element_type> m_schab; //!< Schedule for AB
    assignment_schedule<ND, element_type> m_schd; //!< Schedule for result (D)

public:
    /** \brief Initializes the contraction
        \param contr1 First contraction (A with B).
        \param contr2 Second contraction (AB with C).
        \param bta First tensor argument (A).
        \param btb Second tensor argument (B).
        \param btc Third tensor argument (C).
        \param batch_size Batching size
     **/
    gen_bto_contract3(
        const contraction2<N1, N2 + K2, K1> &contr1,
        const contraction2<N1 + N2, N3, K2> &contr2,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        const scalar_transf<element_type> &ka,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const scalar_transf<element_type> &kb,
        gen_block_tensor_rd_i<NC, bti_traits> &btc,
        const scalar_transf<element_type> &kc,
        const scalar_transf<element_type> &kd);

    /** \brief Returns the block index space of the result
     **/
    const block_index_space<N1 + N2 + N3> &get_bis() const {

        return m_symd.get_bis();
    }

    /** \brief Returns the symmetry of the result
     **/
    const symmetry<N1 + N2 + N3, element_type> &get_symmetry() const {

        return m_symd.get_symmetry();
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    const assignment_schedule<N1 + N2 + N3, element_type> &get_schedule() const {

        return m_schd;
    }

    /** \brief Computes the contraction
     **/
    void perform(gen_block_stream_i<ND, bti_traits> &out);

private:
    void compute_batch_ab(
            const contraction2<N1, N2 + K2, K1> &contr,
            const orbit_list<NA, element_type> &ola,
            const permutation<NA> &perma,
            const symmetry<NA, element_type> &syma, size_t batchsza,
            const orbit_list<NB, element_type> &olb,
            const permutation<NB> &permb,
            const symmetry<NB, element_type> &symb, size_t batchszb,
            const block_index_space<NAB> &bisab,
            const std::vector<size_t> &blst,
            gen_block_stream_i<NAB, bti_traits> &out);

    static const symmetry<N3 + K2, element_type> &retrieve_symmetry(
            gen_block_tensor_base_i<NC, bti_traits> &btc) {

        gen_block_tensor_base_ctrl<NC, bti_traits> cc(btc);
        return cc.req_const_symmetry();
    }

    void make_schedule();
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT3_H

