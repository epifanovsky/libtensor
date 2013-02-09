#ifndef LIBTENSOR_GEN_BTO_SYMMETRIZE3_H
#define LIBTENSOR_GEN_BTO_SYMMETRIZE3_H

#include <map>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "additive_gen_bto.h"

namespace libtensor {


/** \brief (Anti-)symmetrizes the result of a block %tensor operation
        over three groups of indexes
    \tparam N Tensor order.

    The operation symmetrizes or anti-symmetrizes the result of another
    block %tensor operation over three indexes or groups of indexes.

    \f[
        b_{ijk} = P_{\pm} a_{ijk} = a_{ijk} \pm a_{jik} \pm a_{kji} \pm
            a_{ikj} + a_{jki} + a_{kij}
    \f]

    The constructor takes three different indexes to be symmetrized.

    \ingroup libtensor_btod
 **/
template<size_t N, typename Traits, typename Timed>
class gen_bto_symmetrize3 : public timings<Timed>, public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;

private:
    struct schrec {
        size_t ai;
        tensor_transf<N, element_type> tr;
        schrec() : ai(0) { }
        schrec(size_t ai_, const tensor_transf<N, element_type> &tr_) :
            ai(ai_), tr(tr_) { }
    };
    typedef std::multimap<size_t, schrec> sym_schedule_type;

private:
    additive_gen_bto<N, bti_traits> &m_op; //!< Symmetrized operation
    permutation<N> m_perm1; //!< First symmetrization permutation
    permutation<N> m_perm2; //!< Second symmetrization permutation
    bool m_symm; //!< Symmetrization/anti-symmetrization
    symmetry<N, element_type> m_sym; //!< Symmetry of the result
    assignment_schedule<N, element_type> m_sch; //!< Schedule

public:
    /** \brief Initializes the operation
        \param op Operation to be symmetrized.
        \param perm1 First unitary permutation.
        \param perm2 Second unitary permutation.
        \param symm True for symmetrization, false for anti-symmetrization.
     **/
    gen_bto_symmetrize3(
        additive_gen_bto<N, bti_traits> &op,
        const permutation<N> &perm1,
        const permutation<N> &perm2,
        bool symm);

    /** \brief Returns the block index space of the result
     **/
    const block_index_space<N> &get_bis() const {

        return m_op.get_bis();
    }

    /** \brief Returns the symmetry of the result
     **/
    const symmetry<N, element_type> &get_symmetry() const {

        return m_sym;
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    const assignment_schedule<N, element_type> &get_schedule() const {

        return m_sch;
    }

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
    void make_symmetry();
    void make_schedule();
    void make_schedule_blk(const abs_index<N> &ai,
        sym_schedule_type &sch) const;

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SYMMETRIZE3_H
