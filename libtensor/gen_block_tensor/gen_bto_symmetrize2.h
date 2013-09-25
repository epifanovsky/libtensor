#ifndef LIBTENSOR_GEN_BTO_SYMMETRIZE2_H
#define LIBTENSOR_GEN_BTO_SYMMETRIZE2_H

#include <map>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "additive_gen_bto.h"

namespace libtensor {


/** \brief Symmetrizes the result of another block tensor operation
    \tparam N Tensor order.

    \sa gen_bto_symmetrize3, gen_bto_symmetrize4

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits, typename Timed>
class gen_bto_symmetrize2 : public timings<Timed>, public noncopyable {
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
    bool m_symm; //!< Symmetrization sign
    block_index_space<N> m_bis; //!< Block %index space of the result
    symmetry<N, element_type> m_sym; //!< Symmetry of the result
    assignment_schedule<N, element_type> m_sch; //!< Schedule
    sym_schedule_type m_sym_sch; //!< Symmetrization schedule

public:
    /** \brief Initializes the operation using a unitary permutation (P = P^-1)
        \param op Symmetrized operation.
        \param perm Unitary permutation.
        \param symm True for symmetric, false for anti-symmetric.
     **/
    gen_bto_symmetrize2(
        additive_gen_bto<N, bti_traits> &op,
        const permutation<N> &perm,
        bool symm);

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
    /** \brief Constructs the symmetry of the result
     **/
    void make_symmetry();

    /** \brief Constructs the assignment schedule of the operation
     **/
    void make_schedule();

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SYMMETRIZE2_H
