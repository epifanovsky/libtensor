#ifndef LIBTENSOR_GEN_BTO_DIRSUM_SYM_H
#define LIBTENSOR_GEN_BTO_DIRSUM_SYM_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf.h>
#include <libtensor/core/symmetry.h>
#include "../gen_block_tensor_i.h"
#include "gen_bto_contract2_bis.h"

namespace libtensor {


/** \brief Computes the %symmetry of the result of a direct sum

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, typename Traits>
class gen_bto_dirsum_sym : public noncopyable {
public:
    enum {
        NC = N + M //!< Rank of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_bto_contract2_bis<N, M, 0> m_bisc; //!< Block index space of result
    symmetry<NC, element_type> m_symc; //!< Symmetry of result

public:
    /** \brief Computes the symmetry of a direct sum operation
        \param bta Block tensor A
        \param ka Scalar transformation of A
        \param btb Block tensor B
        \param kb Scalar transformation of B
        \param permc Permutation of result
     **/
    gen_bto_dirsum_sym(
            gen_block_tensor_rd_i<N, bti_traits> &bta,
            const scalar_transf<element_type> &ka,
            gen_block_tensor_rd_i<M, bti_traits> &btb,
            const scalar_transf<element_type> &kb,
            const permutation<NC> &permc);

    /** \brief Computes the symmetry of a direct sum operation
        \param syma Symmetry of A
        \param ka Scalar transformation of A
        \param symb Symmetry of B
        \param kb Scalar transformation of B
        \param permc Permutation of result
     **/
    gen_bto_dirsum_sym(
            const symmetry<N, element_type> &syma,
            const scalar_transf<element_type> &ka,
            const symmetry<M, element_type> &symb,
            const scalar_transf<element_type> &kb,
            const permutation<NC> &permc);


    const block_index_space<NC> &get_bis() const {
        return m_bisc.get_bis();
    }

    const symmetry<NC, element_type> &get_symmetry() const {
        return m_symc;
    }
};

/** \brief Computes the %symmetry of the result of a direct sum
           (specialized for same-order A and B)

     \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits>
class gen_bto_dirsum_sym<N, N, Traits> : public noncopyable {
public:
    enum {
        NC = N + N //!< Rank of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_bto_contract2_bis<N, N, 0> m_bisc; //!< Block index space of result
    symmetry<NC, element_type> m_symc; //!< Symmetry of result

public:
    /** \brief Computes the symmetry of a direct sum operation
        \param bta Block tensor A
        \param ka Scalar transformation of A
        \param btb Block tensor B
        \param ka Scalar transformation of B
        \param permc Permutation of result
     **/
    gen_bto_dirsum_sym(
            gen_block_tensor_rd_i<N, bti_traits> &bta,
            const scalar_transf<element_type> &ka,
            gen_block_tensor_rd_i<N, bti_traits> &btb,
            const scalar_transf<element_type> &kb,
            const permutation<NC> &permc);

    /** \brief Computes the symmetry of a direct sum operation
        \param bta Block index space of A
        \param syma Symmetry of A
        \param ka Scalar transformation of A
        \param btb Block index space of B
        \param symb Symmetry of B
        \param kb Scalar transformation of B
        \param permc Permutation of result
        \param self True, if A and B are identical.
     **/
    gen_bto_dirsum_sym(
            const symmetry<N, element_type> &syma,
            const scalar_transf<element_type> &ka,
            const symmetry<N, element_type> &symb,
            const scalar_transf<element_type> &kb,
            const permutation<NC> &permc, bool self);

    const block_index_space<NC> &get_bis() const {
        return m_bisc.get_bis();
    }

    const symmetry<NC, element_type> &get_symmetry() const {
        return m_symc;
    }

private:
    void make_symmetry(
            const symmetry<N, element_type> &syma,
            const scalar_transf<element_type> &ka,
            const symmetry<N, element_type> &symb,
            const scalar_transf<element_type> &kb,
            const permutation<NC> &permc, bool self);
};


} // namespace libtensor

#endif // LIBTENOSR_GEN_BTO_DIRSUM_SYM_H
