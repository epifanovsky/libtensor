#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_H

#include <libtensor/timings.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/tensor_transf.h>

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
class gen_bto_contract2_block : public timings<Timed>, public noncopyable {
public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block (A)
    typedef typename bti_traits::template rd_block_type<NA>::type
        rd_block_a_type;

    //! Type of read-only block (B)
    typedef typename bti_traits::template rd_block_type<NB>::type
        rd_block_b_type;

    //! Type of write-only block (C)
    typedef typename bti_traits::template wr_block_type<NC>::type
        wr_block_c_type;

private:
    contraction2<N, M, K> m_contr; //!< Contraction
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta; //!< First block tensor (A)
    dimensions<NA> m_bidimsa; //!< Block index dims in A
    orbit_list<NA, element_type> m_ola; //!< List of orbits in A
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb; //!< Second block tensor (B)
    dimensions<NB> m_bidimsb; //!< Block index dims in B
    orbit_list<NB, element_type> m_olb; //!< List of orbits in B
    dimensions<NC> m_bidimsc; //!< Block index dims in C

public:
    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param bta First block tensor (A).
        \param btb Second block tensor (B).
     **/
    gen_bto_contract2_block(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        const symmetry<NA, element_type> &syma,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const symmetry<NB, element_type> &symb,
        const block_index_space<NC> &bisc);

    void compute_block(
        bool zero,
        const index<NC> &idxc,
        const tensor_transf<NC, element_type> &trc,
        wr_block_c_type &blkc);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_H
