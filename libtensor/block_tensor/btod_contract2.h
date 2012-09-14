#ifndef LIBTENSOR_BTOD_CONTRACT2_H
#define LIBTENSOR_BTOD_CONTRACT2_H

#include <libtensor/timings.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_contract2_sym.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/block_tensor/bto/bto_stream_i.h>

namespace libtensor {


template<size_t N, size_t M, size_t K>
struct btod_contract2_clazz {
    static const char *k_clazz;
};


/** \brief Computes the contraction of two block tensors
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.

    \sa gen_bto_contract2

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, size_t M, size_t K>
class btod_contract2 :
    public additive_bto<N + M, btod_traits>,
    public timings< btod_contract2<N, M, K> >,
    public noncopyable {

public:
    static const char *k_clazz; //!< Class name

private:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M //!< Order of result (C)
    };

private:
    contraction2<N, M, K> m_contr; //!< Contraction
    block_tensor_rd_i<NA, double> &m_bta; //!< First argument (A)
    block_tensor_rd_i<NB, double> &m_btb; //!< Second argument (B)
    gen_bto_contract2_sym<N, M, K, btod_traits> m_symc; //!< Symmetry of result (C)
    dimensions<NA> m_bidimsa; //!< Block %index dims of A
    dimensions<NB> m_bidimsb; //!< Block %index dims of B
    dimensions<NC> m_bidimsc; //!< Block %index dims of the result
    assignment_schedule<NC, double> m_sch; //!< Assignment schedule

public:
    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param bta Block %tensor A (first argument).
        \param btb Block %tensor B (second argument).
    **/
    btod_contract2(
        const contraction2<N, M, K> &contr,
        block_tensor_rd_i<NA, double> &bta,
        block_tensor_rd_i<NB, double> &btb);

    /** \brief Virtual destructor
     **/
    virtual ~btod_contract2();

    /** \brief Returns the block index space of the result
     **/
    virtual const block_index_space<NC> &get_bis() const {

        return m_symc.get_bisc();
    }

    /** \brief Returns the symmetry of the result
     **/
    virtual const symmetry<N + M, double> &get_symmetry() const {

        return m_symc.get_symc();
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    virtual const assignment_schedule<N + M, double> &get_schedule() const {

        return m_sch;
    }

    /** \brief Turns on synchronization on all arguments
     **/
    virtual void sync_on();

    /** \brief Turns off synchronization on all arguments
     **/
    virtual void sync_off();

    /** \brief Computes the contraction into an output stream
     **/
    virtual void perform(bto_stream_i<NC, btod_traits> &out);

    /** \brief Computes the contraction into an output block tensor
     **/
    virtual void perform(block_tensor_i<NC, double> &btc);

    /** \brief Computes the contraction and adds to an block tensor
        \param btc Output tensor.
        \param d Scaling coefficient.
     **/
    virtual void perform(
        block_tensor_i<NC, double> &btc,
        const double &d);

    using additive_bto<N + M, btod_traits>::compute_block;
    virtual void compute_block(
        bool zero,
        dense_tensor_i<NC, double> &blk,
        const index<NC> &i,
        const tensor_transf<NC, double> &tr,
        const double &c);

private:
    void make_schedule();

    void contract_block(
        block_tensor_rd_i<N + K, double> &bta,
        const orbit_list<N + K, double> &ola,
        block_tensor_rd_i<M + K, double> &btb,
        const orbit_list<M + K, double> &olb,
        const index<NC> &idxc,
        dense_tensor_i<NC, double> &blkc,
        const tensor_transf<NC, double> &trc,
        bool zero, double c);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_H
