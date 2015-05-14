#ifndef LIBTENSOR_CTF_BTOD_CONTRACT2_H
#define LIBTENSOR_CTF_BTOD_CONTRACT2_H

#include <libtensor/core/contraction2.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_contract2_simple.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief Computes the contraction of two distributed block tensors
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.

    \sa gen_bto_contract2, btod_contract2

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N, size_t M, size_t K>
class ctf_btod_contract2 :
    public additive_gen_bto<N + M, ctf_btod_traits::bti_traits>,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

private:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M //!< Order of result (C)
    };

public:
    typedef typename ctf_btod_traits::bti_traits bti_traits;

private:
    gen_bto_contract2_simple< N, M, K, ctf_btod_traits,
        ctf_btod_contract2<N, M, K> > m_gbto;

public:
    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param bta Block %tensor A (first argument).
        \param btb Block %tensor B (second argument).
     **/
    ctf_btod_contract2(
        const contraction2<N, M, K> &contr,
        ctf_block_tensor_rd_i<NA, double> &bta,
        ctf_block_tensor_rd_i<NB, double> &btb);

    /** \brief Initializes the contraction operation with scaling coefficients
        \param contr Contraction.
        \param bta Block tensor A (first argument).
        \param ka Scalar for A.
        \param btb Block tensor B (second argument).
        \param kb Scalar for B.
        \param kc Scalar for result.
     **/
    ctf_btod_contract2(
        const contraction2<N, M, K> &contr,
        ctf_block_tensor_rd_i<NA, double> &bta,
        double ka,
        ctf_block_tensor_rd_i<NB, double> &btb,
        double kb,
        double kc);

    /** \brief Virtual destructor
     **/
    virtual ~ctf_btod_contract2() { }

    /** \brief Returns block_index_space of result
     **/
    virtual const block_index_space<NC> &get_bis() const {

        return m_gbto.get_bis();
    }

    /** \brief Returns symmetry of result
     **/
    virtual const symmetry<N + M, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    /** \brief Returns assignment_schedule
     **/
    virtual const assignment_schedule<N + M, double> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    /** \brief Computes the result into an output stream
     **/
    virtual void perform(gen_block_stream_i<NC, bti_traits> &out);

    /** \brief Computes the result into an output tensor
     **/
    virtual void perform(gen_block_tensor_i<NC, bti_traits> &btc);

    /** \brief Adds the result to an output tensor
     **/
    virtual void perform(gen_block_tensor_i<NC, bti_traits> &btc,
        const scalar_transf<double> &d);

    /** \brief Adds the result to an output tensor
     **/
    void perform(ctf_block_tensor_i<NC, double> &btc, double d) {

        perform(btc, scalar_transf<double>(d));
    }

    /** \brief Computes one block of the result
     **/
    virtual void compute_block(
        bool zero,
        const index<NC> &ic,
        const tensor_transf<NC, double> &trc,
        ctf_dense_tensor_i<NC, double> &blkc) {

        m_gbto.compute_block(zero, ic, trc, blkc);
    }

    /** \brief Computes one block of the result
     **/
    virtual void compute_block(
        const index<NC> &ic,
        ctf_dense_tensor_i<NC, double> &blkc) {

        compute_block(true, ic, tensor_transf<NC, double>(), blkc);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_CONTRACT2_H
