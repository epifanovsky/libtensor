#ifndef LIBTENSOR_CTF_BTOD_EWMULT2_H
#define LIBTENSOR_CTF_BTOD_EWMULT2_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_ewmult2.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief Generalized element-wise (Hadamard) product of two distributed block
        tensors
    \tparam N Order of first argument (A) less the number of shared indices.
    \tparam M Order of second argument (B) less the number of shared indices.
    \tparam K Number of shared indices.

    \sa gen_bto_ewmult2, btod_ewmult2

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N, size_t M, size_t K>
class ctf_btod_ewmult2 :
    public additive_gen_bto<N + M + K, ctf_btod_traits::bti_traits>,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

private:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M + K //!< Order of result (C)
    };

public:
    typedef typename ctf_btod_traits::bti_traits bti_traits;

private:
    gen_bto_ewmult2< N, M, K, ctf_btod_traits, ctf_btod_ewmult2<N, M, K> >
        m_gbto;

public:
    /** \brief Initializes the operation
        \param bta First argument (A).
        \param btb Second argument (B).
        \param d Scaling coefficient.
     **/
    ctf_btod_ewmult2(
        ctf_block_tensor_rd_i<NA, double> &bta,
        ctf_block_tensor_rd_i<NB, double> &btb,
        double d = 1.0);

    /** \brief Initializes the operation
        \param bta First argument (A).
        \param perma Permutation of A.
        \param btb Second argument (B).
        \param permb Permutation of B.
        \param permc Permutation of result (C).
        \param d Scaling coefficient.
     **/
    ctf_btod_ewmult2(
        ctf_block_tensor_rd_i<NA, double> &bta,
        const permutation<NA> &perma,
        ctf_block_tensor_rd_i<NB, double> &btb,
        const permutation<NB> &permb,
        const permutation<NC> &permc,
        double d = 1.0);

    /** \brief Initializes the operation
        \param bta First argument (A).
        \param tra Tensor transformation of A.
        \param btb Second argument (B).
        \param trb Tensor transformation of B.
        \param trc Tensor transformation of result (C).
     **/
    ctf_btod_ewmult2(
        ctf_block_tensor_rd_i<NA, double> &bta,
        const tensor_transf<NA, double> &tra,
        ctf_block_tensor_rd_i<NB, double> &btb,
        const tensor_transf<NB, double> &trb,
        const tensor_transf<NC, double> &trc = tensor_transf<NC, double>());

    /** \brief Virtual destructor
     **/
    virtual ~ctf_btod_ewmult2() { }

    /** \brief Returns block_index_space of result
     **/
    virtual const block_index_space<NC> &get_bis() const {

        return m_gbto.get_bis();
    }

    /** \brief Returns symmetry of result
     **/
    virtual const symmetry<NC, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    /** \brief Returns assignment_schedule
     **/
    virtual const assignment_schedule<NC, double> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    /** \brief Computes the result into an output stream
     **/
    virtual void perform(gen_block_stream_i<NC, bti_traits> &out) {

        m_gbto.perform(out);
    }

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

#endif // LIBTENSOR_CTF_BTOD_EWMULT2_H
