#ifndef LIBTENSOR_CTF_BTOD_DIRSUM_H
#define LIBTENSOR_CTF_BTOD_DIRSUM_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_diag.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief Computes the direct sum of two distributed block tensors
    \tparam N Order of the first tensor.
    \tparam M Order of the second tensor.

    \sa gen_bto_dirsum, btod_dirsum

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N, size_t M>
class ctf_btod_dirsum :
    public additive_gen_bto<N + M, ctf_btod_traits::bti_traits>,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename ctf_btod_traits::bti_traits bti_traits;

private:
    gen_bto_dirsum<N, M, ctf_btod_traits, ctf_btod_dirsum<N, M> > m_gbto;

public:
    /** \brief Initializes the operation
     **/
    ctf_btod_dirsum(
        ctf_block_tensor_rd_i<N, double> &bta, const scalar_transf<double> &ka,
        ctf_block_tensor_rd_i<M, double> &btb, const scalar_transf<double> &kb,
        const tensor_transf<N + M, double> &trc =
            tensor_transf<N + M, double>()) :

        m_gbto(bta, ka, btb, kb, trc) {

    }

    /** \brief Initializes the operation
     **/
    ctf_btod_dirsum(
        ctf_block_tensor_rd_i<N, double> &bta, double ka,
        ctf_block_tensor_rd_i<M, double> &btb, double kb) :

        m_gbto(bta, scalar_transf<double>(ka), btb, scalar_transf<double>(kb)) {
    }

    /** \brief Initializes the operation
     **/
    ctf_btod_dirsum(
        ctf_block_tensor_rd_i<N, double> &bta, double ka,
        ctf_block_tensor_rd_i<M, double> &btb, double kb,
        const permutation<N + M> &permc) :

        m_gbto(bta, scalar_transf<double>(ka), btb, scalar_transf<double>(kb),
            tensor_transf<N + M, double>(permc)) {

    }

    /** \brief Virtual destructor
     **/
    virtual ~ctf_btod_dirsum() { }

    /** \brief Returns block_index_space of result
     **/
    virtual const block_index_space<N + M> &get_bis() const {

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
    virtual void perform(gen_block_stream_i<N + M, bti_traits> &out) {

        m_gbto.perform(out);
    }

    /** \brief Computes the result into an output tensor
     **/
    virtual void perform(gen_block_tensor_i<N + M, bti_traits> &btb);

    /** \brief Adds the result to an output tensor
     **/
    virtual void perform(gen_block_tensor_i<N + M, bti_traits> &btb,
        const scalar_transf<double> &c);

    /** \brief Adds the result to an output tensor
     **/
    void perform(ctf_block_tensor_i<N + M, double> &btb, double c) {

        perform(btb, scalar_transf<double>(c));
    }

    /** \brief Computes one block of the result
     **/
    virtual void compute_block(
        bool zero,
        const index<N + M> &ic,
        const tensor_transf<N + M, double> &trc,
        ctf_dense_tensor_i<N + M, double> &blkc) {

        m_gbto.compute_block(zero, ic, trc, blkc);
    }

    /** \brief Computes one block of the result
     **/
    virtual void compute_block(
        const index<N + M> &ic,
        ctf_dense_tensor_i<N + M, double> &blkc) {

        compute_block(true, ic, tensor_transf<N + M, double>(), blkc);
    }

};


} // namespace libtensor

#endif // LIBTENOSR_CTF_BTOD_DIRSUM_H
