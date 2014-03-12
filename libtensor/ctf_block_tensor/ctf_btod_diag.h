#ifndef LIBTENSOR_CTF_BTOD_DIAG_H
#define LIBTENSOR_CTF_BTOD_DIAG_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_diag.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief Extracts a general diagonal from a distributed block %tensor
    \tparam N Tensor order.
    \tparam M Diagonal order.

    \sa gen_bto_diag, btod_diag

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N, size_t M>
class ctf_btod_diag :
    public additive_gen_bto<N - M + 1, ctf_btod_traits::bti_traits>,
    public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename ctf_btod_traits::bti_traits bti_traits;

private:
    gen_bto_diag<N, M, ctf_btod_traits, ctf_btod_diag<N, M> > m_gbto;

public:
    /** \brief Initializes the diagonal extraction operation
        \param bta Input block %tensor
        \param msk Mask which specifies the indexes to take the diagonal
        \param c Scaling factor
     **/
    ctf_btod_diag(
        ctf_block_tensor_rd_i<N, double> &bta,
        const mask<N> &m,
        double c = 1.0) :

        m_gbto(bta, m, tensor_transf<N - M + 1, double>(
            permutation<N - M + 1>(), scalar_transf<double>(c))) {

    }

    /** \brief Initializes the diagonal extraction operation
        \param bta Input block %tensor
        \param msk Mask which specifies the indexes to take the diagonal
        \param p Permutation of result tensor
        \param c Scaling factor
     **/
    ctf_btod_diag(
        ctf_block_tensor_rd_i<N, double> &bta,
        const mask<N> &m,
        const permutation<N - M + 1> &p,
        double c = 1.0) :

        m_gbto(bta, m, tensor_transf<N - M + 1, double>(
            p, scalar_transf<double>(c))) {
    }

    /** \brief Virtual destructor
     **/
    virtual ~ctf_btod_diag() { }

    /** \brief Returns block_index_space of result
     **/
    virtual const block_index_space<N - M + 1> &get_bis() const {

        return m_gbto.get_bis();
    }

    /** \brief Returns symmetry of result
     **/
    virtual const symmetry<N - M + 1, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    /** \brief Returns assignment_schedule
     **/
    virtual const assignment_schedule<N - M + 1, double> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    /** \brief Computes the result into an output stream
     **/
    virtual void perform(gen_block_stream_i<N - M + 1, bti_traits> &out) {

        m_gbto.perform(out);
    }

    /** \brief Computes the result into an output tensor
     **/
    virtual void perform(gen_block_tensor_i<N - M + 1, bti_traits> &btb);

    /** \brief Adds the result to an output tensor
     **/
    virtual void perform(gen_block_tensor_i<N - M + 1, bti_traits> &btb,
        const scalar_transf<double> &c);

    /** \brief Adds the result to an output tensor
     **/
    void perform(ctf_block_tensor_i<N - M + 1, double> &btb, double c) {

        perform(btb, scalar_transf<double>(c));
    }

    /** \brief Computes one block of the result
     **/
    virtual void compute_block(
        bool zero,
        const index<N - M + 1> &ib,
        const tensor_transf<N - M + 1, double> &trb,
        ctf_dense_tensor_i<N - M + 1, double> &blkb) {

        m_gbto.compute_block(zero, ib, trb, blkb);
    }

    /** \brief Computes one block of the result
     **/
    virtual void compute_block(
        const index<N - M + 1> &ib,
        ctf_dense_tensor_i<N - M + 1, double> &blkb) {

        compute_block(true, ib, tensor_transf<N - M + 1, double>(), blkb);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_DIAG_H
