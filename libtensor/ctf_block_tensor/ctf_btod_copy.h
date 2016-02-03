#ifndef LIBTENSOR_CTF_BTOD_COPY_H
#define LIBTENSOR_CTF_BTOD_COPY_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_copy.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief Copies a distributed CTF block tensor with an optional transformation
    \tparam N Tensor order.

    \sa gen_bto_copy, btod_copy

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N>
class ctf_btod_copy :
    public additive_gen_bto<N, ctf_btod_traits::bti_traits>,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename ctf_btod_traits::bti_traits bti_traits;

private:
    gen_bto_copy< N, ctf_btod_traits, ctf_btod_copy<N> > m_gbto;

public:
    /** \brief Initializes the operation
        \param bta Source block tensor (A).
        \param c Scaling coefficient.
     **/
    ctf_btod_copy(
        ctf_block_tensor_rd_i<N, double> &bta,
        double c = 1.0) :

        m_gbto(bta, tensor_transf<N, double>(
            permutation<N>(), scalar_transf<double>(c))) {

    }

    /** \brief Initializes the operation
        \param bta Source block tensor (A).
        \param perma Permutation of A.
        \param c Scaling coefficient.
     **/
    ctf_btod_copy(
        ctf_block_tensor_rd_i<N, double> &bta,
        const permutation<N> &perma,
        double c = 1.0) :

        m_gbto(bta, tensor_transf<N, double>(perma, scalar_transf<double>(c))) {

    }

    /** \brief Virtual destructor
     **/
    virtual ~ctf_btod_copy() { }

    /** \brief Returns block_index_space of result
     **/
    virtual const block_index_space<N> &get_bis() const {

        return m_gbto.get_bis();
    }

    /** \brief Returns symmetry of result
     **/
    virtual const symmetry<N, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    /** \brief Returns assignment_schedule
     **/
    virtual const assignment_schedule<N, double> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    /** \brief Computes the result into an output stream
     **/
    virtual void perform(gen_block_stream_i<N, bti_traits> &out) {

        m_gbto.perform(out);
    }

    /** \brief Computes the result into an output tensor
     **/
    virtual void perform(gen_block_tensor_i<N, bti_traits> &btb);

    /** \brief Adds the result to an output tensor
     **/
    virtual void perform(gen_block_tensor_i<N, bti_traits> &btb,
        const scalar_transf<double> &c);

    /** \brief Adds the result to an output tensor
     **/
    void perform(ctf_block_tensor_i<N, double> &btb, double c) {

        perform(btb, scalar_transf<double>(c));
    }

    /** \brief Computes one block of the result
     **/
    virtual void compute_block(
        bool zero,
        const index<N> &ib,
        const tensor_transf<N, double> &trb,
        ctf_dense_tensor_i<N, double> &blkb) {

        m_gbto.compute_block(zero, ib, trb, blkb);
    }

    /** \brief Computes one block of the result
     **/
    virtual void compute_block(
        const index<N> &ib,
        ctf_dense_tensor_i<N, double> &blkb) {

        compute_block(true, ib, tensor_transf<N, double>(), blkb);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_COPY_H
