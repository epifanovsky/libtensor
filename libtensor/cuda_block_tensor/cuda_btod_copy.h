#ifndef LIBTENSOR_CUDA_BTOD_COPY_H
#define LIBTENSOR_CUDA_BTOD_COPY_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_copy.h>
#include "cuda_btod_traits.h"

namespace libtensor {


/** \brief Copies a block tensor with an optional transformation
    \tparam N Tensor order.

    \sa gen_bto_copy

    \ingroup libtensor_cuda_btod
 **/
template<size_t N>
class cuda_btod_copy :
    public additive_gen_bto<N, cuda_btod_traits::bti_traits>,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename cuda_btod_traits::bti_traits bti_traits;

private:
    gen_bto_copy< N, cuda_btod_traits, cuda_btod_copy<N> > m_gbto;

public:
    /** \brief Initializes the operation
        \param bta Source block tensor (A).
        \param c Scaling coefficient.
     **/
    cuda_btod_copy(
        cuda_block_tensor_rd_i<N, double> &bta,
        double c = 1.0) :

        m_gbto(bta, tensor_transf<N, double>(
            permutation<N>(), scalar_transf<double>(c))) {

    }

    /** \brief Initializes the operation
        \param bta Source block tensor (A).
        \param perma Permutation of A.
        \param c Scaling coefficient.
     **/
    cuda_btod_copy(
        cuda_block_tensor_rd_i<N, double> &bta,
        const permutation<N> &perma,
        double c = 1.0) :

        m_gbto(bta, tensor_transf<N, double>(perma, scalar_transf<double>(c))) {

    }

    /** \brief Virtual destructor
     **/
    virtual ~cuda_btod_copy() { }

    virtual const block_index_space<N> &get_bis() const {

        return m_gbto.get_bis();
    }

    virtual const symmetry<N, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N, double> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    /** \brief Performs a copy into a stream
        \param out Output block stream.
     **/
    void perform(
        gen_block_stream_i<N, bti_traits> &out) {

        m_gbto.perform(out);
    }

    /** \brief Performs a copy into a block tensor
        \param btb Output block tensor.
     **/
    void perform(
        gen_block_tensor_i<N, bti_traits> &btb);

    /** \brief Performs a copy with addition into a block tensor
        \param btb Output block tensor.
        \param c Scaling coefficient.
     **/
    void perform(
        gen_block_tensor_i<N, bti_traits> &btb,
        const scalar_transf<double> &c);

    /** \brief Performs a copy with addition into a block tensor
        \param btb Output block tensor.
        \param c Scaling coefficient.
     **/
    void perform(
        gen_block_tensor_i<N, bti_traits> &btb,
        double c) {

        perform(btb, scalar_transf<double>(c));
    }

    void compute_block(
        bool zero,
        const index<N> &ib,
        const tensor_transf<N, double> &trb,
        dense_tensor_wr_i<N, double> &blkb) {

        m_gbto.compute_block(zero, ib, trb, blkb);
    }

    void compute_block(
        const index<N> &ib,
        dense_tensor_wr_i<N, double> &blkb) {

        compute_block(true, ib, tensor_transf<N, double>(), blkb);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_BTOD_COPY_H
