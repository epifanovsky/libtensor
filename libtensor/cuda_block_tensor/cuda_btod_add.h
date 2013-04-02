#ifndef LIBTENSOR_CUDA_BTOD_ADD_H
#define LIBTENSOR_CUDA_BTOD_ADD_H

#include <libtensor/cuda_block_tensor/cuda_btod_traits.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_add.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>

namespace libtensor {


/** \brief Linear combination of multiple block tensors
    \tparam N Tensor order.

    \sa gen_bto_add

    \ingroup libtensor_btod
 **/
template<size_t N>
class cuda_btod_add :
    public additive_gen_bto<N, cuda_btod_traits::bti_traits>,
    public noncopyable {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename cuda_btod_traits::bti_traits bti_traits;

private:
    gen_bto_add< N, cuda_btod_traits, cuda_btod_add<N> > m_gbto;

public:
    /** \brief Initializes the operation
        \param bta Source block tensor (A).
        \param c Scaling coefficient.
     **/
    cuda_btod_add(
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
    cuda_btod_add(
            cuda_block_tensor_rd_i<N, double> &bta,
            const permutation<N> &perma,
            double c = 1.0) :

        m_gbto(bta, tensor_transf<N, double>(perma,
                scalar_transf<double>(c))) {
    }

    virtual ~cuda_btod_add() { }

    /** \brief Adds another argument to the linear combination sequence
     **/
    void add_op(
            cuda_block_tensor_rd_i<N, double> &bta,
            double c = 1.0) {

        m_gbto.add_op(bta, tensor_transf<N, double>(permutation<N>(),
            scalar_transf<double>(c)));
    }

    /** \brief Adds another argument to the linear combination sequence
     **/
    void add_op(
            cuda_block_tensor_rd_i<N, double> &bta,
            const permutation<N> &perma,
            double c = 1.0) {

        m_gbto.add_op(bta, tensor_transf<N, double>(perma,
            scalar_transf<double>(c)));
    }

    //! \name Implementation of libtensor::direct_gen_bto<N, bti_traits>
    //@{

    virtual const block_index_space<N> &get_bis() const {

        return m_gbto.get_bis();
    }

    virtual const symmetry<N, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N, double> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    virtual void perform(gen_block_stream_i<N, bti_traits> &out);

    //@}

    //! \name Implementation of libtensor::additive_gen_bto<N, bti_traits>
    //@{

    virtual void perform(gen_block_tensor_i<N, bti_traits> &btb);

    virtual void perform(gen_block_tensor_i<N, bti_traits> &btb,
            const scalar_transf<double> &c);

    virtual void compute_block(
            bool zero,
            const index<N> &ib,
            const tensor_transf<N, double> &trb,
            cuda_dense_tensor_wr_i<N, double> &blkb);

    virtual void compute_block(
            const index<N> &ib,
            cuda_dense_tensor_wr_i<N, double> &blkb) {

        compute_block(true, ib, tensor_transf<N, double>(), blkb);
    }

    //@}

    /** \brief Function for compatability
     **/
    void perform(cuda_block_tensor_i<N, double> &btb, double c);

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_BTOD_ADD_H
