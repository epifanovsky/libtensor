#ifndef LIBTENSOR_BTOD_DIRSUM_H
#define LIBTENSOR_BTOD_DIRSUM_H

#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_dirsum.h>

namespace libtensor {


/** \brief Computes the direct sum of two block tensors
    \tparam N Order of the first %tensor.
    \tparam M Order of the second %tensor.

    Given two tensors \f$ a_{ij\cdots} \f$ and \f$ b_{mn\cdots} \f$,
    the operation computes
    \f$ c_{ij\cdots mn\cdots} = k_a a_{ij\cdots} + k_b b_{mn\cdots} \f$.

    The order of %tensor indexes in the result can be specified using
    a permutation.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, size_t M>
class btod_dirsum :
    public additive_gen_bto<N + M, btod_traits::bti_traits>,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename btod_traits::bti_traits bti_traits;
    typedef typename bti_traits::template wr_block_type<N + M>::type
            wr_block_type;
    typedef tensor_transf<N + M, double> tensor_transf_type;

private:
    gen_bto_dirsum<N, M, btod_traits, btod_dirsum<N, M> > m_gbto;

public:
    /** \brief Initializes the operation
     **/
    btod_dirsum(
            block_tensor_rd_i<N, double> &bta, const scalar_transf<double> &ka,
            block_tensor_rd_i<M, double> &btb, const scalar_transf<double> &kb,
            const tensor_transf_type &trc = tensor_transf_type()) :

        m_gbto(bta, ka, btb, kb, trc) {

    }


    /** \brief Initializes the operation
     **/
    btod_dirsum(
            block_tensor_rd_i<N, double> &bta, double ka,
            block_tensor_rd_i<M, double> &btb, double kb) :

        m_gbto(bta, scalar_transf<double>(ka), btb, scalar_transf<double>(kb)) {
    }

    /** \brief Initializes the operation
     **/
    btod_dirsum(
            block_tensor_rd_i<N, double> &bta, double ka,
            block_tensor_rd_i<M, double> &btb, double kb,
            const permutation<N + M> &permc) :

        m_gbto(bta, scalar_transf<double>(ka), btb, scalar_transf<double>(kb),
            tensor_transf<N + M, double>(permc)) {

    }

    /** \brief Virtual destructor
     **/
    virtual ~btod_dirsum() { }

    //! \name Implementation of libtensor::direct_gen_bto<N, bti_traits>
    //@{

    virtual const block_index_space<N + M> &get_bis() const {

        return m_gbto.get_bis();
    }

    virtual const symmetry<N + M, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N + M, double> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    virtual void perform(gen_block_stream_i<N + M, bti_traits> &out) {

        m_gbto.perform(out);
    }

    //@}

    //! \name Implementation of libtensor::additive_gen_bto<N, bti_traits>
    //@{

    virtual void perform(gen_block_tensor_i<N + M, bti_traits> &btb);

    virtual void perform(gen_block_tensor_i<N + M, bti_traits> &btb,
            const scalar_transf<double> &c);

    virtual void compute_block(
            bool zero,
            const index<N + M> &ic,
            const tensor_transf<N + M, double> &trc,
            dense_tensor_wr_i<N + M, double> &blkc);

    virtual void compute_block(
            const index<N + M> &ic,
            wr_block_type &blkc) {

        compute_block(true, ic, tensor_transf<N + M, double>(), blkc);
    }

    //@}

    void perform(block_tensor_i<N + M, double> &btb, double c);
};


} // namespace libtensor

#endif // LIBTENOSR_BTOD_DIRSUM_H
