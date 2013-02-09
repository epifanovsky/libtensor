#ifndef LIBTENSOR_CUDA_BTOD_SYMMETRIZE3_H
#define LIBTENSOR_CUDA_BTOD_SYMMETRIZE3_H

#include <map>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/gen_bto_symmetrize3.h>
#include "cuda_btod_traits.h"

namespace libtensor {


/** \brief (Anti-)symmetrizes the result of a block tensor operation
        over three groups of indexes
    \tparam N Tensor order.

    The operation symmetrizes or anti-symmetrizes the result of another
    block tensor operation over three indexes or groups of indexes.

    \f[
        b_{ijk} = P_{\pm} a_{ijk} = a_{ijk} \pm a_{jik} \pm a_{kji} \pm
            a_{ikj} + a_{jki} + a_{kij}
    \f]

    The constructor takes three different indexes to be symmetrized.

    \ingroup libtensor_btod
 **/
template<size_t N>
class cuda_btod_symmetrize3 :
    public additive_gen_bto<N, cuda_btod_traits::bti_traits>,
    public timings< cuda_btod_symmetrize3<N> >,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename cuda_btod_traits::bti_traits bti_traits;

private:
    gen_bto_symmetrize3< N, cuda_btod_traits, cuda_btod_symmetrize3<N> > m_gbto;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param op Operation to be symmetrized.
        \param i1 First index.
        \param i2 Second index.
        \param i3 Third index.
        \param symm True for symmetrization, false for anti-symmetrization.
     **/
    cuda_btod_symmetrize3(additive_gen_bto<N, bti_traits> &op,
        size_t i1, size_t i2, size_t i3, bool symm) :

        m_gbto(op, permutation<N>().permute(i1, i2),
            permutation<N>().permute(i1, i3), symm)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~cuda_btod_symmetrize3() { }

    //@}


    //!    \name Implementation of direct_gen_bto<N, bti_traits>
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
    virtual void perform(gen_block_tensor_i<N, bti_traits> &btc);
    virtual void perform(gen_block_tensor_i<N, bti_traits> &btc,
        const scalar_transf<double> &d);
    virtual void perform(cuda_block_tensor_i<N, double> &btc, double d);

    //@}

protected:
    //!    \brief Implementation of additive_bto<N, cuda_btod_traits>
    //@{

    virtual void compute_block(
        bool zero,
        const index<N> &ib,
        const tensor_transf<N, double> &trb,
        dense_tensor_wr_i<N, double> &blkb);

    virtual void compute_block(
        const index<N> &ib,
        dense_tensor_wr_i<N, double> &blkb);

    //@}

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_BTOD_SYMMETRIZE3_H
