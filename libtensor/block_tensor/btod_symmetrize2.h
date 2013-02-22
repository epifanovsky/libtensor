#ifndef LIBTENSOR_BTOD_SYMMETRIZE2_H
#define LIBTENSOR_BTOD_SYMMETRIZE2_H

#include <map>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_symmetrize2.h>

namespace libtensor {


/** \brief Symmetrizes the result of another block %tensor operation
    \tparam N Tensor order.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N>
class btod_symmetrize2 :
    public additive_gen_bto<N, btod_traits::bti_traits>,
    public timings< btod_symmetrize2<N> >,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename btod_traits::bti_traits bti_traits;

private:
    gen_bto_symmetrize2< N, btod_traits, btod_symmetrize2<N> > m_gbto;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation to symmetrize two indexes
        \param op Symmetrized operation.
        \param i1 First tensor index.
        \param i2 Second tensor index.
        \param symm True for symmetric, false for anti-symmetric.
     **/
    btod_symmetrize2(additive_gen_bto<N, bti_traits> &op,
        size_t i1, size_t i2, bool symm) :

        m_gbto(op, permutation<N>().permute(i1, i2), symm)
    { }

    /** \brief Initializes the operation using a unitary permutation (P = P^-1)
        \param op Symmetrized operation.
        \param perm Unitary permutation.
        \param symm True for symmetric, false for anti-symmetric.
     **/
    btod_symmetrize2(additive_gen_bto<N, bti_traits> &op,
        const permutation<N> &perm, bool symm) :

        m_gbto(op, perm, symm)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~btod_symmetrize2() { }

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
    virtual void perform(block_tensor_i<N, double> &btc, double d);

    //@}

    //!    \brief Implementation of additive_bto<N, btod_traits>
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

#endif // LIBTENSOR_BTOD_SYMMETRIZE2_H
