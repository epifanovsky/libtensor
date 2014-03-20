#ifndef LIBTENSOR_CTF_BTOD_SYMMETRIZE3_H
#define LIBTENSOR_CTF_BTOD_SYMMETRIZE3_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/gen_bto_symmetrize3.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief (Anti-)symmetrizes the result of a CTF block tensor operation
        over three groups of indexes
    \tparam N Tensor order.

    \sa gen_bto_symmetrize3, btod_symmetrize3

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N>
class ctf_btod_symmetrize3 :
    public additive_gen_bto<N, ctf_btod_traits::bti_traits>,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename ctf_btod_traits::bti_traits bti_traits;

private:
    gen_bto_symmetrize3< N, ctf_btod_traits, ctf_btod_symmetrize3<N> > m_gbto;

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
    ctf_btod_symmetrize3(additive_gen_bto<N, bti_traits> &op,
        size_t i1, size_t i2, size_t i3, bool symm) :

        m_gbto(op, permutation<N>().permute(i1, i2),
            permutation<N>().permute(i1, i3), symm)
    { }

    /** \brief Initializes the operation
        \param op Operation to be symmetrized.
        \param perm1 First generating permutation.
        \param perm2 Second generating permutation.
        \param symm True for symmetrization, false for anti-symmetrization.
     **/
    ctf_btod_symmetrize3(additive_gen_bto<N, bti_traits> &op,
        const permutation<N> &perm1, const permutation<N> &perm2, bool symm) :

        m_gbto(op, perm1, perm2, symm)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~ctf_btod_symmetrize3() { }

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

    virtual void perform(gen_block_stream_i<N, bti_traits> &out) {

        m_gbto.perform(out);
    }

    virtual void perform(gen_block_tensor_i<N, bti_traits> &btc);
    virtual void perform(gen_block_tensor_i<N, bti_traits> &btc,
        const scalar_transf<double> &d);

    virtual void perform(ctf_block_tensor_i<N, double> &btc, double d) {

        perform(btc, scalar_transf<double>(d));
    }

    //@}

    //!    \brief Implementation of additive_bto<N, btod_traits>
    //@{

    virtual void compute_block(
        bool zero,
        const index<N> &ib,
        const tensor_transf<N, double> &trb,
        ctf_dense_tensor_i<N, double> &blkb) {

        m_gbto.compute_block(zero, ib, trb, blkb);
    }

    virtual void compute_block(
        const index<N> &ib,
        ctf_dense_tensor_i<N, double> &blkb) {

        compute_block(true, ib, tensor_transf<N, double>(), blkb);
    }

    //@}

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_SYMMETRIZE3_H
