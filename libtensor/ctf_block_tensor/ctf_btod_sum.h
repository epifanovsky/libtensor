#ifndef LIBTENSOR_CTF_BTOD_SUM_H
#define LIBTENSOR_CTF_BTOD_SUM_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_sum.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief Adds results of a sequence of block tensor operations
        (for double)
    \tparam N Tensor order.

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N>
class ctf_btod_sum : public additive_gen_bto<N, ctf_btod_traits::bti_traits> {
public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename ctf_btod_traits::bti_traits bti_traits;

private:
    gen_bto_sum<N, ctf_btod_traits> m_gbto;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the base operation
        \param op Operation.
        \param c Coefficient.
     **/
    ctf_btod_sum(additive_gen_bto<N, bti_traits> &op, double c = 1.0) :
        m_gbto(op, scalar_transf<double>(c))
    { }

    /** \brief Virtual destructor
     **/
    virtual ~ctf_btod_sum() { }

    //@}


    //!    \name Implementation of libtensor::direct_tensor_operation<N>
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

    //@}


    //!    \name Implementation of libtensor::additive_bto<N, btod_traits>
    //@{

    virtual void compute_block(
        bool zero,
        const index<N> &i,
        const tensor_transf<N, double> &tr,
        ctf_dense_tensor_i<N, double> &blk) {

        m_gbto.compute_block(zero, i, tr, blk);
    }

    virtual void compute_block(
        const index<N> &ib,
        ctf_dense_tensor_i<N, double> &blkb) {

        compute_block(true, ib, tensor_transf<N, double>(), blkb);
    }

    virtual void perform(gen_block_stream_i<N, bti_traits> &out) {
        m_gbto.perform(out);
    }

    virtual void perform(gen_block_tensor_i<N, bti_traits> &btb);
    virtual void perform(gen_block_tensor_i<N, bti_traits> &btb,
        const scalar_transf<double> &c);

    //@}

    void perform(gen_block_tensor_i<N, bti_traits> &btb, double c) {
        perform(btb, scalar_transf<double>(c));
    }

    //!    \name Manipulations
    //@{

    /** \brief Adds an operation to the sequence
        \param op Operation.
        \param c Coefficient.
     **/
    void add_op(additive_gen_bto<N, bti_traits> &op, double c = 1.0) {
        m_gbto.add_op(op, scalar_transf<double>(c));
    }

    //@}

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_SUM_H

