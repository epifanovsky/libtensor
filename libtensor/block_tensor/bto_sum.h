#ifndef LIBTENSOR_BTO_SUM_H
#define LIBTENSOR_BTO_SUM_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_sum.h>
#include "bto_traits.h"

namespace libtensor {


/** \brief Adds results of a sequence of block %tensor operations
        (for T)
    \tparam N Tensor order.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, typename T>
class bto_sum :
    public additive_gen_bto<N, typename bto_traits<T>::bti_traits>,
    public timings< bto_sum<N, T> > {

public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename bto_traits<T>::bti_traits bti_traits;

private:
    gen_bto_sum<N, bto_traits<T> > m_gbto;

private:

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the base operation
        \param op Operation.
        \param c Coefficient.
     **/
    bto_sum(additive_gen_bto<N, bti_traits> &op, T c = 1.0) :
        m_gbto(op, scalar_transf<T>(c))
    { }

    /** \brief Virtual destructor
     **/
    virtual ~bto_sum() { }

    //@}


    //!    \name Implementation of libtensor::direct_tensor_operation<N>
    //@{

    virtual const block_index_space<N> &get_bis() const {
        return m_gbto.get_bis();
    }

    virtual const symmetry<N, T> &get_symmetry() const {
        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N, T> &get_schedule() const {
        return m_gbto.get_schedule();
    }

    //@}


    //!    \name Implementation of libtensor::additive_bto<N, bto_traits<T> >
    //@{

    virtual void compute_block(
        bool zero,
        const index<N> &i,
        const tensor_transf<N, T> &tr,
        dense_tensor_wr_i<N, T> &blk) {

        m_gbto.compute_block(zero, i, tr, blk);
    }

    virtual void compute_block(
        const index<N> &ib,
        dense_tensor_wr_i<N, T> &blkb) {

        compute_block(true, ib, tensor_transf<N, T>(), blkb);
    }

    virtual void perform(gen_block_stream_i<N, bti_traits> &out) {
        m_gbto.perform(out);
    }

    virtual void perform(gen_block_tensor_i<N, bti_traits> &btb);
    virtual void perform(gen_block_tensor_i<N, bti_traits> &btb,
        const scalar_transf<T> &c);

    //@}

    void perform(gen_block_tensor_i<N, bti_traits> &btb, T c) {
        perform(btb, scalar_transf<T>(c));
    }

    //!    \name Manipulations
    //@{

    /** \brief Adds an operation to the sequence
        \param op Operation.
        \param c Coefficient.
     **/
    void add_op(additive_gen_bto<N, bti_traits> &op, T c = 1.0) {
        m_gbto.add_op(op, scalar_transf<T>(c));
    }

    //@}

};

template<size_t N>
using btod_sum = bto_sum<N, double>;

} // namespace libtensor

#endif // LIBTENSOR_BTO_SUM_H

