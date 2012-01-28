#ifndef LIBTENSOR_ADDITIVE_BTOD_H
#define LIBTENSOR_ADDITIVE_BTOD_H

#include <cmath>
#include "../tod/tod_add.h"
#include "../tod/tod_copy.h"
#include <libtensor/dense_tensor/tod_set.h>
#include "basic_btod.h"
#include "addition_schedule.h"
#include "transf_double.h"

namespace libtensor {


/**	\brief Base class for additive block %tensor operations
    \tparam N Tensor order.

    Additive block %tensor operations are those that can add their result
    to the output block %tensor as opposed to simply replacing it. This
    class extends basic_btod<N> with two new functions: one is invoked
    to perform the block %tensor operation additively, the other does that
    for only one canonical block.

    The coefficient provided in both functions scales the result of the
    operation before adding it to the output block %tensor.

    \ingroup libtensor_btod
 **/
template<size_t N>
class additive_btod: public basic_btod<N> {
private:
    class task: public task_i {
    private:
        additive_btod<N> &m_btod;
        block_tensor_i<N, double> &m_bt;
        const dimensions<N> &m_bidims;
        const addition_schedule<N, double> &m_sch;
        typename addition_schedule<N, double>::iterator m_i;
        double m_c;

    public:
        task(additive_btod<N> &btod, block_tensor_i<N, double> &bt,
            const dimensions<N> &bidims,
            const addition_schedule<N, double> &sch,
            typename addition_schedule<N, double>::iterator &i, double c) :
            m_btod(btod), m_bt(bt), m_bidims(bidims), m_sch(sch), m_i(i),
                m_c(c) {
        }
        virtual ~task() {
        }
        virtual void perform(cpu_pool &cpus) throw (exception);
    };

public:
    using basic_btod<N>::get_bis;
    using basic_btod<N>::get_symmetry;
    using basic_btod<N>::get_schedule;
    using basic_btod<N>::sync_on;
    using basic_btod<N>::sync_off;
    using basic_btod<N>::perform;

public:
    /**	\brief Computes the result of the operation and adds it to the
            output block %tensor
        \param bt Output block %tensor.
        \param c Scaling coefficient.
     **/
    virtual void perform(block_tensor_i<N, double> &bt, double c);

    /** \brief Implementation of basic_btod<N>::compute_block
        \param blk Output %tensor.
        \param i Index of the block to compute.
        \param cpus Pool of CPUs.
     **/
    virtual void compute_block(dense_tensor_i<N, double> &blk, const index<N> &i,
        cpu_pool &cpus);

    /**	\brief Computes a single block of the result and adds it to
            the output %tensor
        \param zero Zero out the output before the computation.
        \param blk Output %tensor.
        \param i Index of the block to compute.
        \param tr Transformation of the block.
        \param c Scaling coefficient.
        \param cpus Pool of CPUs.
     **/
    virtual void compute_block(bool zero, dense_tensor_i<N, double> &blk,
        const index<N> &i, const transf<N, double> &tr, double c,
        cpu_pool &cpus) = 0;

protected:
    /**	\brief Invokes compute_block on another additive operation;
            allows derived classes to call other additive operations
     **/
    void compute_block(additive_btod<N> &op, bool zero,
        dense_tensor_i<N, double> &blk, const index<N> &i,
        const transf<N, double> &tr, double c, cpu_pool &cpus);

private:
    typedef addition_schedule<N, double> schedule_t;

};


} // namespace libtensor


#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

    extern template class additive_btod<1>;
    extern template class additive_btod<2>;
    extern template class additive_btod<3>;
    extern template class additive_btod<4>;
    extern template class additive_btod<5>;
    extern template class additive_btod<6>;

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "additive_btod_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_ADDITIVE_BTOD_H
