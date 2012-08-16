#ifndef LIBTENSOR_ADDITIVE_BTO_H
#define LIBTENSOR_ADDITIVE_BTO_H

#include <vector>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/tensor_transf.h>
#include "basic_bto.h"
#include "addition_schedule.h"

namespace libtensor {


/** \brief Base class for additive block %tensor operations
    \tparam N Tensor order.

    Additive block %tensor operations are those that can add their result
    to the output block %tensor as opposed to simply replacing it. This
    class extends basic_btod<N> with two new functions: one is invoked
    to perform the block %tensor operation additively, the other does that
    for only one canonical block.

    The scalar transformation provided in both functions transforms the
    elements of the result of the operation before adding it to the output
    block %tensor.

    \ingroup libtensor_btod
 **/
template<size_t N, typename Traits>
class additive_bto: public basic_bto<N, Traits> {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_t;

    //! Type of block tensors
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_t;

    //! Type of blocks of block tensors
    typedef typename Traits::template block_type<N>::type block_t;

private:
    typedef addition_schedule<N, Traits> schedule_t;

private:
    class task: public libutil::task_i {
    private:
        additive_bto<N, Traits> &m_bto;
        block_tensor_t &m_bt;
        const dimensions<N> &m_bidims;
        const addition_schedule<N, Traits> &m_sch;
        typename addition_schedule<N, Traits>::iterator m_i;
        element_t m_c;

    public:
        task(additive_bto<N, Traits> &bto, block_tensor_t &bt,
            const dimensions<N> &bidims,
            const addition_schedule<N, Traits> &sch,
            typename addition_schedule<N, Traits>::iterator &i,
            const element_t &c) :
                m_bto(bto), m_bt(bt), m_bidims(bidims),
                m_sch(sch), m_i(i), m_c(c) {
        }
        virtual ~task() { }
        virtual void perform();
    };

    class task_iterator : public libutil::task_iterator_i {
    private:
        std::vector<task*> &m_tl;
        typename std::vector<task*>::iterator m_i;
    public:
        task_iterator(std::vector<task*> &tl) :
            m_tl(tl), m_i(m_tl.begin()) { }
        virtual bool has_more() const;
        virtual libutil::task_i *get_next();
    };

    class task_observer : public libutil::task_observer_i {
    public:
        virtual void notify_start_task(libutil::task_i *t) { }
        virtual void notify_finish_task(libutil::task_i *t) { }
    };

public:
    using basic_bto<N, Traits>::get_bis;
    using basic_bto<N, Traits>::get_schedule;
    using basic_bto<N, Traits>::get_symmetry;
    using basic_bto<N, Traits>::sync_on;
    using basic_bto<N, Traits>::sync_off;
    using basic_bto<N, Traits>::perform;

public:
    /** \brief Computes the result of the operation and adds it to the
            output block tensor
        \param bt Output block tensor.
        \param c Scaling coefficient.
     **/
    virtual void perform(block_tensor_t &bt, const element_t &c);

    /** \brief Partially computes the result of the operation and adds it to the
            output block tensor
        \param bt Output block tensor.
        \param c Scaling coefficient.
        \param blst List of canonical blocks to compute.
     **/
    virtual void perform(block_tensor_t &bt, const element_t &c,
        const std::vector<size_t> &blst);

    /** \brief Implementation of basic_btod<N>::compute_block
        \param blk Output %tensor.
        \param i Index of the block to compute.
     **/
    virtual void compute_block(block_t &blk, const index<N> &i);

    /** \brief Computes a single block of the result and adds it to
            the output %tensor
        \param zero Zero out the output before the computation.
        \param blk Output %tensor.
        \param i Index of the block to compute.
        \param tr Transformation of the block.
        \param c Scaling coefficient.
     **/
    virtual void compute_block(bool zero, block_t &blk, const index<N> &i,
        const tensor_transf<N, element_t> &tr, const element_t &c) = 0;

protected:
    /** \brief Invokes compute_block on another additive operation;
            allows derived classes to call other additive operations
     **/
    void compute_block(additive_bto<N, Traits> &op, bool zero, block_t &blk,
        const index<N> &i, const tensor_transf<N, element_t> &tr,
        const element_t &c);

private:
    void perform_inner(block_tensor_t &bt, const element_t &c, bool filter,
        const std::vector<size_t> &blst);

};


} // namespace libtensor


#endif // LIBTENSOR_ADDITIVE_BTO_H
