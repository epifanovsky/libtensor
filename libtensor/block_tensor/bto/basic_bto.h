#ifndef LIBTENSOR_BASIC_BTO_H
#define LIBTENSOR_BASIC_BTO_H

#include <libutil/thread_pool/thread_pool.h>
#include "direct_bto.h"

namespace libtensor {


/** \brief Basic functionality of block %tensor operations
    \tparam N Tensor order.

    Extends the direct_block_tensor_operation interface. Implements
    a method to compute the result in its entirety using the assignment
    schedule (preferred order of block computation) provided by derived
    block %tensor operations.

    Derived classes shall implement get_schedule().

    \sa assignment_schedule, direct_block_tensor_operation

    \ingroup libtensor_btod
 **/
template<size_t N, typename Traits>
class basic_bto :
    public direct_bto<N, Traits> {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_t;

    //! Type of block tensors
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_t;

    using direct_bto<N, Traits>::get_bis;
    using direct_bto<N, Traits>::get_symmetry;
    using direct_bto<N, Traits>::get_schedule;
    using direct_bto<N, Traits>::sync_on;
    using direct_bto<N, Traits>::sync_off;

private:
    class task : public libutil::task_i {
    private:
        basic_bto<N, Traits> &m_bto;
        block_tensor_t &m_bt;
        const dimensions<N> &m_bidims;
        const assignment_schedule<N, element_t> &m_sch;
        typename assignment_schedule<N, element_t>::iterator m_i;

    public:
        task(basic_bto<N, Traits> &bto, block_tensor_t &bt,
            const dimensions<N> &bidims,
            const assignment_schedule<N, element_t> &sch,
            typename assignment_schedule<N, element_t>::iterator i) :
            m_bto(bto), m_bt(bt), m_bidims(bidims), m_sch(sch),
            m_i(i) { }
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
    /** \brief Computes the result of the operation into an output
            block %tensor
        \param bt Output block %tensor.
     **/
    virtual void perform(block_tensor_t &bt);

protected:
    using direct_bto<N, Traits>::compute_block;

};


} // namespace libtensor


#endif // LIBTENSOR_BASIC_BTO_H
