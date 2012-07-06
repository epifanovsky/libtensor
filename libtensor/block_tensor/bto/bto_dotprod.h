#ifndef LIBTENSOR_BTO_DOTPROD_H
#define LIBTENSOR_BTO_DOTPROD_H

#include <list>
#include <vector>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/defs.h>
#include <libtensor/timings.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation.h>

namespace libtensor {


/** \brief Computes the dot product of two block tensors
    \tparam N Tensor order.

    The dot product of two tensors is defined as the sum of elements of
    the element-wise product:

    \f[ c = \sum_i a_i b_i \f]

    This operation computes the dot product for a series of arguments.

    \ingroup libtensor_btod
 **/
template<size_t N, typename Traits>
class bto_dotprod : public timings< bto_dotprod<N, Traits> > {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_t;

    //! Type of block tensors
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_t;

public:
    static const char *k_clazz; //!< Class name

private:
    struct arg {
        block_tensor_t &bt1;
        block_tensor_t &bt2;
        permutation<N> perm1;
        permutation<N> perm2;

        arg(block_tensor_t &bt1_,
            block_tensor_t &bt2_) :
            bt1(bt1_), bt2(bt2_) { }

        arg(block_tensor_t &bt1_,
            const permutation<N> &perm1_,
            block_tensor_t &bt2_,
            const permutation<N> &perm2_) :
            bt1(bt1_), bt2(bt2_), perm1(perm1_), perm2(perm2_) { }
    };

    class dotprod_in_orbit_task:
        public libutil::task_i,
        public timings<dotprod_in_orbit_task> {

    public:
        static const char *k_clazz;

    private:
        block_tensor_t &m_bt1;
        const orbit_list<N, element_t> &m_ol1;
        permutation<N> m_pinv1;
        block_tensor_t &m_bt2;
        const orbit_list<N, element_t> &m_ol2;
        permutation<N> m_pinv2;
        const symmetry<N, element_t> &m_sym;
        dimensions<N> m_bidims;
        index<N> m_idx;
        element_t m_d;

    public:
        dotprod_in_orbit_task(block_tensor_t &bt1,
            const orbit_list<N, element_t> &ol1,
            const permutation<N> &pinv1,
            block_tensor_t &bt2,
            const orbit_list<N, element_t> &ol2,
            const permutation<N> &pinv2,
            const symmetry<N, element_t> &sym,
            const dimensions<N> &bidims, const index<N> &idx) :
            m_bt1(bt1), m_ol1(ol1), m_pinv1(pinv1),
            m_bt2(bt2), m_ol2(ol2), m_pinv2(pinv2),
            m_sym(sym), m_bidims(bidims), m_idx(idx), m_d(0.0) { }

        virtual ~dotprod_in_orbit_task() { }
        virtual void perform();

        element_t get_d() const { return m_d; }
    };

    class dotprod_task_iterator : public libutil::task_iterator_i {
    private:
        std::vector<dotprod_in_orbit_task*> &m_tl;
        typename std::vector<dotprod_in_orbit_task*>::iterator m_i;

    public:
        dotprod_task_iterator(std::vector<dotprod_in_orbit_task*> &tl) :
            m_tl(tl), m_i(m_tl.begin()) { }
        virtual ~dotprod_task_iterator() { }
        virtual bool has_more() const;
        virtual libutil::task_i *get_next();
    };

    class dotprod_task_observer : public libutil::task_observer_i {
    public:
        virtual void notify_start_task(libutil::task_i *t) { }
        virtual void notify_finish_task(libutil::task_i *t) { }
    };

private:
    block_index_space<N> m_bis; //!< Block %index space of arguments
    std::list<arg> m_args; //!< Arguments

public:
    /** \brief Initializes the first argument pair
            (identity permutation)
     **/
    bto_dotprod(block_tensor_t &bt1, block_tensor_t &bt2);

    /** \brief Initializes the first argument pair
     **/
    bto_dotprod(block_tensor_t &bt1, const permutation<N> &perm1,
            block_tensor_t &bt2, const permutation<N> &perm2);

    /** \brief Adds a pair of arguments (identity permutation)
     **/
    void add_arg(block_tensor_t &bt1,
        block_tensor_t &bt2);

    /** \brief Adds a pair of arguments
     **/
    void add_arg(block_tensor_t &bt1, const permutation<N> &perm1,
            block_tensor_t &bt2, const permutation<N> &perm2);

    /** \brief Returns the dot product of the first argument pair
     **/
    element_t calculate();

    /** \brief Computes the dot product for all argument pairs
     **/
    void calculate(std::vector<element_t> &v);

private:
    bto_dotprod(const bto_dotprod<N, Traits>&);
    const bto_dotprod<N, Traits> &operator=(const bto_dotprod<N, Traits>&);

};


} // namespace libtensor


#endif // LIBTENSOR_BTOD_DOTPROD_H
