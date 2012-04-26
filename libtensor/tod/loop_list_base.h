#ifndef LIBTENSOR_LOOP_LIST_BASE_H
#define LIBTENSOR_LOOP_LIST_BASE_H

#include <list>

namespace libtensor {


/** \brief Base driver for nested loops
    \tparam N Number of input arrays.
    \tparam M Number of output arrays.
    \tparam Impl Loop implementation.

    \ingroup libtensor_tod
 **/
template<size_t N, size_t M, typename Impl>
class loop_list_base {
protected:
    /** \brief Structure keeps track of the current location in arrays
     **/
    struct registers {
        const double *m_ptra[N]; //!< Position in argument arrays
        double *m_ptrb[M]; //!< Position in result arrays
        const double *m_ptra_end[N]; //!< End of argument arrays
        double *m_ptrb_end[M]; //!< End of result arrays
    };

    /** \brief Node on the list of nested loops
     **/
    struct node {
    public:
        typedef void (Impl::*fnptr_t)(registers&) const;

    private:
        size_t m_weight; //!< Number of iterations in the loop
        fnptr_t m_fn; //!< Kernel function
        size_t m_stepa[N]; //!< Increments in the argument arrays
        size_t m_stepb[M]; //!< Increments in the result arrays

    public:
        /** \brief Default constructor
         **/
        node() : m_weight(0), m_fn(0) {
            init_incs();
        }

        /** \brief Initializing constructor
         **/
        node(size_t weight) : m_weight(weight), m_fn(0) {
            init_incs();
        }

        size_t &weight() { return m_weight; }
        fnptr_t &fn() { return m_fn; }
        size_t &stepa(size_t i) { return m_stepa[i]; }
        size_t &stepb(size_t i) { return m_stepb[i]; }

    private:
        void init_incs() {
            for(register size_t i = 0; i < N; i++) m_stepa[i] = 0;
            for(register size_t i = 0; i < M; i++) m_stepb[i] = 0;
        }
    };

    typedef std::list<node> list_t; //!< List of nested loops (type)
    typedef typename std::list<node>::iterator
        iterator_t; //!< List iterator type

protected:
    void exec(Impl &impl, iterator_t &i, iterator_t &iend, registers &r);

private:
    void fn_loop(Impl &impl, iterator_t &i, iterator_t &iend, registers &r);
};


template<size_t N, size_t M, typename Impl>
void loop_list_base<N, M, Impl>::exec(Impl &impl, iterator_t &i,
    iterator_t &iend, registers &r) {

    typename node::fnptr_t fn = i->fn();

    if(fn == 0) fn_loop(impl, i, iend, r);
    else (impl.*fn)(r);
}


template<size_t N, size_t M, typename Impl>
void loop_list_base<N, M, Impl>::fn_loop(Impl &impl, iterator_t &i,
    iterator_t &iend, registers &r) {

    iterator_t j = i; j++;
    if(j == iend) return;

    const double *ptra[N];
    double *ptrb[M];
    for(register size_t ii = 0; ii < N; ii++) ptra[ii] = r.m_ptra[ii];
    for(register size_t ii = 0; ii < M; ii++) ptrb[ii] = r.m_ptrb[ii];

    for(size_t k = 0; k < i->weight(); k++) {

        exec(impl, j, iend, r);
        for(register size_t ii = 0; ii < N; ii++) {
            ptra[ii] += i->stepa(ii); r.m_ptra[ii] = ptra[ii];
        }
        for(register size_t ii = 0; ii < M; ii++) {
            ptrb[ii] += i->stepb(ii); r.m_ptrb[ii] = ptrb[ii];
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_LOOP_LIST_BASE_H
