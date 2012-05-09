#ifndef LIBTENSOR_TOD_APPLY_H
#define LIBTENSOR_TOD_APPLY_H

#include <libtensor/timings.h>
#include <libtensor/tod/loop_list_apply.h>
#include <libtensor/tod/bad_dimensions.h>
#include "dense_tensor_ctrl.h"

namespace libtensor {


/** \brief Applies a functor to all tensor elements and scales / permutes them
        before, if necessary
    \tparam N Tensor order.

    This operation applies the given functor to each tensor element, scaling
    and permuting them before. The result can replace or be added to the
    output tensor.

    A class to be used as functor needs to have
    1. a proper copy constructor
    \code
        Functor(const Functor &f);
    \endcode
    2. an implementation of the function
    \code
        double Functor::operator()(const double &a);
    \endcode

    The latter function should perform the intended operation of the functor
    on the tensor data.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N, typename Functor>
class tod_apply :
    public loop_list_apply<Functor>,
    public timings< tod_apply<N, Functor> > {

public:
    static const char *k_clazz; //!< Class name

private:
    dense_tensor_rd_i<N, double> &m_ta; //!< Source %tensor
    Functor m_fn; //!< Functor
    permutation<N> m_perm; //!< Permutation of elements
    double m_c; //!< Scaling coefficient
    dimensions<N> m_dimsb; //!< Dimensions of output %tensor

public:
    /** \brief Prepares the copy operation
        \param ta Source tensor.
        \param c Coefficient.
     **/
    tod_apply(dense_tensor_rd_i<N, double> &ta, const Functor &fn,
        double c = 1.0);

    /** \brief Prepares the permute & copy operation
        \param ta Source tensor.
        \param p Permutation of tensor elements.
        \param c Coefficient.
     **/
    tod_apply(dense_tensor_rd_i<N, double> &ta, const Functor &fn,
        const permutation<N> &p, double c = 1.0);

    void perform(bool zero, double c, dense_tensor_wr_i<N, double> &t);

    void perform(dense_tensor_wr_i<N, double> &t);
    void perform(dense_tensor_wr_i<N, double> &t, double c);

private:
    /** \brief Creates the dimensions of the output using an input
            tensor and a permutation of indexes
     **/
    static dimensions<N> mk_dimsb(dense_tensor_rd_i<N, double> &ta,
        const permutation<N> &perm);

    void build_loop(typename loop_list_apply<Functor>::list_t &loop,
            const dimensions<N> &dimsa, const permutation<N> &perma,
            const dimensions<N> &dimsb);

};


template<size_t N, typename Functor>
const char *tod_apply<N, Functor>::k_clazz = "tod_apply<N, Functor>";


template<size_t N, typename Functor>
tod_apply<N, Functor>::tod_apply(dense_tensor_rd_i<N, double> &ta,
    const Functor &fn, double c) :

    m_ta(ta), m_fn(fn), m_c(c), m_dimsb(mk_dimsb(m_ta, m_perm)) {

}


template<size_t N, typename Functor>
tod_apply<N, Functor>::tod_apply(dense_tensor_rd_i<N, double> &ta,
    const Functor &fn, const permutation<N> &p, double c) :

    m_ta(ta), m_fn(fn), m_perm(p), m_c(c), m_dimsb(mk_dimsb(ta, p)) {

}


template<size_t N, typename Functor>
void tod_apply<N, Functor>::perform(bool zero, double c,
    dense_tensor_wr_i<N, double> &tb) {

    static const char *method =
        "perform(bool, double, dense_tensor_wr_i<N, double>&)";

    if(!tb.get_dims().equals(m_dimsb)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }
    if(!zero && c == 0) return;

    typedef typename loop_list_apply<Functor>::list_t list_t;
    typedef typename loop_list_apply<Functor>::registers registers_t;
    typedef typename loop_list_apply<Functor>::node node_t;

    tod_apply<N, Functor>::start_timer();

    try {

    dense_tensor_rd_ctrl<N, double> ca(m_ta);
    dense_tensor_wr_ctrl<N, double> cb(tb);
    ca.req_prefetch();
    cb.req_prefetch();

    const dimensions<N> &dimsa = m_ta.get_dims();
    const dimensions<N> &dimsb = tb.get_dims();

    list_t loop;
    build_loop(loop, dimsa, m_perm, dimsb);

    const double *pa = ca.req_const_dataptr();
    double *pb = cb.req_dataptr();

    registers_t r;
    r.m_ptra[0] = pa;
    r.m_ptrb[0] = pb;
    r.m_ptra_end[0] = pa + dimsa.get_size();
    r.m_ptrb_end[0] = pb + dimsb.get_size();

    loop_list_apply<Functor>::run_loop(loop, r, m_fn, c, m_c, !zero);

    ca.ret_const_dataptr(pa);
    cb.ret_dataptr(pb);

    } catch(...) {
        tod_apply<N, Functor>::stop_timer();
        throw;
    }
    tod_apply<N, Functor>::stop_timer();
}


template<size_t N, typename Functor>
void tod_apply<N, Functor>::perform(dense_tensor_wr_i<N, double> &tb) {

    perform(true, 1.0, tb);
}


template<size_t N, typename Functor>
void tod_apply<N, Functor>::perform(dense_tensor_wr_i<N, double> &tb,
    double c) {

    perform(false, c, tb);
}


template<size_t N, typename Functor>
dimensions<N> tod_apply<N, Functor>::mk_dimsb(
    dense_tensor_rd_i<N, double> &ta, const permutation<N> &perm) {

    dimensions<N> dims(ta.get_dims());
    dims.permute(perm);
    return dims;
}


template<size_t N, typename Functor>
void tod_apply<N, Functor>::build_loop(
    typename loop_list_apply<Functor>::list_t &loop,
    const dimensions<N> &dimsa, const permutation<N> &perma,
    const dimensions<N> &dimsb) {

    typedef typename loop_list_apply<Functor>::iterator_t iterator_t;
    typedef typename loop_list_apply<Functor>::node node_t;

    sequence<N, size_t> map;
    for(register size_t i = 0; i < N; i++) map[i] = i;
    perma.apply(map);

    //
    //    Go over indexes in B and connect them with indexes in A
    //    trying to glue together consecutive indexes
    //
    for(size_t idxb = 0; idxb < N;) {
        size_t len = 1;
        size_t idxa = map[idxb];
        do {
            len *= dimsa.get_dim(idxa);
            idxa++; idxb++;
        } while(idxb < N && map[idxb] == idxa);

        iterator_t inode = loop.insert(loop.end(), node_t(len));
        inode->stepa(0) = dimsa.get_increment(idxa - 1);
        inode->stepb(0) = dimsb.get_increment(idxb - 1);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_APPLY_H
