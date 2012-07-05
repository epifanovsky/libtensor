#ifndef LIBTENSOR_TOD_APPLY_H
#define LIBTENSOR_TOD_APPLY_H

#include <libtensor/timings.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include <libtensor/mp/auto_cpu_lock.h>
#include <libtensor/tod/loop_list_apply.h>
#include <libtensor/tod/bad_dimensions.h>
#include "dense_tensor_ctrl.h"

namespace libtensor {


/** \brief Applies a functor to all tensor elements and scales / permutes them
        before, if necessary
    \tparam N Tensor order.

    This operation applies the given functor to each tensor element,
    transforming the tensor before and after applying the functor.
    The result can replace or be added to the output tensor.

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

    typedef tensor_transf<N, double> tensor_transf_t;

private:
    dense_tensor_rd_i<N, double> &m_ta; //!< Source %tensor
    Functor m_fn; //!< Functor
    tensor_transf<N, double> m_tr1; //!< Tensor transformation before
    tensor_transf<N, double> m_tr2; //!< Tensor transformation after
    dimensions<N> m_dimsb; //!< Dimensions of output %tensor

public:
    /** \brief Initializes the addition operation
        \param t First tensor in the series.
        \param tr1 Tensor transformation before
        \param tr2 Tensor transformation after
     **/
    tod_apply(dense_tensor_rd_i<N, double> &ta, const Functor &fn,
            const tensor_transf_t &tr1 = tensor_transf_t(),
            const tensor_transf_t &tr2 = tensor_transf_t());

    /** \brief Prepares the copy operation
        \param ta Source tensor.
        \param c Coefficient (apply before).
     **/
    tod_apply(dense_tensor_rd_i<N, double> &ta, const Functor &fn, double c);

    /** \brief Prepares the permute & copy operation
        \param ta Source tensor.
        \param p Permutation of tensor elements (apply before).
        \param c Coefficient (apply before).
     **/
    tod_apply(dense_tensor_rd_i<N, double> &ta, const Functor &fn,
        const permutation<N> &p, double c = 1.0);

    /** \brief Performs the operation
        \param cpus CPUs to perform the operation on
        \param zero Zero result first
        \param c Scaling factor
        \param tb Add result to
     **/
    void perform(cpu_pool &cpus, bool zero,
            double c, dense_tensor_wr_i<N, double> &t);

private:
    /** \brief Creates the dimensions of the output using an input
            tensor and a permutation of indexes
     **/
    static dimensions<N> mk_dimsb(dense_tensor_rd_i<N, double> &ta,
        const permutation<N> &perm1, const permutation<N> &perm2);

    void build_loop(typename loop_list_apply<Functor>::list_t &loop,
            const dimensions<N> &dimsa, const permutation<N> &perma,
            const dimensions<N> &dimsb);

};


template<size_t N, typename Functor>
const char *tod_apply<N, Functor>::k_clazz = "tod_apply<N, Functor>";


template<size_t N, typename Functor>
tod_apply<N, Functor>::tod_apply(dense_tensor_rd_i<N, double> &ta,
    const Functor &fn, const tensor_transf_t &tr1, const tensor_transf_t &tr2) :

    m_ta(ta), m_fn(fn), m_tr1(tr1), m_tr2(tr2),
    m_dimsb(mk_dimsb(m_ta, m_tr1.get_perm(), m_tr2.get_perm())) {

    m_tr1.permute(m_tr2.get_perm());
    m_tr2.get_perm().reset();
}


template<size_t N, typename Functor>
tod_apply<N, Functor>::tod_apply(dense_tensor_rd_i<N, double> &ta,
    const Functor &fn, double c) :

    m_ta(ta), m_fn(fn), m_tr1(permutation<N>(), scalar_transf<double>(c)),
    m_dimsb(m_ta.get_dims()) {

}


template<size_t N, typename Functor>
tod_apply<N, Functor>::tod_apply(dense_tensor_rd_i<N, double> &ta,
    const Functor &fn, const permutation<N> &p, double c) :

    m_ta(ta), m_fn(fn), m_tr1(p, scalar_transf<double>(c)),
    m_dimsb(mk_dimsb(ta, p, permutation<N>())) {

}


template<size_t N, typename Functor>
void tod_apply<N, Functor>::perform(cpu_pool &cpus, bool zero, double c,
    dense_tensor_wr_i<N, double> &tb) {

    static const char *method =
        "perform(cpu_pool&, bool, double, dense_tensor_wr_i<N, double>&)";

    if(!tb.get_dims().equals(m_dimsb)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }
    if(! zero && c == 0) return;

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
    build_loop(loop, dimsa, m_tr1.get_perm(), dimsb);

    const double *pa = ca.req_const_dataptr();
    double *pb = cb.req_dataptr();

    {
        auto_cpu_lock cpu(cpus);

        registers_t r;
        r.m_ptra[0] = pa;
        r.m_ptrb[0] = pb;
        r.m_ptra_end[0] = pa + dimsa.get_size();
        r.m_ptrb_end[0] = pb + dimsb.get_size();

        loop_list_apply<Functor>::run_loop(loop, r, m_fn,
                c * m_tr2.get_scalar_tr().get_coeff(),
                m_tr1.get_scalar_tr().get_coeff(), !zero);
    }

    ca.ret_const_dataptr(pa);
    cb.ret_dataptr(pb);

    } catch(...) {
        tod_apply<N, Functor>::stop_timer();
        throw;
    }
    tod_apply<N, Functor>::stop_timer();
}


template<size_t N, typename Functor>
dimensions<N> tod_apply<N, Functor>::mk_dimsb(dense_tensor_rd_i<N, double> &ta,
    const permutation<N> &perm1, const permutation<N> &perm2) {

    dimensions<N> dims(ta.get_dims());
    dims.permute(perm1);
    dims.permute(perm2);
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
