#ifndef LIBTENSOR_TOD_COPY_IMPL_H
#define LIBTENSOR_TOD_COPY_IMPL_H

#include <libtensor/tod/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_set.h"
#include "../tod_copy.h"


namespace libtensor {


template<size_t N>
const char *tod_copy<N>::k_clazz = "tod_copy<N>";


template<size_t N>
tod_copy<N>::tod_copy(dense_tensor_rd_i<N, double> &ta, double c) :

    m_ta(ta), m_tr(permutation<N>(), scalar_transf<double>(c)),
    m_dimsb(mk_dimsb(m_ta, m_tr.get_perm())) {

}


template<size_t N>
tod_copy<N>::tod_copy(dense_tensor_rd_i<N, double> &ta, const permutation<N> &p,
    double c) :

    m_ta(ta), m_tr(p, scalar_transf<double>(c)), m_dimsb(mk_dimsb(ta, p)) {

}


template<size_t N>
tod_copy<N>::tod_copy(dense_tensor_rd_i<N, double> &ta,
        const tensor_transf<N, double> &tr) :

    m_ta(ta), m_tr(tr), m_dimsb(mk_dimsb(ta, m_tr.get_perm())) {

}


template<size_t N>
void tod_copy<N>::prefetch() {

    dense_tensor_rd_ctrl<N, double>(m_ta).req_prefetch();
}


template<size_t N>
void tod_copy<N>::perform(bool zero, double c,
    dense_tensor_wr_i<N, double> &tb) {

    static const char *method =
        "perform(bool, double, dense_tensor_wr_i<N, double>&)";

    if(!tb.get_dims().equals(m_dimsb)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }

    //  Special case
    if(c == 0.0) {
        if(zero) tod_set<N>().perform(tb);
        return;
    }

    //  Run copy/add loop
    if(zero) do_perform<loop_list_copy>(c, tb);
    else do_perform<loop_list_add>(c, tb);
}


template<size_t N>
dimensions<N> tod_copy<N>::mk_dimsb(dense_tensor_rd_i<N, double> &ta,
    const permutation<N> &perm) {

    dimensions<N> dims(ta.get_dims());
    dims.permute(perm);
    return dims;
}


template<size_t N> template<typename Base>
void tod_copy<N>::do_perform(double c, dense_tensor_wr_i<N, double> &tb) {

    typedef typename Base::list_t list_t;
    typedef typename Base::registers registers_t;
    typedef typename Base::node node_t;

    tod_copy<N>::start_timer();

    try {

    dense_tensor_rd_ctrl<N, double> ca(m_ta);
    dense_tensor_wr_ctrl<N, double> cb(tb);
    ca.req_prefetch();
    cb.req_prefetch();

    const dimensions<N> &dimsa = m_ta.get_dims();
    const dimensions<N> &dimsb = tb.get_dims();

    const double *pa = ca.req_const_dataptr();
    double *pb = cb.req_dataptr();

    list_t loop;
    build_loop<Base>(loop, dimsa, m_tr.get_perm(), dimsb);

    registers_t r;
    r.m_ptra[0] = pa;
    r.m_ptrb[0] = pb;
    r.m_ptra_end[0] = pa + dimsa.get_size();
    r.m_ptrb_end[0] = pb + dimsb.get_size();

    Base::run_loop(loop, r, m_tr.get_scalar_tr().get_coeff() * c);

    ca.ret_const_dataptr(pa);
    cb.ret_dataptr(pb);

    } catch(...) {
        tod_copy<N>::stop_timer();
        throw;
    }
    tod_copy<N>::stop_timer();
}


template<size_t N> template<typename Base>
void tod_copy<N>::build_loop(typename Base::list_t &loop,
    const dimensions<N> &dimsa, const permutation<N> &perma,
    const dimensions<N> &dimsb) {

    typedef typename Base::iterator_t iterator_t;
    typedef typename Base::node node_t;

    sequence<N, size_t> map(0);
    for(register size_t i = 0; i < N; i++) map[i] = i;
    perma.apply(map);

    //
    //  Go over indexes in B and connect them with indexes in A
    //  trying to glue together consecutive indexes
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

#endif // LIBTENSOR_TOD_COPY_IMPL_H
