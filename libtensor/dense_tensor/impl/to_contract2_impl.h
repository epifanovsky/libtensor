#ifndef LIBTENSOR_TO_CONTRACT2_IMPL_H
#define LIBTENSOR_TO_CONTRACT2_IMPL_H

#include <cstring> // for memset
#include <memory>
#include <libtensor/core/allocator.h>
#include <libtensor/core/bad_dimensions.h>
#include <libtensor/core/contraction2_align.h>
#include <libtensor/core/contraction2_list_builder.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_add1.h>
#include <libtensor/kernels/kern_copy.h>
#include <libtensor/kernels/kern_mul2.h>
#include <libtensor/kernels/loop_list_node.h>
#include <libtensor/kernels/loop_list_runner.h>
#include "../dense_tensor.h"
#include "../dense_tensor_ctrl.h"
#include "../to_contract2.h"


namespace libtensor {


template<size_t N, size_t M, size_t K, typename T>
const char *to_contract2<N, M, K, T>::k_clazz = "to_contract2<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
to_contract2<N, M, K, T>::to_contract2(
    const contraction2<N, M, K> &contr,
    dense_tensor_rd_i<k_ordera, T> &ta,
    const scalar_transf<T> &ka,
    dense_tensor_rd_i<k_orderb, T> &tb,
    const scalar_transf<T> &kb,
    const scalar_transf<T> &kc) :

    m_dimsc(contr, ta.get_dims(), tb.get_dims()) {

    add_args(contr, ta, ka, tb, kb, kc);
}


template<size_t N, size_t M, size_t K, typename T>
to_contract2<N, M, K, T>::to_contract2(
    const contraction2<N, M, K> &contr,
    dense_tensor_rd_i<k_ordera, T> &ta,
    dense_tensor_rd_i<k_orderb, T> &tb,
    T d) :

    m_dimsc(contr, ta.get_dims(), tb.get_dims()) {

    add_args(contr, ta, tb, d);
}


template<size_t N, size_t M, size_t K, typename T>
inline void to_contract2<N, M, K, T>::add_args(
    const contraction2<N, M, K> &contr,
    dense_tensor_rd_i<k_ordera, T> &ta,
    const scalar_transf<T> &ka,
    dense_tensor_rd_i<k_orderb, T> &tb,
    const scalar_transf<T> &kb,
    const scalar_transf<T> &kc) {

    T d = ka.get_coeff() * kb.get_coeff() * kc.get_coeff();
    add_args(contr, ta, tb, d);
}


template<size_t N, size_t M, size_t K, typename T>
void to_contract2<N, M, K, T>::add_args(
    const contraction2<N, M, K> &contr,
    dense_tensor_rd_i<k_ordera, T> &ta,
    dense_tensor_rd_i<k_orderb, T> &tb,
    T d) {

    static const char *method = "add_args(const contraction2<N, M, K>&, "
        "dense_tensor_i<N + K, T>&, dense_tensor_i<M + K, T>&, "
        "T)";

    if(!to_contract2_dims<N, M, K>(contr, ta.get_dims(), tb.get_dims()).
        get_dims().equals(m_dimsc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "ta,tb");
    }

    m_argslst.push_back(args(contr, ta, tb, d));
}


template<size_t N, size_t M, size_t K, typename T>
void to_contract2<N, M, K, T>::prefetch() {

    for(typename std::list<args>::iterator i = m_argslst.begin();
        i != m_argslst.end(); ++i) {

        dense_tensor_rd_ctrl<k_ordera, T>(i->ta).req_prefetch();
        dense_tensor_rd_ctrl<k_orderb, T>(i->tb).req_prefetch();
    }
}


template<size_t N, size_t M, size_t K, typename T>
void to_contract2<N, M, K, T>::perform(bool zero,
        dense_tensor_wr_i<k_orderc, T> &tc) {

    static const char *method =
        "perform(bool, dense_tensor_i<N + M, T>&)";

    if(!m_dimsc.get_dims().equals(tc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tc");
    }

    to_contract2<N, M, K, T>::start_timer();

    try {

        dense_tensor_wr_ctrl<k_orderc, T> cc(tc);
        T *pc = cc.req_dataptr();
        const dimensions<k_orderc> &dimsc = tc.get_dims();

        //  Pre-process the arguments by aligning indexes

        to_contract2<N, M, K, T>::start_timer("align");
        std::list<aligned_args> argslst;
        for(typename std::list<args>::iterator i = m_argslst.begin();
            i != m_argslst.end(); ++i) {

        	if (i->d == 0.0) continue;

        	contraction2_align<N, M, K> align(i->contr);
        	argslst.push_back(aligned_args(*i,
        			align.get_perma(), align.get_permb(), align.get_permc()));
        }
        to_contract2<N, M, K, T>::stop_timer("align");

        //  Special case when no calculation is required

        if(argslst.empty() && zero) {
            to_contract2<N, M, K, T>::start_timer("zeroc");
            memset(pc, 0, sizeof(T) * dimsc.get_size());
            to_contract2<N, M, K, T>::stop_timer("zeroc");
        }

        //  Compute the contractions grouping them by the permutation of C

        bool zero1 = zero;
        T *pc1 = 0, *pc2 = 0;
        typename allocator::pointer_type vpc;
        vpc = allocator::allocate(dimsc.get_size());
        pc1 = (T*)allocator::lock_rw(vpc);

        while(!argslst.empty()) {

            typename std::list<aligned_args>::iterator iarg = argslst.begin();
            permutation<k_orderc> permc(iarg->permc);
            permutation<k_orderc> pinvc(permc, true);
            dimensions<k_orderc> dimsc1(dimsc); dimsc1.permute(permc);

            if(iarg->permc.is_identity()) {
                pc2 = pc;
                if(zero1) {
                    to_contract2<N, M, K, T>::start_timer("zeroc");
                    memset(pc, 0, sizeof(T) * dimsc.get_size());
                    zero1 = false;
                    to_contract2<N, M, K, T>::stop_timer("zeroc");
                }
            } else {
                pc2 = pc1;
                to_contract2<N, M, K, T>::start_timer("zeroc1");
                memset(pc1, 0, sizeof(T) * dimsc1.get_size());
                to_contract2<N, M, K, T>::stop_timer("zeroc1");
            }

            do {
                if(iarg->permc.equals(permc)) {
                    perform_internal(*iarg, pc2, dimsc1);
                    iarg = argslst.erase(iarg);
                } else {
                    ++iarg;
                }
            } while(iarg != argslst.end());

            if(pc2 == pc1) {

                sequence<k_orderc, size_t> seqc1(0);
                for(size_t i = 0; i < k_orderc; i++) seqc1[i] = i;
                pinvc.apply(seqc1);

                std::list< loop_list_node<1, 1> > loop_in, loop_out;
                typename std::list< loop_list_node<1, 1> >::iterator inode =
                    loop_in.end();

                for(size_t idxc = 0; idxc < k_orderc;) {
                    size_t len = 1;
                    size_t idxc1 = seqc1[idxc];
                    do {
                        len *= dimsc1.get_dim(idxc1);
                        idxc1++; idxc++;
                    } while(idxc < k_orderc && seqc1[idxc] == idxc1);

                    inode = loop_in.insert(loop_in.end(),
                        loop_list_node<1, 1>(len));
                    inode->stepa(0) = dimsc1.get_increment(idxc1 - 1);
                    inode->stepb(0) = dimsc.get_increment(idxc - 1);
                }

                loop_registers_x<1, 1, T> r;
                r.m_ptra[0] = pc1;
                r.m_ptrb[0] = pc;
                r.m_ptra_end[0] = pc1 + dimsc1.get_size();
                r.m_ptrb_end[0] = pc + dimsc.get_size();

                {
                    std::unique_ptr< kernel_base<linalg, 1, 1, T> > kern(
                        zero1 ?
                            kern_copy<linalg, T>::match(1.0, loop_in, loop_out) :
                            kern_add1<linalg, T>::match(1.0, loop_in, loop_out));
                    to_contract2<N, M, K, T>::start_timer("permc");
                    to_contract2<N, M, K, T>::start_timer(kern->get_name());
                    loop_list_runner_x<linalg, 1, 1, T>(loop_in).run(0, r, *kern);
                    to_contract2<N, M, K, T>::stop_timer(kern->get_name());
                    to_contract2<N, M, K, T>::stop_timer("permc");
                    zero1 = false;
                }
            }
        }

        allocator::unlock_rw(vpc); pc1 = 0;
        allocator::deallocate(vpc);

        cc.ret_dataptr(pc); pc = 0;

    } catch(...) {
        to_contract2<N, M, K, T>::stop_timer();
        throw;
    }

    to_contract2<N, M, K, T>::stop_timer();
}


template<size_t N, size_t M, size_t K, typename T>
void to_contract2<N, M, K, T>::perform_internal(aligned_args &ar,
    T *pc, const dimensions<k_orderc> &dimsc) {

    dense_tensor_rd_ctrl<k_ordera, T> ca(ar.ta);
    dense_tensor_rd_ctrl<k_orderb, T> cb(ar.tb);

    const dimensions<k_ordera> &dimsa = ar.ta.get_dims();
    const dimensions<k_orderb> &dimsb = ar.tb.get_dims();

    dimensions<k_ordera> dimsa1(dimsa); dimsa1.permute(ar.perma);
    dimensions<k_orderb> dimsb1(dimsb); dimsb1.permute(ar.permb);

    const T *pa = 0, *pb = 0;
    T *pa1 = 0, *pb1 = 0;
    const T *pa2 = 0, *pb2 = 0;

    typename allocator::pointer_type vpa, vpb;

    pa2 = pa = ca.req_const_dataptr();
    if(!ar.perma.is_identity()) {

        vpa = allocator::allocate(dimsa1.get_size());
        pa1 = (T*)allocator::lock_rw(vpa);

        sequence<k_ordera, size_t> seqa(0);
        for(size_t i = 0; i < k_ordera; i++) seqa[i] = i;
        ar.perma.apply(seqa);

        std::list< loop_list_node<1, 1> > loop_in, loop_out;
        typename std::list< loop_list_node<1, 1> >::iterator inode =
            loop_in.end();

        for(size_t idxa1 = 0; idxa1 < k_ordera;) {
            size_t len = 1;
            size_t idxa = seqa[idxa1];
            do {
                len *= dimsa.get_dim(idxa);
                idxa++; idxa1++;
            } while(idxa1 < k_ordera && seqa[idxa1] == idxa);

            inode = loop_in.insert(loop_in.end(), loop_list_node<1, 1>(len));
            inode->stepa(0) = dimsa.get_increment(idxa - 1);
            inode->stepb(0) = dimsa1.get_increment(idxa1 - 1);
        }

        loop_registers_x<1, 1, T> r;
        r.m_ptra[0] = pa;
        r.m_ptrb[0] = pa1;
        r.m_ptra_end[0] = pa + dimsa.get_size();
        r.m_ptrb_end[0] = pa1 + dimsa1.get_size();

        {
            std::unique_ptr< kernel_base<linalg, 1, 1, T> >kern(
                kern_copy<linalg, T>::match(1.0, loop_in, loop_out));
            to_contract2<N, M, K, T>::start_timer("perma");
            to_contract2<N, M, K, T>::start_timer(kern->get_name());
            loop_list_runner_x<linalg, 1, 1, T>(loop_in).run(0, r, *kern);
            to_contract2<N, M, K, T>::stop_timer(kern->get_name());
            to_contract2<N, M, K, T>::stop_timer("perma");
        }

        pa2 = pa1;
    }

    pb2 = pb = cb.req_const_dataptr();
    if(!ar.permb.is_identity()) {

        vpb = allocator::allocate(dimsb1.get_size());
        pb1 = (T*)allocator::lock_rw(vpb);

        sequence<k_orderb, size_t> seqb(0);
        for(size_t i = 0; i < k_orderb; i++) seqb[i] = i;
        ar.permb.apply(seqb);

        std::list< loop_list_node<1, 1> > loop_in, loop_out;
        typename std::list< loop_list_node<1, 1> >::iterator inode =
            loop_in.end();

        for(size_t idxb1 = 0; idxb1 < k_orderb;) {
            size_t len = 1;
            size_t idxb = seqb[idxb1];
            do {
                len *= dimsb.get_dim(idxb);
                idxb++; idxb1++;
            } while(idxb1 < k_orderb && seqb[idxb1] == idxb);

            inode = loop_in.insert(loop_in.end(), loop_list_node<1, 1>(len));
            inode->stepa(0) = dimsb.get_increment(idxb - 1);
            inode->stepb(0) = dimsb1.get_increment(idxb1 - 1);
        }

        loop_registers_x<1, 1, T> r;
        r.m_ptra[0] = pb;
        r.m_ptrb[0] = pb1;
        r.m_ptra_end[0] = pb + dimsb.get_size();
        r.m_ptrb_end[0] = pb1 + dimsb1.get_size();

        {
            std::unique_ptr< kernel_base<linalg, 1, 1, T> >kern(
                kern_copy<linalg, T>::match(1.0, loop_in, loop_out));
            to_contract2<N, M, K, T>::start_timer("permb");
            to_contract2<N, M, K, T>::start_timer(kern->get_name());
            loop_list_runner_x<linalg, 1, 1, T>(loop_in).run(0, r, *kern);
            to_contract2<N, M, K, T>::stop_timer(kern->get_name());
            to_contract2<N, M, K, T>::stop_timer("permb");
        }

        pb2 = pb1;
    }

    contraction2<N, M, K> contr1(ar.contr);
    contr1.permute_a(ar.perma);
    contr1.permute_b(ar.permb);
    contr1.permute_c(ar.permc);

    std::list< loop_list_node<2, 1> > loop_in, loop_out;
    loop_list_adapter list_adapter(loop_in);
    contraction2_list_builder<N, M, K>(contr1).
        populate(list_adapter, dimsa1, dimsb1, dimsc);

    {
        loop_registers_x<2, 1, T> r;
        r.m_ptra[0] = pa2;
        r.m_ptra[1] = pb2;
        r.m_ptrb[0] = pc;
        r.m_ptra_end[0] = pa2 + dimsa1.get_size();
        r.m_ptra_end[1] = pb2 + dimsb1.get_size();
        r.m_ptrb_end[0] = pc + dimsc.get_size();

        std::unique_ptr< kernel_base<linalg, 2, 1, T> > kern(
            kern_mul2<linalg, T>::match(ar.d, loop_in, loop_out));
        to_contract2<N, M, K, T>::start_timer("kernel");
        to_contract2<N, M, K, T>::start_timer(kern->get_name());
        loop_list_runner_x<linalg, 2, 1, T>(loop_in).run(0, r, *kern);
        to_contract2<N, M, K, T>::stop_timer("kernel");
        to_contract2<N, M, K, T>::stop_timer(kern->get_name());
    }

    if(pa1) {
        allocator::unlock_rw(vpa); pa1 = 0;
        allocator::deallocate(vpa);
    }
    ca.ret_const_dataptr(pa);

    if(pb1) {
        allocator::unlock_rw(vpb); pb1 = 0;
        allocator::deallocate(vpb);
    }
    cb.ret_const_dataptr(pb);
}


} // namespace libtensor

#endif // LIBTENSOR_TO_CONTRACT2_IMPL_H
