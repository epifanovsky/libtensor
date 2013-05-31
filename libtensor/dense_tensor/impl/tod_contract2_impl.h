#ifndef LIBTENSOR_TOD_CONTRACT2_IMPL_H
#define LIBTENSOR_TOD_CONTRACT2_IMPL_H

#include <cstring> // for memset
#include <memory>
#include <libtensor/core/allocator.h>
#include <libtensor/core/bad_dimensions.h>
#include <libtensor/core/contraction2_list_builder.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_dadd1.h>
#include <libtensor/kernels/kern_dcopy.h>
#include <libtensor/kernels/kern_dmul2.h>
#include <libtensor/kernels/loop_list_node.h>
#include <libtensor/kernels/loop_list_runner.h>
#include "../dense_tensor.h"
#include "../dense_tensor_ctrl.h"
#include "../tod_contract2.h"


namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::k_clazz = "tod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
tod_contract2<N, M, K>::tod_contract2(
    const contraction2<N, M, K> &contr,
    dense_tensor_rd_i<k_ordera, double> &ta,
    const scalar_transf<double> &ka,
    dense_tensor_rd_i<k_orderb, double> &tb,
    const scalar_transf<double> &kb,
    const scalar_transf<double> &kc) :

    m_dimsc(contr, ta.get_dims(), tb.get_dims()) {

    add_args(contr, ta, ka, tb, kb, kc);
}


template<size_t N, size_t M, size_t K>
tod_contract2<N, M, K>::tod_contract2(
    const contraction2<N, M, K> &contr,
    dense_tensor_rd_i<k_ordera, double> &ta,
    dense_tensor_rd_i<k_orderb, double> &tb,
    double d) :

    m_dimsc(contr, ta.get_dims(), tb.get_dims()) {

    add_args(contr, ta, tb, d);
}


template<size_t N, size_t M, size_t K>
inline void tod_contract2<N, M, K>::add_args(
    const contraction2<N, M, K> &contr,
    dense_tensor_rd_i<k_ordera, double> &ta,
    const scalar_transf<double> &ka,
    dense_tensor_rd_i<k_orderb, double> &tb,
    const scalar_transf<double> &kb,
    const scalar_transf<double> &kc) {

    double d = ka.get_coeff() * kb.get_coeff() * kc.get_coeff();
    add_args(contr, ta, tb, d);
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::add_args(
    const contraction2<N, M, K> &contr,
    dense_tensor_rd_i<k_ordera, double> &ta,
    dense_tensor_rd_i<k_orderb, double> &tb,
    double d) {

    static const char *method = "add_args(const contraction2<N, M, K>&, "
        "dense_tensor_i<N + K, double>&, dense_tensor_i<M + K, double>&, "
        "double)";

    if(!to_contract2_dims<N, M, K>(contr, ta.get_dims(), tb.get_dims()).
        get_dims().equals(m_dimsc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "ta,tb");
    }

    m_argslst.push_back(args(contr, ta, tb, d));
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::prefetch() {

    for(typename std::list<args>::iterator i = m_argslst.begin();
        i != m_argslst.end(); ++i) {

        dense_tensor_rd_ctrl<k_ordera, double>(i->ta).req_prefetch();
        dense_tensor_rd_ctrl<k_orderb, double>(i->tb).req_prefetch();
    }
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::perform(bool zero,
        dense_tensor_wr_i<k_orderc, double> &tc) {

    static const char *method =
        "perform(bool, dense_tensor_i<N + M, double>&)";

    if(!m_dimsc.get_dims().equals(tc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tc");
    }

    tod_contract2<N, M, K>::start_timer();

    try {

        dense_tensor_wr_ctrl<k_orderc, double> cc(tc);
        double *pc = cc.req_dataptr();
        const dimensions<k_orderc> &dimsc = tc.get_dims();

        //  Pre-process the arguments by aligning indexes

        tod_contract2<N, M, K>::start_timer("align");
        std::list<aligned_args> argslst;
        for(typename std::list<args>::iterator i = m_argslst.begin();
            i != m_argslst.end(); ++i) {
            aligned_args ar(*i);
            align(i->contr.get_conn(), ar.perma, ar.permb, ar.permc);
            if(ar.d != 0.0) argslst.push_back(ar);
        }
        tod_contract2<N, M, K>::stop_timer("align");

        //  Special case when no calculation is required

        if(argslst.empty() && zero) {
            tod_contract2<N, M, K>::start_timer("zeroc");
            memset(pc, 0, sizeof(double) * dimsc.get_size());
            tod_contract2<N, M, K>::stop_timer("zeroc");
        }

        //  Compute the contractions grouping them by the permutation of C

        bool zero1 = zero;
        double *pc1 = 0, *pc2 = 0;
        typename allocator<double>::pointer_type vpc;
        vpc = allocator<double>::allocate(dimsc.get_size());
        pc1 = allocator<double>::lock_rw(vpc);

        while(!argslst.empty()) {

            typename std::list<aligned_args>::iterator iarg = argslst.begin();
            permutation<k_orderc> permc(iarg->permc);
            permutation<k_orderc> pinvc(permc, true);
            dimensions<k_orderc> dimsc1(dimsc); dimsc1.permute(permc);

            if(iarg->permc.is_identity()) {
                pc2 = pc;
                if(zero1) {
                    tod_contract2<N, M, K>::start_timer("zeroc");
                    memset(pc, 0, sizeof(double) * dimsc.get_size());
                    zero1 = false;
                    tod_contract2<N, M, K>::stop_timer("zeroc");
                }
            } else {
                pc2 = pc1;
                tod_contract2<N, M, K>::start_timer("zeroc1");
                memset(pc1, 0, sizeof(double) * dimsc1.get_size());
                tod_contract2<N, M, K>::stop_timer("zeroc1");
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

                loop_registers<1, 1> r;
                r.m_ptra[0] = pc1;
                r.m_ptrb[0] = pc;
                r.m_ptra_end[0] = pc1 + dimsc1.get_size();
                r.m_ptrb_end[0] = pc + dimsc.get_size();

                {
                    std::auto_ptr< kernel_base<linalg, 1, 1> > kern(
                        zero1 ?
                            kern_dcopy<linalg>::match(1.0, loop_in, loop_out) :
                            kern_dadd1<linalg>::match(1.0, loop_in, loop_out));
                    tod_contract2<N, M, K>::start_timer("permc");
                    tod_contract2<N, M, K>::start_timer(kern->get_name());
                    loop_list_runner<linalg, 1, 1>(loop_in).run(0, r, *kern);
                    tod_contract2<N, M, K>::stop_timer(kern->get_name());
                    tod_contract2<N, M, K>::stop_timer("permc");
                    zero1 = false;
                }
            }
        }

        allocator<double>::unlock_rw(vpc); pc1 = 0;
        allocator<double>::deallocate(vpc);

        cc.ret_dataptr(pc); pc = 0;

    } catch(...) {
        tod_contract2<N, M, K>::stop_timer();
        throw;
    }

    tod_contract2<N, M, K>::stop_timer();
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::align(
    const sequence<2 * (N + M + K), size_t> &conn,
    permutation<N + K> &perma, permutation<M + K> &permb,
    permutation<N + M> &permc) {

    //  This algorithm reorders indexes in A, B, C so that the whole contraction
    //  can be done in a single matrix multiplication.
    //  Returned permutations perma, permb, permc need to be applied to
    //  the indexes of A, B, and C to get the matricized form.

    //  Numbering scheme:
    //  0     .. N - 1         -- outer indexes from A
    //  N     .. N + M - 1     -- outer indexes from B
    //  N + M .. N + M + K - 1 -- inner indexes

    size_t ioa = 0, iob = N, ii = N + M;

    sequence<k_ordera, size_t> idxa1(0), idxa2(0);
    sequence<k_orderb, size_t> idxb1(0), idxb2(0);
    sequence<k_orderc, size_t> idxc1(0), idxc2(0);

    //  Build initial index ordering

    for(size_t i = 0; i < k_orderc; i++) {
        size_t j = conn[i] - k_orderc;
        if(j < k_ordera) {
            idxc1[i] = ioa;
            idxa1[j] = ioa;
            ioa++;
        } else {
            j -= k_ordera;
            idxc1[i] = iob;
            idxb1[j] = iob;
            iob++;
        }
    }
    for(size_t i = 0; i < k_ordera; i++) {
        if(conn[k_orderc + i] < k_orderc) continue;
        size_t j = conn[k_orderc + i] - k_orderc - k_ordera;
        idxa1[i] = ii;
        idxb1[j] = ii;
        ii++;
    }

    //  Build matricized index ordering

    size_t iai, iao, ibi, ibo, ica, icb;
    if(idxa1[k_ordera - 1] >= N + M) {
        //  Last index in A is an inner index
        iai = N + K; iao = N;
    } else {
        //  Last index in A is an outer index
        iai = K; iao = N + K;
    }
    if(idxb1[k_orderb - 1] >= N + M) {
        //  Last index in B is an inner index
        ibi = M + K; ibo = M;
    } else {
        //  Last index in B is an outer index
        ibi = K; ibo = M + K;
    }
    if(idxc1[k_orderc - 1] < N) {
        //  Last index in C comes from A
        ica = N + M; icb = M;
    } else {
        //  Last index in C comes from B
        ica = N; icb = N + M;
    }

    for(size_t i = 0; i < k_ordera; i++) {
        if(idxa1[k_ordera - i - 1] >= N + M) {
            idxa2[iai - 1] = idxa1[k_ordera - i - 1];
            iai--;
        } else {
            idxa2[iao - 1] = idxa1[k_ordera - i - 1];
            iao--;
        }
    }
    for(size_t i = 0; i < k_orderb; i++) {
        if(idxb1[k_orderb - i - 1] >= N + M) {
            idxb2[ibi - 1] = idxb1[k_orderb - i - 1];
            ibi--;
        } else {
            idxb2[ibo - 1] = idxb1[k_orderb - i - 1];
            ibo--;
        }
    }
    for(size_t i = 0; i < k_orderc; i++) {
        if(idxc1[k_orderc - i - 1] < N) {
            idxc2[ica - 1] = idxc1[k_orderc - i - 1];
            ica--;
        } else {
            idxc2[icb - 1] = idxc1[k_orderc - i - 1];
            icb--;
        }
    }

    bool lasta_i = (idxa2[k_ordera - 1] >= N + M);
    bool lastb_i = (idxb2[k_orderb - 1] >= N + M);
    bool lastc_a = (idxc2[k_orderc - 1] < N);

    if(lastc_a) {
        if(lasta_i) {
            if(lastb_i) {
                //  C(ji) = A(ik) B(jk)
                for(size_t i = 0; i < N; i++) idxa2[i] = idxc2[M + i];
                for(size_t i = 0; i < M; i++) idxc2[i] = idxb2[i];
                for(size_t i = 0; i < K; i++) idxa2[N + i] = idxb2[M + i];
            } else {
                //  C(ji) = A(ik) B(kj)
                for(size_t i = 0; i < N; i++) idxa2[i] = idxc2[M + i];
                for(size_t i = 0; i < M; i++) idxc2[i] = idxb2[K + i];
                for(size_t i = 0; i < K; i++) idxb2[i] = idxa2[N + i];
            }
        } else {
            if(lastb_i) {
                //  C(ji) = A(ki) B(jk)
                for(size_t i = 0; i < N; i++) idxa2[K + i] = idxc2[M + i];
                for(size_t i = 0; i < M; i++) idxc2[i] = idxb2[i];
                for(size_t i = 0; i < K; i++) idxa2[i] = idxb2[M + i];
            } else {
                //  C(ji) = A(ki) B(kj)
                for(size_t i = 0; i < N; i++) idxa2[K + i] = idxc2[M + i];
                for(size_t i = 0; i < M; i++) idxc2[i] = idxb2[K + i];
                for(size_t i = 0; i < K; i++) idxb2[i] = idxa2[i];
            }
        }
    } else {
        if(lasta_i) {
            if(lastb_i) {
                //  C(ij) = A(ik) B(jk)
                for(size_t i = 0; i < N; i++) idxa2[i] = idxc2[i];
                for(size_t i = 0; i < M; i++) idxb2[i] = idxc2[N + i];
                for(size_t i = 0; i < K; i++) idxa2[N + i] = idxb2[M + i];
            } else {
                //  C(ij) = A(ik) B(kj)
                for(size_t i = 0; i < N; i++) idxc2[i] = idxa2[i];
                for(size_t i = 0; i < M; i++) idxb2[K + i] = idxc2[N + i];
                for(size_t i = 0; i < K; i++) idxb2[i] = idxa2[N + i];
            }
        } else {
            if(lastb_i) {
                //  C(ij) = A(ki) B(jk)
                for(size_t i = 0; i < N; i++) idxc2[i] = idxa2[K + i];
                for(size_t i = 0; i < M; i++) idxb2[i] = idxc2[N + i];
                for(size_t i = 0; i < K; i++) idxa2[i] = idxb2[M + i];
            } else {
                //  C(ij) = A(ki) B(kj)
                for(size_t i = 0; i < N; i++) idxc2[i] = idxa2[K + i];
                for(size_t i = 0; i < M; i++) idxc2[N + i] = idxb2[K + i];
                for(size_t i = 0; i < K; i++) idxb2[i] = idxa2[i];
            }
        }
    }

    permutation_builder<k_ordera> pba(idxa2, idxa1);
    permutation_builder<k_orderb> pbb(idxb2, idxb1);
    permutation_builder<k_orderc> pbc(idxc2, idxc1);
    perma.permute(pba.get_perm());
    permb.permute(pbb.get_perm());
    permc.permute(pbc.get_perm());
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::perform_internal(aligned_args &ar,
    double *pc, const dimensions<k_orderc> &dimsc) {

    dense_tensor_rd_ctrl<k_ordera, double> ca(ar.ta);
    dense_tensor_rd_ctrl<k_orderb, double> cb(ar.tb);

    const dimensions<k_ordera> &dimsa = ar.ta.get_dims();
    const dimensions<k_orderb> &dimsb = ar.tb.get_dims();

    dimensions<k_ordera> dimsa1(dimsa); dimsa1.permute(ar.perma);
    dimensions<k_orderb> dimsb1(dimsb); dimsb1.permute(ar.permb);

    const double *pa = 0, *pb = 0;
    double *pa1 = 0, *pb1 = 0;
    const double *pa2 = 0, *pb2 = 0;

    typename allocator<double>::pointer_type vpa, vpb;

    pa2 = pa = ca.req_const_dataptr();
    if(!ar.perma.is_identity()) {

        vpa = allocator<double>::allocate(dimsa1.get_size());
        pa1 = allocator<double>::lock_rw(vpa);

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

        loop_registers<1, 1> r;
        r.m_ptra[0] = pa;
        r.m_ptrb[0] = pa1;
        r.m_ptra_end[0] = pa + dimsa.get_size();
        r.m_ptrb_end[0] = pa1 + dimsa1.get_size();

        {
            std::auto_ptr< kernel_base<linalg, 1, 1> >kern(
                kern_dcopy<linalg>::match(1.0, loop_in, loop_out));
            tod_contract2<N, M, K>::start_timer("perma");
            tod_contract2<N, M, K>::start_timer(kern->get_name());
            loop_list_runner<linalg, 1, 1>(loop_in).run(0, r, *kern);
            tod_contract2<N, M, K>::stop_timer(kern->get_name());
            tod_contract2<N, M, K>::stop_timer("perma");
        }

        pa2 = pa1;
    }

    pb2 = pb = cb.req_const_dataptr();
    if(!ar.permb.is_identity()) {

        vpb = allocator<double>::allocate(dimsb1.get_size());
        pb1 = allocator<double>::lock_rw(vpb);

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

        loop_registers<1, 1> r;
        r.m_ptra[0] = pb;
        r.m_ptrb[0] = pb1;
        r.m_ptra_end[0] = pb + dimsb.get_size();
        r.m_ptrb_end[0] = pb1 + dimsb1.get_size();

        {
            std::auto_ptr< kernel_base<linalg, 1, 1> >kern(
                kern_dcopy<linalg>::match(1.0, loop_in, loop_out));
            tod_contract2<N, M, K>::start_timer("permb");
            tod_contract2<N, M, K>::start_timer(kern->get_name());
            loop_list_runner<linalg, 1, 1>(loop_in).run(0, r, *kern);
            tod_contract2<N, M, K>::stop_timer(kern->get_name());
            tod_contract2<N, M, K>::stop_timer("permb");
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
        loop_registers<2, 1> r;
        r.m_ptra[0] = pa2;
        r.m_ptra[1] = pb2;
        r.m_ptrb[0] = pc;
        r.m_ptra_end[0] = pa2 + dimsa1.get_size();
        r.m_ptra_end[1] = pb2 + dimsb1.get_size();
        r.m_ptrb_end[0] = pc + dimsc.get_size();

        std::auto_ptr< kernel_base<linalg, 2, 1> > kern(
            kern_dmul2<linalg>::match(ar.d, loop_in, loop_out));
        tod_contract2<N, M, K>::start_timer("kernel");
        tod_contract2<N, M, K>::start_timer(kern->get_name());
        loop_list_runner<linalg, 2, 1>(loop_in).run(0, r, *kern);
        tod_contract2<N, M, K>::stop_timer("kernel");
        tod_contract2<N, M, K>::stop_timer(kern->get_name());
    }

    if(pa1) {
        allocator<double>::unlock_rw(vpa); pa1 = 0;
        allocator<double>::deallocate(vpa);
    }
    ca.ret_const_dataptr(pa);

    if(pb1) {
        allocator<double>::unlock_rw(vpb); pb1 = 0;
        allocator<double>::deallocate(vpb);
    }
    cb.ret_const_dataptr(pb);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_IMPL_H
