#ifndef LIBTENSOR_DIAG_TOD_CONTRACT2_PART_H
#define LIBTENSOR_DIAG_TOD_CONTRACT2_PART_H

#include <deque>
#include <memory>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/bad_dimensions.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/kernels/kern_dmul2.h>
#include <libtensor/dense_tensor/to_contract2_dims.h>
#include "../diag_tensor_space.h"

namespace libtensor {


/** \brief Partial contraction of two diagonal tensors

    This class performs the contraction of two single tensor diagonals.

    \ingroup libtensor_diag_tensor
 **/
template<size_t N, size_t M, size_t K>
class diag_tod_contract2_part {
public:
    static const char *k_clazz; //!< Class name

private:
    const contraction2<N, M, K> &m_contr;
    const diag_tensor_subspace<N + K> &m_ssa;
    const dimensions<N + K> &m_dimsa;
    const double *m_pa;
    const diag_tensor_subspace<M + K> &m_ssb;
    const dimensions<M + K> &m_dimsb;
    const double *m_pb;

public:
    diag_tod_contract2_part(const contraction2<N, M, K> &contr,
        const diag_tensor_subspace<N + K> &ssa, const dimensions<N + K> &dimsa,
        const double *pa, const diag_tensor_subspace<M + K> &ssb,
        const dimensions<M + K> &dimsb, const double *pb) :
        m_contr(contr), m_ssa(ssa), m_dimsa(dimsa), m_pa(pa), m_ssb(ssb),
        m_dimsb(dimsb), m_pb(pb) { }

    void perform(const diag_tensor_subspace<N + M> &ssc,
        const dimensions<N + M> &dimsc, double *pc, double d);

private:
    template<size_t L>
    size_t get_increment(size_t i, const diag_tensor_subspace<L> &ss,
        const dimensions<L> &rdims, mask<L> &mdiag);

};


template<size_t N, size_t M, size_t K>
const char *diag_tod_contract2_part<N, M, K>::k_clazz =
    "diag_tod_contract2_part<N, M, K>";


template<size_t N, size_t M, size_t K>
void diag_tod_contract2_part<N, M, K>::perform(
    const diag_tensor_subspace<N + M> &ssc, const dimensions<N + M> &dimsc,
    double *pc, double d) {

    static const char *method = "perform()";

#ifdef LIBTENSOR_DEBUG
    to_contract2_dims<N, M, K> dimsc2(m_contr, m_dimsa, m_dimsb);
    if(!dimsc2.get_dims().equals(dimsc)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "dimsc");
    }
#endif // LIBTENSOR_DEBUG

    sequence<N + K, size_t> nda(N + K);
    index<N + K> ria1, ria2;
    for(size_t i = 0; i < N + K; i++) ria2[i] = m_dimsa[i] - 1;
    for(size_t i = 0; i < m_ssa.get_ndiag(); i++) {
        const mask<N + K> &m = m_ssa.get_diag_mask(i);
        bool first = true;
        for(size_t j = 0; j < N + K; j++) if(m[j]) {
            nda[j] = i;
            if(!first) ria2[j] = 0;
            first = false;
        }
    }
    dimensions<N + K> rdimsa(index_range<N + K>(ria1, ria2));

    sequence<M + K, size_t> ndb(M + K);
    index<M + K> rib1, rib2;
    for(size_t i = 0; i < M + K; i++) rib2[i] = m_dimsb[i] - 1;
    for(size_t i = 0; i < m_ssb.get_ndiag(); i++) {
        const mask<M + K> &m = m_ssb.get_diag_mask(i);
        bool first = true;
        for(size_t j = 0; j < M + K; j++) if(m[j]) {
            ndb[j] = i;
            if(!first) rib2[j] = 0;
            first = false;
        }
    }
    dimensions<M + K> rdimsb(index_range<M + K>(rib1, rib2));

    sequence<N + M, size_t> ndc(N + M);
    index<N + M> ric1, ric2;
    for(size_t i = 0; i < N + M; i++) ric2[i] = dimsc[i] - 1;
    for(size_t i = 0; i < ssc.get_ndiag(); i++) {
        const mask<N + M> &m = ssc.get_diag_mask(i);
        bool first = true;
        for(size_t j = 0; j < N + M; j++) if(m[j]) {
            ndc[j] = i;
            if(!first) ric2[j] = 0;
            first = false;
        }
    }
    dimensions<N + M> rdimsc(index_range<N + M>(ric1, ric2));

    const sequence<2 * (N + M + K), size_t> &conn = m_contr.get_conn();
    mask<2 * (N + M + K)> mdone;
    std::list< loop_list_node<2, 1> > loop_in, loop_out;

    for(size_t i = 0; i < 2 * (N + M + K); i++) if(!mdone[i]) {

        mask<N + K> mdonea;
        mask<M + K> mdoneb;
        mask<N + M> mdonec;

        size_t w = 0;
        size_t inca = 0, incb = 0, incc = 0;
        std::deque<size_t> nextj;
        nextj.push_back(i);

        do {
            size_t j = nextj.front();
            nextj.pop_front();

            size_t ia = N + K, ib = M + K, ic = N + M;
            if(j < N + M) ic = j;
            else if(j < (N + M) + (N + K)) ia = j - (N + M);
            else ib = j - (N + M) - (N + K);
            if(conn[j] < N + M) ic = conn[j];
            else if(conn[j] < (N + M) + (N + K)) ia = conn[j] - (N + M);
            else ib = conn[j] - (N + M) - (N + K);

            if(w == 0) w = (ic < N + M) ? dimsc[ic] : m_dimsa[ia];

            if(ia < N + K) {
                mask<N + K> mda, ma0, ma;
                size_t inca1 = get_increment(ia, m_ssa, rdimsa, mda);
                ma = mda & mdonea;
                if(ma.equals(ma0)) inca += inca1;
                mdonea[ia] = true;
                for(size_t k = 0; k < N + K; k++) if(mda[k] && !mdonea[k]) {
                    nextj.push_back((N + M) + k);
                    break;
                }
            }

            if(ib < M + K) {
                mask<M + K> mdb, mb0, mb;
                size_t incb1 = get_increment(ib, m_ssb, rdimsb, mdb);
                mb = mdb & mdoneb;
                if(mb.equals(mb0)) incb += incb1;
                mdoneb[ib] = true;
                for(size_t k = 0; k < M + K; k++) if(mdb[k] && !mdoneb[k]) {
                    nextj.push_back((N + M) + (N + K) + k);
                    break;
                }
            }

            if(ic < N + M) {
                mask<N + M> mdc, mc0, mc;
                size_t incc1 = get_increment(ic, ssc, rdimsc, mdc);
                mc = mdc & mdonec;
                if(mc.equals(mc0)) incc += incc1;
                mdonec[ic] = true;
                for(size_t k = 0; k < N + M; k++) if(mdc[k] && !mdonec[k]) {
                    nextj.push_back(k);
                    break;
                }
            }

        } while(!nextj.empty());

        typename std::list< loop_list_node<2, 1> >::iterator inode =
            loop_in.insert(loop_in.end(), loop_list_node<2, 1>(w));
        inode->stepa(0) = inca;
        inode->stepa(1) = incb;
        inode->stepb(0) = incc;

        mask<2 * (N + M + K)> mdone1;
        for(size_t k = 0; k < N + K; k++) {
            mdone1[(N + M) + k] = mdonea[k];
        }
        for(size_t k = 0; k < M + K; k++) {
            mdone1[(N + M) + (N + K) + k] = mdoneb[k];
        }
        for(size_t k = 0; k < N + M; k++) {
            mdone1[k] = mdonec[k];
        }
        mdone |= mdone1;
    }

    {
//        auto_cpu_lock cpu(cpus);

        loop_registers_x<2, 1, double> r;
        r.m_ptra[0] = m_pa;
        r.m_ptra[1] = m_pb;
        r.m_ptrb[0] = pc;
        r.m_ptra_end[0] = m_pa + rdimsa.get_size();
        r.m_ptra_end[1] = m_pb + rdimsb.get_size();
        r.m_ptrb_end[0] = pc + rdimsc.get_size();

        std::unique_ptr< kernel_base<linalg, 2, 1, double> > kern(
            kern_mul2<linalg, double>::match(d, loop_in, loop_out));
//        diag_tod_contract2_part<N, M, K>::start_timer(kern->get_name());
        loop_list_runner_x<linalg, 2, 1, double>(loop_in).run(0, r, *kern);
//        diag_tod_contract2_part<N, M, K>::stop_timer(kern->get_name());
    }
}


template<size_t N, size_t M, size_t K> template<size_t L>
size_t diag_tod_contract2_part<N, M, K>::get_increment(
    size_t i, const diag_tensor_subspace<L> &ss, const dimensions<L> &rdims,
    mask<L> &mdiag) {

    //  Returns the array index increment that corresponds to a diagonal
    //  Also returns the mask of that diagonal

    size_t i1 = i;
    for(size_t id = 0; id < ss.get_ndiag(); id++) {
        const mask<L> &m = ss.get_diag_mask(id);
        if(m[i]) {
            size_t j = 0;
            while(j < L && !m[j]) j++;
            if(j < L) {
                i1 = j;
                mdiag |= m;
            }
            break;
        }
    }
    mdiag[i] = true;
    return rdims.get_increment(i1);
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_CONTRACT2_PART_H
