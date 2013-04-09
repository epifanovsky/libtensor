#ifndef LIBTENSOR_BTOD_SUM_IMPL_H
#define LIBTENSOR_BTOD_SUM_IMPL_H

#include <libtensor/core/bad_block_index_space.h>
#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/subgroup_orbits.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/symmetry/so_dirsum.h>
#include <libtensor/symmetry/so_merge.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_chsym.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_transform.h>
#include "../btod_sum.h"

namespace libtensor {


template<size_t N>
const char btod_sum<N>::k_clazz[] = "btod_sum<N>";


template<size_t N>
inline btod_sum<N>::btod_sum(additive_gen_bto<N, bti_traits> &op, double c) :
    m_bis(op.get_bis()), m_bidims(m_bis.get_block_index_dims()),
    m_sym(m_bis), m_dirty_sch(true), m_sch(0) {

    so_copy<N, double>(op.get_symmetry()).perform(m_sym);
    add_op(op, c);
}


template<size_t N>
btod_sum<N>::~btod_sum() {

    delete m_sch;
}


template<size_t N>
void btod_sum<N>::perform(gen_block_stream_i<N, bti_traits> &out) {

    if(m_ops.empty()) return;

    for(typename std::list<node_t>::iterator iop = m_ops.begin();
        iop != m_ops.end(); ++iop) {

        tensor_transf<N, double> tr(permutation<N>(),
            scalar_transf<double>(iop->get_coeff()));

        gen_bto_aux_chsym<N, btod_traits> out1(iop->get_op().get_symmetry(),
            m_sym, out);
        gen_bto_aux_transform<N, btod_traits> out2(tr, m_sym, out1);

        out1.open();
        out2.open();
        iop->get_op().perform(out2);
        out1.close();
        out2.close();
    }
}


template<size_t N>
void btod_sum<N>::compute_block(
    bool zero,
    const index<N> &i,
    const tensor_transf<N, double> &tr,
    dense_tensor_wr_i<N, double> &blk) {

    bool zero1 = zero;

    abs_index<N> ai(i, m_bidims);

    for(typename std::list<node_t>::iterator iop = m_ops.begin();
        iop != m_ops.end(); iop++) {

        scalar_transf<double> kc(iop->get_coeff());
        if(iop->get_op().get_schedule().contains(ai.get_abs_index())) {
            tensor_transf<N, double> tra(permutation<N>(), kc);
            tra.transform(tr);
            iop->get_op().compute_block(zero1, i, tra, blk);
            zero1 = false;
        } else {
            const symmetry<N, double> &sym = iop->get_op().get_symmetry();
            orbit<N, double> orb(sym, i);
            if(!orb.is_allowed()) continue;
            abs_index<N> ci(orb.get_acindex(), m_bidims);

            if(iop->get_op().get_schedule().contains(ci.get_abs_index())) {
                tensor_transf<N, double> tra(orb.get_transf(i));
                tra.transform(kc);
                tra.transform(tr);

                iop->get_op().compute_block(zero1, ci.get_index(), tra, blk);
                zero1 = false;
            }
        }
    }

    if(zero1) tod_set<N>().perform(blk);
}


template<size_t N>
void btod_sum<N>::perform(gen_block_tensor_i<N, bti_traits> &btb) {

    gen_bto_aux_copy<N, btod_traits> out(m_sym, btb);
    out.open();
    perform(out);
    out.close();
}


template<size_t N>
void btod_sum<N>::perform(gen_block_tensor_i<N, bti_traits> &btb,
    const scalar_transf<double> &c) {

    gen_block_tensor_rd_ctrl<N, bti_traits> cb(btb);
    addition_schedule<N, btod_traits> asch(m_sym, cb.req_const_symmetry());
    asch.build(get_schedule(), cb);

    gen_bto_aux_add<N, btod_traits> out(m_sym, asch, btb, c);
    out.open();
    perform(out);
    out.close();
}


template<size_t N>
void btod_sum<N>::perform(gen_block_tensor_i<N, bti_traits> &btb, double c) {

    perform(btb, scalar_transf<double>(c));
}


template<size_t N>
void btod_sum<N>::add_op(additive_gen_bto<N, bti_traits> &op, double c) {

    static const char method[] =
        "add_op(additive_gen_bto<N, bti_traits>&, double)";

    block_index_space<N> bis(m_bis), bis1(op.get_bis());
    bis.match_splits();
    bis1.match_splits();
    if(!bis.equals(bis1)) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "op");
    }
    if(c == 0.0) return;

    if(m_ops.empty()) {
        so_copy<N, double>(op.get_symmetry()).perform(m_sym);
    } else {
        permutation<N + N> perm0;
        block_index_space_product_builder<N, N> bbx(m_bis, m_bis, perm0);

        symmetry<N + N, double> symx(bbx.get_bis());
        so_dirsum<N, N, double>(m_sym, op.get_symmetry(), perm0).perform(symx);
        mask<N + N> msk;
        sequence<N + N, size_t> seq;
        for (register size_t i = 0; i < N; i++) {
            msk[i] = msk[i + N] = true;
            seq[i] = seq[i + N] = i;
        }
        so_merge<N + N, N, double>(symx, msk, seq).perform(m_sym);
    }
    m_ops.push_back(node_t(op, c));
    m_dirty_sch = true;
}


template<size_t N>
void btod_sum<N>::make_schedule() const {

    delete m_sch;
    m_sch = new assignment_schedule<N, double>(m_bidims);

    for(typename std::list<node_t>::iterator iop = m_ops.begin();
        iop != m_ops.end(); ++iop) {

        const symmetry<N, double> &sym1 = iop->get_op().get_symmetry();
        const assignment_schedule<N, double> &sch1 =
            iop->get_op().get_schedule();

        for(typename assignment_schedule<N, double>::iterator i = sch1.begin();
            i != sch1.end(); ++i) {

            subgroup_orbits<N, double> so(sym1, m_sym, sch1.get_abs_index(i));

            for(typename subgroup_orbits<N, double>::iterator j = so.begin();
                j != so.end(); ++j) {

                size_t aidx = so.get_abs_index(j);
                if(!m_sch->contains(aidx)) m_sch->insert(aidx);
            }
        }
    }

    m_dirty_sch = false;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SUM_IMPL_H
