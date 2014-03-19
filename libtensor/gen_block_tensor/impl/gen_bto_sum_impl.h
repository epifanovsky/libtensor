#ifndef LIBTENSOR_GEN_BTO_SUM_IMPL_H
#define LIBTENSOR_GEN_BTO_SUM_IMPL_H

#include <libtensor/core/bad_block_index_space.h>
#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/subgroup_orbits.h>
#include <libtensor/symmetry/so_dirsum.h>
#include <libtensor/symmetry/so_merge.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_chsym.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_transform.h>
#include "../gen_bto_sum.h"

namespace libtensor {


template<size_t N, typename Traits>
const char gen_bto_sum<N, Traits>::k_clazz[] = "gen_bto_sum<N, Traits>";


template<size_t N, typename Traits>
gen_bto_sum<N, Traits>::gen_bto_sum(
    additive_gen_bto<N, bti_traits> &op,
    const scalar_transf<element_type> &c) :

    m_bis(op.get_bis()), m_bidims(m_bis.get_block_index_dims()),
    m_sym(m_bis), m_dirty_sch(true), m_sch(0) {

    so_copy<N, element_type>(op.get_symmetry()).perform(m_sym);
    add_op(op, c);
}


template<size_t N, typename Traits>
gen_bto_sum<N, Traits>::~gen_bto_sum() {

    delete m_sch;
}


template<size_t N, typename Traits>
void gen_bto_sum<N, Traits>::add_op(
    additive_gen_bto<N, bti_traits> &op,
    const scalar_transf<element_type> &c) {

    static const char method[] = "add_op(additive_gen_bto<N, bti_traits>&, "
        "const scalar_transf<element_type>&)";

    block_index_space<N> bis(m_bis), bis1(op.get_bis());
    bis.match_splits();
    bis1.match_splits();
    if(!bis.equals(bis1)) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "op");
    }
    if(Traits::is_zero(c)) return;

    if(m_ops.empty()) {
        so_copy<N, element_type>(op.get_symmetry()).perform(m_sym);
    } else {
        permutation<N + N> perm0;
        block_index_space_product_builder<N, N> bbx(m_bis, m_bis, perm0);

        symmetry<N + N, element_type> symx(bbx.get_bis());
        so_dirsum<N, N, element_type>(m_sym, op.get_symmetry(), perm0).
            perform(symx);
        mask<N + N> msk;
        sequence<N + N, size_t> seq;
        for(size_t i = 0; i < N; i++) {
            msk[i] = msk[i + N] = true;
            seq[i] = seq[i + N] = i;
        }
        so_merge<N + N, N, element_type>(symx, msk, seq).perform(m_sym);
    }
    m_ops.push_back(std::make_pair(&op, c));
    m_dirty_sch = true;
}


template<size_t N, typename Traits>
void gen_bto_sum<N, Traits>::perform(gen_block_stream_i<N, bti_traits> &out) {

    if(m_ops.empty()) return;

    if(m_ops.size() == 1) {

        typename std::list<op_type>::iterator iop = m_ops.begin();

        tensor_transf<N, element_type> tr(permutation<N>(), iop->second);
        gen_bto_aux_transform<N, Traits> out1(tr, m_sym, out);

        out1.open();
        iop->first->perform(out1);
        out1.close();

    } else {

        for(typename std::list<op_type>::iterator iop = m_ops.begin();
            iop != m_ops.end(); ++iop) {

            tensor_transf<N, double> tr(permutation<N>(), iop->second);

            gen_bto_aux_chsym<N, Traits> out1(iop->first->get_symmetry(),
                m_sym, out);
            gen_bto_aux_transform<N, Traits> out2(tr, m_sym, out1);

            out1.open();
            out2.open();
            iop->first->perform(out2);
            out1.close();
            out2.close();
        }

    }
}


template<size_t N, typename Traits>
void gen_bto_sum<N, Traits>::compute_block(
    bool zero,
    const index<N> &i,
    const tensor_transf<N, element_type> &tr,
    wr_block_type &blk) {

    typedef typename Traits::template to_set_type<N>::type to_set;

    bool zero1 = zero;

    abs_index<N> ai(i, m_bidims);

    for(typename std::list<op_type>::iterator iop = m_ops.begin();
        iop != m_ops.end(); iop++) {

        if(iop->first->get_schedule().contains(ai.get_abs_index())) {
            tensor_transf<N, element_type> tra(permutation<N>(), iop->second);
            tra.transform(tr);
            iop->first->compute_block(zero1, i, tra, blk);
            zero1 = false;
        } else {
            const symmetry<N, element_type> &sym = iop->first->get_symmetry();
            orbit<N, element_type> orb(sym, i);
            if(!orb.is_allowed()) continue;
            abs_index<N> ci(orb.get_acindex(), m_bidims);

            if(iop->first->get_schedule().contains(ci.get_abs_index())) {
                tensor_transf<N, element_type> tra(orb.get_transf(i));
                tra.transform(iop->second);
                tra.transform(tr);

                iop->first->compute_block(zero1, ci.get_index(), tra, blk);
                zero1 = false;
            }
        }
    }

    if(zero1) to_set().perform(blk);
}


template<size_t N, typename Traits>
void gen_bto_sum<N, Traits>::make_schedule() const {

    delete m_sch;
    m_sch = new assignment_schedule<N, element_type>(m_bidims);

    for(typename std::list<op_type>::const_iterator iop = m_ops.begin();
        iop != m_ops.end(); ++iop) {

        const symmetry<N, element_type> &sym1 = iop->first->get_symmetry();
        const assignment_schedule<N, element_type> &sch1 =
            iop->first->get_schedule();

        for(typename assignment_schedule<N, element_type>::iterator i =
            sch1.begin(); i != sch1.end(); ++i) {

            subgroup_orbits<N, element_type> so(sym1, m_sym,
                sch1.get_abs_index(i));

            for(typename subgroup_orbits<N, element_type>::iterator j =
                so.begin(); j != so.end(); ++j) {

                size_t aidx = so.get_abs_index(j);
                if(!m_sch->contains(aidx)) m_sch->insert(aidx);
            }
        }
    }

    m_dirty_sch = false;
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SUM_IMPL_H
