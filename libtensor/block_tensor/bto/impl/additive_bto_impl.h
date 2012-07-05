#ifndef LIBTENSOR_ADDITIVE_BTO_IMPL_H
#define LIBTENSOR_ADDITIVE_BTO_IMPL_H

#include <libtensor/symmetry/so_copy.h>
#include <libtensor/symmetry/so_dirsum.h>
#include <libtensor/symmetry/so_merge.h>

namespace libtensor {


template<size_t N, typename Traits>
void additive_bto<N, Traits>::compute_block(block_t &blk,
        const index<N> &i, cpu_pool &cpus) {

    compute_block(true, blk, i, tensor_transf<N, element_t>(),
            Traits::identity(), cpus);
}


template<size_t N, typename Traits>
void additive_bto<N, Traits>::compute_block(additive_bto<N, Traits> &op,
        bool zero, block_t &blk, const index<N> &i,
        const tensor_transf<N, element_t> &tr,
        const element_t &c, cpu_pool &cpus) {

    op.compute_block(zero, blk, i, tr, c, cpus);
}


template<size_t N, typename Traits>
void additive_bto<N, Traits>::perform(block_tensor_t &bt, const element_t &c) {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    if(Traits::is_zero(c)) return;

    sync_on();

    block_tensor_ctrl_t ctrl(bt);
    ctrl.req_sync_on();

    symmetry<N, element_t> symcopy(bt.get_bis());
    so_copy<N, element_t>(ctrl.req_const_symmetry()).perform(symcopy);

    permutation<N + N> p0;
    block_index_space_product_builder<N, N> bbx(get_bis(), bt.get_bis(), p0);
    symmetry<N + N, element_t> symx(bbx.get_bis());
    so_dirsum<N, N, element_t>(symcopy, get_symmetry(), p0).perform(symx);
    mask<N + N> msk;
    sequence<N + N, size_t> seq;
    for (register size_t i = 0; i < N; i++) {
        msk[i] = msk[i + N] = true;
        seq[i] = seq[i + N] = i;
    }
    so_merge<N + N, N, element_t>(symx, msk, seq).perform(ctrl.req_symmetry());

    dimensions<N> bidims(bt.get_bis().get_block_index_dims());
    schedule_t sch(get_symmetry(), symcopy);
    sch.build(get_schedule(), ctrl);

    std::vector<task*> tasks;
    task_batch batch;

    for(typename schedule_t::iterator igrp = sch.begin(); igrp != sch.end();
            ++igrp) {

        task *t = new task(*this, bt, bidims, sch, igrp, c);
        tasks.push_back(t);
        batch.push(*t);
    }

    batch.wait();
    for(typename std::vector<task*>::iterator i = tasks.begin();
            i != tasks.end(); i++) {
        delete *i;
    }
    tasks.clear();

    ctrl.req_sync_off();
    sync_off();
}


template<size_t N, typename Traits>
void additive_bto<N, Traits>::task::perform(cpu_pool &cpus) throw (exception) {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;
    typedef typename Traits::template to_set_type<N>::type to_set_t;
    typedef typename Traits::template to_copy_type<N>::type to_copy_t;

    block_tensor_ctrl_t ctrl(m_bt);

    const typename schedule_t::schedule_group &grp = m_sch.get_node(m_i);

    typedef std::pair<size_t, block_t*> la_pair_t;
    std::list<la_pair_t> la;

    for(typename std::list<typename schedule_t::schedule_node>::const_iterator
            inode = grp.lst.begin(); inode != grp.lst.end(); ++inode) {

        const typename schedule_t::schedule_node &node = *inode;

        if(node.zeroa) continue;

        typename std::list<la_pair_t>::iterator ila = la.begin();
        for (; ila != la.end(); ila++) {
            if(ila->first == node.cia) break;
        }
        if(ila == la.end()) {
            abs_index<N> aia(node.cia, m_bidims);
            block_t &blka = ctrl.req_aux_block(aia.get_index());
            to_set_t().perform(cpus, blka);
            m_bto.compute_block(false, blka, aia.get_index(),
                    node.tra, m_c, cpus);
            la.push_back(la_pair_t(node.cia, &blka));
        }
    }

    for(typename std::list<typename schedule_t::schedule_node>::const_iterator
            inode = grp.lst.begin(); inode != grp.lst.end(); ++inode) {

        const typename schedule_t::schedule_node &node = *inode;
        if(node.cib == node.cic) continue;

        if(node.zeroa) {

            abs_index<N> aib(node.cib, m_bidims), aic(node.cic, m_bidims);
            bool zerob = ctrl.req_is_zero_block(aib.get_index());
            block_t &blkc = ctrl.req_block(aic.get_index());
            if(zerob) {
                // this should actually never happen, but just in case
                to_set_t().perform(cpus, blkc);
            } else {
                block_t &blkb = ctrl.req_block(aib.get_index());
                to_copy_t(blkb, node.trb).perform(cpus, true, 1.0, blkc);
                ctrl.ret_block(aib.get_index());
            }
            ctrl.ret_block(aic.get_index());

        } else {

            typename std::list<la_pair_t>::iterator ila = la.begin();
            for (; ila != la.end(); ++ila) {
                if(ila->first == node.cia) break;
            }

            abs_index<N> aib(node.cib, m_bidims), aic(node.cic, m_bidims);
            bool zerob = ctrl.req_is_zero_block(aib.get_index());
            block_t &blkc = ctrl.req_block(aic.get_index());
            if(zerob) {
                abs_index<N> aia(node.cia, m_bidims);
                to_copy_t(*ila->second,
                        node.tra).perform(cpus, true, 1.0, blkc);
            } else {
                abs_index<N> aia(node.cia, m_bidims);
                block_t &blkb = ctrl.req_block(aib.get_index());
                to_copy_t(*ila->second, node.tra).perform(cpus,
                        true, 1.0, blkc);
                to_copy_t(blkb, node.trb).perform(cpus,
                        false, 1.0, blkc);
                ctrl.ret_block(aib.get_index());
            }
            ctrl.ret_block(aic.get_index());
        }
    }

    for(typename std::list<typename schedule_t::schedule_node>::const_iterator
            inode = grp.lst.begin(); inode != grp.lst.end(); ++inode) {

        const typename schedule_t::schedule_node &node = *inode;
        if(node.cib != node.cic) continue;
        if(node.zeroa) continue;

        typename std::list<la_pair_t>::iterator ila = la.begin();
        for(; ila != la.end(); ++ila) if(ila->first == node.cia) break;

        abs_index<N> aib(node.cib, m_bidims);
        bool zerob = ctrl.req_is_zero_block(aib.get_index());
        block_t &blkb = ctrl.req_block(aib.get_index());
        if(zerob) {
            abs_index<N> aia(node.cia, m_bidims);
            to_copy_t(*ila->second, node.tra).perform(cpus, true, 1.0, blkb);
        } else {
            abs_index<N> aia(node.cia, m_bidims);
            to_copy_t(*ila->second, node.tra).perform(cpus, false, 1.0, blkb);
        }
        ctrl.ret_block(aib.get_index());
    }

    for(typename std::list<la_pair_t>::iterator ila = la.begin();
            ila != la.end(); ++ila) {

        abs_index<N> aia(ila->first, m_bidims);
        ctrl.ret_aux_block(aia.get_index());
    }
    la.clear();
}


} // namespace libtensor

#endif // LIBTENSOR_ADDITIVE_BTOD_IMPL_H
