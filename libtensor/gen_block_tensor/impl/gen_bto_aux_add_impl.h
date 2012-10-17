#ifndef LIBTENSOR_GEN_BTO_AUX_ADD_IMPL_H
#define LIBTENSOR_GEN_BTO_AUX_ADD_IMPL_H

#include <libutil/threads/auto_lock.h>
#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/symmetry/so_dirsum.h>
#include <libtensor/symmetry/so_merge.h>
#include "../gen_bto_aux_add.h"

namespace libtensor {


template<size_t N, typename Traits>
gen_bto_aux_add<N, Traits>::gen_bto_aux_add(
    const symmetry<N, element_type> &syma,
    const addition_schedule<N, Traits> &asch,
    gen_block_tensor_i<N, bti_traits> &btb,
    const scalar_transf<element_type> &c) :

    m_bis(syma.get_bis()), m_bidims(m_bis.get_block_index_dims()),
    m_syma(m_bis), m_asch(asch), m_btb(btb), m_c(c), m_cb(m_btb),
    m_open(false), m_grpcount(0) {

    so_copy<N, element_type>(syma).perform(m_syma);
}


template<size_t N, typename Traits>
gen_bto_aux_add<N, Traits>::~gen_bto_aux_add() {

    close();
}


template<size_t N, typename Traits>
void gen_bto_aux_add<N, Traits>::open() {

    if(m_open) return;

    //  Compute the symmetry of the result of the addition

    symmetry<N, element_type> symcopy(m_syma.get_bis());
    so_copy<N, element_type>(m_syma).perform(symcopy);

    permutation<N + N> p0;
    block_index_space_product_builder<N, N> bbx(m_syma.get_bis(),
        m_btb.get_bis(), p0);
    symmetry<N + N, element_type> symx(bbx.get_bis());
    so_dirsum<N, N, element_type>(m_syma, m_cb.req_const_symmetry(), p0).
        perform(symx);
    mask<N + N> msk;
    sequence<N + N, size_t> seq(0);
    for(size_t i = 0; i < N; i++) {
        msk[i] = msk[i + N] = true;
        seq[i] = seq[i + N] = i;
    }
    so_merge<N + N, N, element_type>(symx, msk, seq).
        perform(m_cb.req_symmetry());

    m_open = true;
}


template<size_t N, typename Traits>
void gen_bto_aux_add<N, Traits>::close() {

    typedef typename Traits::template to_copy_type<N>::type to_copy_type;

    typedef addition_schedule<N, Traits> schedule_type;
    typedef typename schedule_type::iterator schedule_iterator;
    typedef typename schedule_type::schedule_group schedule_group;
    typedef typename schedule_type::node schedule_node;
    typedef typename std::list<schedule_node>::const_iterator group_iterator;

    if(!m_open) return;

    //  Touch untouched orbits

    for(schedule_iterator igrp = m_asch.begin(); igrp != m_asch.end(); ++igrp) {

        const schedule_group &grp = m_asch.get_node(igrp);

        bool touched = false;
        for(group_iterator inode = grp.begin(); inode != grp.end(); ++inode) {
            abs_index<N> aia(inode->cia, m_bidims);
            if(inode->zeroa) continue;
            if(m_grpmap.find(inode->cia) != m_grpmap.end()) {
                touched = true;
            }
        }
        if(touched) continue;

        for(group_iterator inode = grp.begin(); inode != grp.end(); ++inode) {

            //  Skip the canonical block in B
            if(inode->cib == inode->cic) continue;

            abs_index<N> aib(inode->cib, m_bidims),
                aic(inode->cic, m_bidims);

            //  Skip zero blocks
            if(m_cb.req_is_zero_block(aib.get_index())) continue;

            block_type &blkb = m_cb.req_block(aib.get_index());
            block_type &blkc = m_cb.req_block(aic.get_index());
            to_copy_type(blkb, inode->trb).perform(true, blkc);
            m_cb.ret_block(aib.get_index());
            m_cb.ret_block(aic.get_index());
        }
    }

    //  Clean up

    for(size_t i = 0; i < m_grpcount; i++) delete m_grpmtx[i];
    m_grpcount = 0;
    m_grpmap.clear();
    m_grpmtx.clear();

    m_open = false;
}


template<size_t N, typename Traits>
void gen_bto_aux_add<N, Traits>::put(
    const index<N> &idx,
    block_type &blk,
    const tensor_transf<N, element_type> &tr) {

    typedef typename Traits::template to_copy_type<N>::type to_copy_type;

    typedef addition_schedule<N, Traits> schedule_type;
    typedef typename schedule_type::iterator schedule_iterator;
    typedef typename schedule_type::schedule_group schedule_group;
    typedef typename schedule_type::node schedule_node;
    typedef typename std::list<schedule_node>::const_iterator group_iterator;

    abs_index<N> aia(idx, m_bidims);

    for(schedule_iterator igrp = m_asch.begin(); igrp != m_asch.end(); ++igrp) {

        const schedule_group &grp = m_asch.get_node(igrp);

        bool contains = false;
        for(group_iterator inode = grp.begin(); inode != grp.end(); ++inode) {
            if(!inode->zeroa && inode->cia == aia.get_abs_index()) {
                contains = true;
                break;
            }
        }
        if(!contains) continue;

        bool touch = false;
        libutil::mutex *mtx = 0;
        {
            libutil::auto_lock<libutil::mutex> lock(m_mtx);

            if(m_grpmap.find(aia.get_abs_index()) == m_grpmap.end()) {
                size_t grpnum = m_grpcount++;
                for(group_iterator inode = grp.begin(); inode != grp.end();
                    ++inode) if(!inode->zeroa) m_grpmap[inode->cia] = grpnum;
                mtx = new libutil::mutex;
                m_grpmtx.push_back(mtx);
                touch = true;
                mtx->lock();
            } else {
                size_t grpnum = m_grpmap[aia.get_abs_index()];
                mtx = m_grpmtx[grpnum];
            }
        }

        //  Touch the group if necessary; group mutex is locked already
        if(touch) {

            try {
                for(group_iterator inode = grp.begin(); inode != grp.end();
                    ++inode) {

                    //  Skip the canonical block in B
                    if(inode->cib == inode->cic) continue;

                    abs_index<N> aib(inode->cib, m_bidims),
                        aic(inode->cic, m_bidims);

                    //  Skip zero blocks
                    if(m_cb.req_is_zero_block(aib.get_index())) continue;

                    block_type &blkb = m_cb.req_block(aib.get_index());
                    block_type &blkc = m_cb.req_block(aic.get_index());
                    to_copy_type(blkb, inode->trb).perform(true, blkc);
                    m_cb.ret_block(aib.get_index());
                    m_cb.ret_block(aic.get_index());
                }
            } catch(...) {
                mtx->unlock();
                throw;
            }

            mtx->unlock();
        }

        //  Add contribution from A
        {
            libutil::auto_lock<libutil::mutex> lock(*mtx);

            for(group_iterator inode = grp.begin(); inode != grp.end();
                ++inode) {

                //  Skip non-pertinent nodes
                if(inode->zeroa || inode->cia != aia.get_abs_index()) continue;

                abs_index<N> aic(inode->cic, m_bidims);
                bool zeroc = m_cb.req_is_zero_block(aic.get_index());

                block_type &blkc = m_cb.req_block(aic.get_index());
                tensor_transf<N, element_type> tra(tr);
                tra.transform(inode->tra);
                tra.transform(m_c);
                to_copy_type(blk, tra).perform(zeroc, blkc);
                m_cb.ret_block(aic.get_index());
            }
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_AUX_ADD_IMPL_H
