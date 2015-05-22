#ifndef LIBTENSOR_GEN_BTO_SYMMETRIZE4_IMPL_H
#define LIBTENSOR_GEN_BTO_SYMMETRIZE4_IMPL_H

#include <set>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/short_orbit.h>
#include <libtensor/symmetry/so_symmetrize.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_symmetrize.h>
#include "../gen_bto_symmetrize4.h"

namespace libtensor {


template<size_t N, typename Traits, typename Timed>
const char gen_bto_symmetrize4<N, Traits, Timed>::k_clazz[] =
    "gen_bto_symmetrize4<N, Traits, Timed>";


template<size_t N, typename Traits, typename Timed>
gen_bto_symmetrize4<N, Traits, Timed>::gen_bto_symmetrize4(
    additive_gen_bto<N, bti_traits> &op,
    const permutation<N> &perm1,
    const permutation<N> &perm2,
    const permutation<N> &perm3,
    bool symm) :

    m_op(op), m_perm1(perm1), m_perm2(perm2), m_perm3(perm3), m_symm(symm),
    m_sym(op.get_bis()), m_sch(0) {

    static const char method[] =
        "gen_bto_symmetrize4(additive_bto<N, btod_traits>&, "
        "const permutation<N>&, const permutation<N>&, const permutation<N>&, "
        "bool)";

    permutation<N> p1(perm1); p1.permute(perm1);
    if(perm1.is_identity() || !p1.is_identity()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "perm1");
    }
    permutation<N> p2(perm2); p2.permute(perm2);
    if(perm2.is_identity() || !p2.is_identity()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "perm2");
    }
    permutation<N> p3(perm3); p3.permute(perm3);
    if(perm3.is_identity() || !p3.is_identity()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "perm3");
    }
    permutation<N> p12, p13, p23;
    p12.permute(perm1).permute(perm2);
    p13.permute(perm1).permute(perm3);
    p23.permute(perm2).permute(perm3);
    permutation<N> p12i, p13i, p23i;
    p12i.permute(p12).permute(p12).permute(p12);
    p13i.permute(p13).permute(p13).permute(p13);
    p23i.permute(p23).permute(p23).permute(p23);
    if(p12.is_identity() || !p12i.is_identity()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "perm1,perm2");
    }
    if(p13.is_identity() || !p13i.is_identity()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "perm1,perm3");
    }
    if(p23.is_identity() || !p23i.is_identity()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "perm2,perm3");
    }
    permutation<N> p123;
    p123.permute(perm1).permute(perm2).permute(perm3);
    permutation<N> p123i;
    p123i.permute(p123).permute(p123).permute(p123).permute(p123);
    if(p123.is_identity() || !p123i.is_identity()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "perm1,perm2,perm3");
    }

    make_symmetry();
    make_schedule();
}


template<size_t N, typename Traits, typename Timed>
gen_bto_symmetrize4<N, Traits, Timed>::~gen_bto_symmetrize4() {

    delete m_sch;
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize4<N, Traits, Timed>::perform(
    gen_block_stream_i<N, bti_traits> &out) {

    try {

        scalar_transf<element_type> str(m_symm ? 1.0 : -1.0);

        tensor_transf<N, element_type> tr1(m_perm1, str);
        tensor_transf<N, element_type> tr2(m_perm2, str);
        tensor_transf<N, element_type> tr3(m_perm3, str);
        tensor_transf<N, element_type> trshift2, trshift3, trshift4;
        trshift2.transform(tr1);
        trshift3.transform(tr1).transform(tr2);
        trshift4.transform(tr1).transform(tr2).transform(tr3);

        gen_bto_aux_symmetrize<N, Traits> out2(m_op.get_symmetry(), m_sym, out);

        tensor_transf<N, element_type> trc4;
        for(int i = 0; i < 4; i++) {
            tensor_transf<N, element_type> trc3(trc4);
            for(int j = 0; j < 3; j++) {
                tensor_transf<N, element_type> trc2e(trc3), trc2o(trc3);
                trc2o.transform(trshift2);
                out2.add_transf(trc2e);
                out2.add_transf(trc2o);
                trc3.transform(trshift3);
            }
            trc4.transform(trshift4);
        }
        out2.open();
        m_op.perform(out2);
        out2.close();

    } catch(...) {
        throw;
    }
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize4<N, Traits, Timed>::compute_block(
    bool zero,
    const index<N> &ib,
    const tensor_transf<N, element_type> &trb,
    wr_block_type &blkb) {

    typedef typename Traits::template temp_block_type<N>::type temp_block_type;
    typedef typename Traits::template to_copy_type<N>::type to_copy;
    typedef typename Traits::template to_set_type<N>::type to_set;

    typedef typename sym_schedule_type::iterator iterator_t;

    bool zero1 = zero;

    dimensions<N> bidims(m_op.get_bis().get_block_index_dims());
    abs_index<N> aib(ib, bidims);

    sym_schedule_type sch;
    make_schedule_blk(aib, sch);

    std::pair<iterator_t, iterator_t> jr = sch.equal_range(aib.get_abs_index());
    std::list<schrec> sch1;
    for(iterator_t j = jr.first; j != jr.second; ++j) {
        sch1.push_back(j->second);
    }
    sch.clear();

    while(!sch1.empty()) {

        abs_index<N> ai(sch1.front().ai, bidims);
        size_t n = 0;
        for(typename std::list<schrec>::iterator j = sch1.begin();
            j != sch1.end(); ++j) {
            if(j->ai == ai.get_abs_index()) n++;
        }

        if(n == 1) {

            tensor_transf<N, element_type> tri(sch1.front().tr);
            tri.transform(trb);
            m_op.compute_block(zero1, ai.get_index(), tri, blkb);
            zero1 = false;
            sch1.pop_front();

        } else {

            dimensions<N> dims(blkb.get_dims());
            dims.permute(permutation<N>(trb.get_perm(), true));
            dims.permute(permutation<N>(sch1.front().tr.get_perm(), true));
            temp_block_type tblk(dims);
            m_op.compute_block(true, ai.get_index(),
                tensor_transf<N, element_type>(), tblk);
            for(typename std::list<schrec>::iterator j = sch1.begin();
                j != sch1.end();) {

                if(j->ai != ai.get_abs_index()) {
                    ++j; continue;
                }
                tensor_transf<N, element_type> trj(j->tr);
                trj.transform(trb);
                to_copy(tblk, trj).perform(zero1, blkb);
                zero1 = false;
                j = sch1.erase(j);
            }

        }
    }

    if(zero1) to_set().perform(zero1, blkb);
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize4<N, Traits, Timed>::make_symmetry() {

    sequence<N, size_t> seq1(0), seq2(0), seq3(0);
    for(size_t i = 0; i < N; i++) seq1[i] = seq2[i] = seq3[i] = i;
    m_perm1.apply(seq1);
    m_perm2.apply(seq2);
    m_perm3.apply(seq3);
    mask<N> m1, m2, m3, m1i, m2i, m3i;
    for(size_t i = 0; i < N; i++) {
        m1[i] = (seq1[i] != i); m1i[i] = (seq1[i] == i);
        m2[i] = (seq2[i] != i); m2i[i] = (seq2[i] == i);
        m3[i] = (seq3[i] != i); m3i[i] = (seq3[i] == i);
    }

    mask<N> mg1(m1), mg2(m2), mg3(m2), mg4(m3);
    mg1 &= m2i; mg2 &= m3i; mg3 &= m2i; mg4 &= m1i;

    sequence<N, size_t> idxgrp(0), symidx(0);
    size_t i1 = 1, i2 = 1, i3 = 1, i4 = 1;
    for(size_t i = 0; i < N; i++) {
        if(mg1[i]) {
            idxgrp[i] = 1;
            symidx[i] = i1++;
        }
        if(mg2[i]) {
            idxgrp[i] = 2;
            symidx[i] = i2++;
        }
        if(mg3[i]) {
            idxgrp[i] = 3;
            symidx[i] = i3++;
        }
        if(mg4[i]) {
            idxgrp[i] = 4;
            symidx[i] = i4++;
        }
    }
std::cout << "idxgrp = ";
for(size_t i = 0; i < N; i++) std::cout << idxgrp[i] << " ";
std::cout << std::endl;
std::cout << "symidx = ";
for(size_t i = 0; i < N; i++) std::cout << symidx[i] << " ";
std::cout << std::endl;
    scalar_transf<element_type> tr0, tr1(-1.0);
    so_symmetrize<N, double>(m_op.get_symmetry(), idxgrp, symidx,
        m_symm ? tr0 : tr1, tr0).perform(m_sym);
}


namespace {


template<size_t N, typename T>
void visit_orbit(const orbit<N, T> &o, std::set<size_t> &visited) {

    for(typename orbit<N, T>::iterator j = o.begin(); j != o.end(); j++) {
        visited.insert(o.get_abs_index(j));
    }
}


template<size_t N, typename Traits>
class gen_bto_symmetrize4_sch_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    additive_gen_bto<N, bti_traits> &m_op;
    const permutation<N> &m_perm1;
    const permutation<N> &m_perm2;
    const permutation<N> &m_perm3;
    const symmetry<N, element_type> &m_sym;
    const dimensions<N> &m_bidims;
    size_t m_aia;
    assignment_schedule<N, element_type> &m_sch;
    libutil::mutex &m_mtx;

public:
    gen_bto_symmetrize4_sch_task(
        additive_gen_bto<N, bti_traits> &op,
        const permutation<N> &perm1,
        const permutation<N> &perm2,
        const permutation<N> &perm3,
        const symmetry<N, element_type> &sym,
        const dimensions<N> &bidims,
        size_t aia,
        assignment_schedule<N, element_type> &sch,
        libutil::mutex &mtx) :

        m_op(op), m_perm1(perm1), m_perm2(perm2), m_perm3(perm3), m_sym(sym),
        m_bidims(bidims), m_aia(aia), m_sch(sch), m_mtx(mtx) { }

    virtual ~gen_bto_symmetrize4_sch_task() { }

    virtual unsigned long get_cost() const { return 0; }

    virtual void perform() {

        std::set<size_t> sch;
        std::set<size_t> visited;

        abs_index<N> ai0(m_aia, m_bidims);
        orbit<N, element_type> o(m_op.get_symmetry(), ai0.get_index());

        permutation<N> pshift2, pshift3, pshift4;
        pshift4.permute(m_perm1).permute(m_perm2).permute(m_perm3);
        pshift3.permute(m_perm1).permute(m_perm2);
        pshift2.permute(m_perm1);

        for(typename orbit<N, element_type>::iterator j = o.begin();
            j != o.end(); j++) {

            abs_index<N> aj(o.get_abs_index(j), m_bidims);

            permutation<N> pc4;
            for(int i4 = 0; i4 < 4; i4++) {
                permutation<N> pc3(pc4);
                for(int i3 = 0; i3 < 3; i3++) {
                    permutation<N> pc2e(pc3), pc2o(pc3);
                    pc2o.permute(pshift2);

                    index<N> j2e(aj.get_index()), j2o(aj.get_index());
                    j2e.permute(pc2e);
                    j2o.permute(pc2o);
                    abs_index<N> aj2e(j2e, m_bidims), aj2o(j2o, m_bidims);
                    if(visited.count(aj2e.get_abs_index()) == 0) {
                        orbit<N, element_type> o(m_sym, aj2e.get_abs_index());
                        sch.insert(o.get_acindex());
                        visit_orbit(o, visited);
                    }
                    if(visited.count(aj2o.get_abs_index()) == 0) {
                        orbit<N, element_type> o(m_sym, aj2o.get_abs_index());
                        sch.insert(o.get_acindex());
                        visit_orbit(o, visited);
                    }

                    pc3.permute(pshift3);
                }
                pc4.permute(pshift4);
            }
        }

        {
            libutil::auto_lock<libutil::mutex> lock(m_mtx);
            for(typename std::set<size_t>::const_iterator i = sch.begin();
                i != sch.end(); i++) m_sch.insert(*i);
        }
    }

};


template<size_t N, typename Traits>
class gen_bto_symmetrize4_sch_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    additive_gen_bto<N, bti_traits> &m_op;
    const permutation<N> &m_perm1;
    const permutation<N> &m_perm2;
    const permutation<N> &m_perm3;
    const symmetry<N, element_type> &m_sym;
    const dimensions<N> &m_bidims;
    const assignment_schedule<N, element_type> &m_sch0;
    assignment_schedule<N, element_type> &m_sch1;
    typename assignment_schedule<N, element_type>::iterator m_i;
    libutil::mutex m_mtx;

public:
    gen_bto_symmetrize4_sch_task_iterator(
        additive_gen_bto<N, bti_traits> &op,
        const permutation<N> &perm1,
        const permutation<N> &perm2,
        const permutation<N> &perm3,
        const symmetry<N, element_type> &sym,
        const dimensions<N> &bidims,
        const assignment_schedule<N, element_type> &sch0,
        assignment_schedule<N, element_type> &sch1) :

        m_op(op), m_perm1(perm1), m_perm2(perm2), m_perm3(perm3), m_sym(sym),
        m_bidims(bidims), m_sch0(sch0), m_sch1(sch1), m_i(m_sch0.begin()) { }

    virtual bool has_more() const {

        return m_i != m_sch0.end();
    }

    virtual libutil::task_i *get_next() {

        gen_bto_symmetrize4_sch_task<N, Traits> *t =
            new gen_bto_symmetrize4_sch_task<N, Traits>(m_op, m_perm1, m_perm2,
                m_perm3, m_sym, m_bidims, m_sch0.get_abs_index(m_i), m_sch1,
                m_mtx);
        ++m_i;
        return t;
    }

};


template<size_t N, typename Traits>
class gen_bto_symmetrize4_sch_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t) { delete t; }

};


} // unnamed namespace


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize4<N, Traits, Timed>::make_schedule() const {

    delete m_sch;
    m_sch = 0;

    gen_bto_symmetrize4::start_timer("make_schedule");

    try {

        dimensions<N> bidims(m_op.get_bis().get_block_index_dims());
        assignment_schedule<N, element_type> *sch =
            new assignment_schedule<N, element_type>(bidims);
        gen_bto_symmetrize4_sch_task_iterator<N, Traits> ti(m_op, m_perm1,
            m_perm2, m_perm3, m_sym, bidims, m_op.get_schedule(), *sch);
        gen_bto_symmetrize4_sch_task_observer<N, Traits> to;
        libutil::thread_pool::submit(ti, to);
        m_sch = sch;

    } catch(...) {

        gen_bto_symmetrize4::stop_timer("make_schedule");

    }

    gen_bto_symmetrize4::stop_timer("make_schedule");
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_symmetrize4<N, Traits, Timed>::make_schedule_blk(
    const abs_index<N> &ai, sym_schedule_type &sch) const {

    element_type scal = m_symm ? 1.0 : -1.0;

    permutation<N> pshift2, pshift3, pshift4;
    pshift4.permute(m_perm1).permute(m_perm2).permute(m_perm3);
    pshift3.permute(m_perm1).permute(m_perm2);
    pshift2.permute(m_perm1);

    const symmetry<N, element_type> &sym0 = m_op.get_symmetry();
    const assignment_schedule<N, element_type> &sch0 = m_op.get_schedule();

    //  This is a temporary schedule for the formation of the block
    std::list<schrec> sch1;

    //  Form the temporary schedule

    index<N> idxc4;
    permutation<N> pc4;
    for(int i4 = 0; i4 < 4; i4++) {
        index<N> idxc3(idxc4);
        permutation<N> pc3(pc4);
        for(int i3 = 0; i3 < 3; i3++) {
            index<N> idx1(idxc3), idx2(idxc3);
            permutation<N> p1(pc3), p2(pc3);
            idx2.permute(pshift2);
            p2.permute(pshift2);
            orbit<N, double> o1(sym0, idx1), o2(sym0, idx2);
            if(sch0.contains(o1.get_acindex())) {
                tensor_transf<N, element_type> tr(o1.get_transf(idx1));
                tr.permute(permutation<N>(p1, true));
                tr.transform(scalar_transf<element_type>(scal));
                sch1.push_back(schrec(o1.get_acindex(), tr));
            }
            if(sch0.contains(o2.get_acindex())) {
                tensor_transf<N, element_type> tr(o2.get_transf(idx2));
                tr.permute(permutation<N>(p2, true));
                tr.transform(scalar_transf<element_type>(scal));
                sch1.push_back(schrec(o2.get_acindex(), tr));
            }
            idxc3.permute(pshift3);
            pc3.permute(pshift3);
        }
        idxc4.permute(pshift4);
        pc4.permute(pshift4);
    }

    //  Consolidate and transfer the temporary schedule

    while(!sch1.empty()) {

        typename std::list<schrec>::iterator i = sch1.begin();
        abs_index<N> aidx(i->ai, ai.get_dims());
        double c = 0.0;
        tensor_transf<N, double> tr0(i->tr);

        do {
            if(i->ai != aidx.get_abs_index()) {
                ++i;
                continue;
            }
            if(tr0.get_perm().equals(i->tr.get_perm())) {
                c += i->tr.get_scalar_tr().get_coeff();
                i = sch1.erase(i);
                continue;
            }
            ++i;
        } while(i != sch1.end());
        if(c != 0.0) {
            tensor_transf<N, element_type> tr;
            tr.permute(tr0.get_perm());
            tr.transform(scalar_transf<element_type>(c));
            sch.insert(std::make_pair(
                ai.get_abs_index(), schrec(aidx.get_abs_index(), tr)));
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SYMMETRIZE4_IMPL_H
