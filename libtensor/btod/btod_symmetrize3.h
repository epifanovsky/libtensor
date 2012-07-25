#ifndef LIBTENSOR_BTOD_SYMMETRIZE3_H
#define LIBTENSOR_BTOD_SYMMETRIZE3_H

#include <algorithm>
#include "../exception.h"
#include "../timings.h"
#include "../core/allocator.h"
#include "../core/block_index_subspace_builder.h"
#include "../core/permutation_builder.h"
#include "../core/transf_list.h"
#include <libtensor/core/orbit.h>
#include "../symmetry/so_copy.h"
#include "../symmetry/so_symmetrize.h"
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


/** \brief (Anti-)symmetrizes the result of a block %tensor operation
        over three groups of indexes
    \tparam N Tensor order.

    The operation symmetrizes or anti-symmetrizes the result of another
    block %tensor operation over three indexes or groups of indexes.

    \f[
        b_{ijk} = P_{\pm} a_{ijk} = a_{ijk} \pm a_{jik} \pm a_{kji} \pm
            a_{ikj} + a_{jki} + a_{kij}
    \f]

    The constructor takes three different indexes to be symmetrized.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_symmetrize3 :
    public additive_bto<N, bto_traits<double> >,
    public timings< btod_symmetrize3<N> > {

public:
    static const char *k_clazz; //!< Class name

private:
    struct schrec {
        size_t ai;
        tensor_transf<N, double> tr;
        schrec() : ai(0) { }
        schrec(size_t ai_, const tensor_transf<N, double> &tr_) :
            ai(ai_), tr(tr_) { }
    };
    typedef std::pair<size_t, schrec> sym_schedule_pair_t;
    typedef std::multimap<size_t, schrec> sym_schedule_t;

private:
    additive_bto<N, bto_traits<double> > &m_op; //!< Symmetrized operation
    size_t m_i1; //!< First %index
    size_t m_i2; //!< Second %index
    size_t m_i3; //!< Third %index
    bool m_symm; //!< Symmetrization/anti-symmetrization
    symmetry<N, double> m_sym; //!< Symmetry of the result
    assignment_schedule<N, double> m_sch; //!< Schedule

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param op Operation to be symmetrized.
        \param i1 First %index.
        \param i2 Second %index.
        \param i3 Third %index.
        \param symm True for symmetrization, false for
            anti-symmetrization.
     **/
    btod_symmetrize3(additive_bto<N, bto_traits<double> > &op,
            size_t i1, size_t i2, size_t i3, bool symm);

    /** \brief Virtual destructor
     **/
    virtual ~btod_symmetrize3() { }

    //@}


    //!    \name Implementation of direct_block_tensor_operation<N, double>
    //@{

    virtual const block_index_space<N> &get_bis() const {
        return m_op.get_bis();
    }

    virtual const symmetry<N, double> &get_symmetry() const {
        return m_sym;
    }

    virtual const assignment_schedule<N, double> &get_schedule() const {
        return m_sch;
    }

    virtual void sync_on() {
        m_op.sync_on();
    }

    virtual void sync_off() {
        m_op.sync_off();
    }

    //@}

protected:
    //!    \brief Implementation of additive_bto<N, bto_traits<double> >
    //@{

    virtual void compute_block(bool zero, dense_tensor_i<N, double> &blk,
        const index<N> &i, const tensor_transf<N, double> &tr,
        const double &c);

    //@}

private:
    void make_symmetry();
    void make_schedule();
    void make_schedule_blk(const abs_index<N> &ai,
        sym_schedule_t &sch) const;

private:
    btod_symmetrize3(const btod_symmetrize3<N>&);
    const btod_symmetrize3<N> &operator=(const btod_symmetrize3<N>&);

};


template<size_t N>
const char *btod_symmetrize3<N>::k_clazz = "btod_symmetrize3<N>";


template<size_t N>
btod_symmetrize3<N>::btod_symmetrize3(additive_bto<N, bto_traits<double> > &op, size_t i1,
    size_t i2, size_t i3, bool symm) :

    m_op(op), m_i1(i1), m_i2(i2), m_i3(i3), m_symm(symm),
    m_sym(op.get_bis()),
    m_sch(op.get_bis().get_block_index_dims()) {

    static const char *method = "btod_symmetrize3(additive_bto<N, bto_traits<double> >&, "
        "size_t, size_t, size_t, bool)";

    if(i1 == i2 || i2 == i3 || i1 == i3) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "i1,i2,i3");
    }

    if(m_i1 > m_i2) std::swap(m_i1, m_i2);
    if(m_i2 > m_i3) std::swap(m_i2, m_i3);
    if(m_i1 > m_i2) std::swap(m_i1, m_i2);

    make_symmetry();
    make_schedule();
}

/*
template<size_t N>
void btod_symmetrize3<N>::compute_block(dense_tensor_i<N, double> &blk,
    const index<N> &i) {

    typedef typename sym_schedule_t::iterator iterator_t;

    dimensions<N> bidims(m_op.get_bis().get_block_index_dims());
    abs_index<N> ai(i, bidims);

    sym_schedule_t sch;
    make_schedule_blk(ai, sch);

    tod_set<N>().perform(blk);
    compute_block(blk, i, tensor_transf<N, double>(), 1.0);
}*/


template<size_t N>
void btod_symmetrize3<N>::compute_block(bool zero,
    dense_tensor_i<N, double> &blk, const index<N> &i,
    const tensor_transf<N, double> &tr, const double &c) {

    typedef typename sym_schedule_t::iterator iterator_t;

    if(zero) tod_set<N>().perform(blk);

    dimensions<N> bidims(m_op.get_bis().get_block_index_dims());
    abs_index<N> ai(i, bidims);

    sym_schedule_t sch;
    make_schedule_blk(ai, sch);

    std::pair<iterator_t, iterator_t> jr =
        sch.equal_range(ai.get_abs_index());
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
            tensor_transf<N, double> tri(sch1.front().tr);
            tri.transform(tr);
            additive_bto<N, bto_traits<double> >::compute_block(m_op, false,
                blk, ai.get_index(), tri, c);
            sch1.pop_front();
        } else {
            dimensions<N> dims(blk.get_dims());
            dims.permute(permutation<N>(tr.get_perm(), true));
            dims.permute(permutation<N>(sch1.front().tr.get_perm(),
                true));
            // TODO: replace with "temporary block" feature
            dense_tensor< N, double, allocator<double> > tmp(dims);
            additive_bto<N, bto_traits<double> >::compute_block(m_op, true, tmp,
                ai.get_index(), tensor_transf<N, double>(), c);
            for(typename std::list<schrec>::iterator j =
                sch1.begin(); j != sch1.end();) {

                if(j->ai != ai.get_abs_index()) {
                    ++j; continue;
                }
                tensor_transf<N, double> trj(j->tr);
                trj.transform(tr);
                tod_copy<N>(tmp, trj.get_perm(),
                    trj.get_scalar_tr().get_coeff()).perform(false, 1.0, blk);
                j = sch1.erase(j);
            }
        }
    }
}


template<size_t N>
void btod_symmetrize3<N>::make_symmetry() {

    sequence<N, size_t> seq1, seq2;
    seq1[m_i1] = 1; seq1[m_i2] = 2; seq1[m_i3] = 3;
    seq2[m_i1] = seq2[m_i2] = seq2[m_i3] = 1;
    scalar_transf<double> tr0, tr1(-1.);
    so_symmetrize<N, double>(m_op.get_symmetry(),
            seq1, seq2, m_symm ? tr0 : tr1, tr0).perform(m_sym);

}


template<size_t N>
void btod_symmetrize3<N>::make_schedule() {

    btod_symmetrize3<N>::start_timer("make_schedule");

    dimensions<N> bidims(m_op.get_bis().get_block_index_dims());
    orbit_list<N, double> ol(m_sym);

    for(typename orbit_list<N, double>::iterator io = ol.begin();
        io != ol.end(); io++) {

        abs_index<N> ai(ol.get_index(io), bidims);
        sym_schedule_t sch;
        make_schedule_blk(ai, sch);
        if(!sch.empty()) m_sch.insert(ai.get_abs_index());
    }

    btod_symmetrize3<N>::stop_timer("make_schedule");
}


template<size_t N>
void btod_symmetrize3<N>::make_schedule_blk(const abs_index<N> &ai,
    sym_schedule_t &sch) const {

    permutation<N> perm1, perm2, perm3;
    perm1.permute(m_i1, m_i2);
    perm2.permute(m_i1, m_i3);
    perm3.permute(m_i2, m_i3);
    double scal = m_symm ? 1.0 : -1.0;

    index<N> idx0(ai.get_index()), idx1(idx0), idx2(idx0), idx3(idx0),
        idx4(idx0), idx5(idx0);
    idx1.permute(perm1);
    idx2.permute(perm2);
    idx3.permute(perm3);
    idx4.permute(perm1).permute(perm2);
    idx5.permute(perm1).permute(perm3);

    const symmetry<N, double> &sym0 = m_op.get_symmetry();
    const assignment_schedule<N, double> &sch0 = m_op.get_schedule();

    orbit<N, double> o0(sym0, idx0), o1(sym0, idx1), o2(sym0, idx2),
        o3(sym0, idx3), o4(sym0, idx4), o5(sym0, idx5);

    //  This is a temporary schedule for the formation of the block
    std::list<schrec> sch1;

    //  Form the temporary schedule

    if(sch0.contains(o0.get_abs_canonical_index())) {
        tensor_transf<N, double> tr(o0.get_transf(idx0));
        sch1.push_back(schrec(o0.get_abs_canonical_index(), tr));
    }
    if(sch0.contains(o1.get_abs_canonical_index())) {
        tensor_transf<N, double> tr(o1.get_transf(idx1));
        tr.permute(perm1);
        tr.transform(scalar_transf<double>(scal));
        sch1.push_back(schrec(o1.get_abs_canonical_index(), tr));
    }
    if(sch0.contains(o2.get_abs_canonical_index())) {
        tensor_transf<N, double> tr(o2.get_transf(idx2));
        tr.permute(perm2);
        tr.transform(scalar_transf<double>(scal));
        sch1.push_back(schrec(o2.get_abs_canonical_index(), tr));
    }
    if(sch0.contains(o3.get_abs_canonical_index())) {
        tensor_transf<N, double> tr(o3.get_transf(idx3));
        tr.permute(perm3);
        tr.transform(scalar_transf<double>(scal));
        sch1.push_back(schrec(o3.get_abs_canonical_index(), tr));
    }
    if(sch0.contains(o4.get_abs_canonical_index())) {
        tensor_transf<N, double> tr(o4.get_transf(idx4));
        tr.permute(perm1);
        tr.permute(perm3);
        sch1.push_back(schrec(o4.get_abs_canonical_index(), tr));
    }
    if(sch0.contains(o5.get_abs_canonical_index())) {
        tensor_transf<N, double> tr(o5.get_transf(idx5));
        tr.permute(perm1);
        tr.permute(perm2);
        sch1.push_back(schrec(o5.get_abs_canonical_index(), tr));
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
            tensor_transf<N, double> tr;
            tr.permute(tr0.get_perm());
            tr.transform(scalar_transf<double>(c));
            sch.insert(sym_schedule_pair_t(ai.get_abs_index(),
                schrec(aidx.get_abs_index(), tr)));
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMMETRIZE3_H
