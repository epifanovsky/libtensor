#ifndef LIBTENSOR_ER_REDUCE_IMPL_H
#define LIBTENSOR_ER_REDUCE_IMPL_H

#include <libtensor/core/abs_index.h>
#include "../product_table_container.h"
#include "../bad_symmetry.h"


namespace libtensor {


template<size_t N, size_t M>
const char *er_reduce<N, M>::k_clazz = "er_reduce<N, M>";


template<size_t N, size_t M>
er_reduce<N, M>::er_reduce(
	const evaluation_rule<N> &rule, const sequence<N, size_t> &rmap,
	const sequence<M, label_group_t> &rdims, const std::string &id) :
	m_rule(rule),
	m_pt(product_table_container::get_instance().req_const_table(id)),
	m_rmap(rmap), m_rdims(rdims), m_nrsteps(0) {

    for (; m_nrsteps < M && ! m_rdims[m_nrsteps].empty(); m_nrsteps++) ;

#ifdef LIBTENSOR_DEBUG
    for (size_t i = 0; i < N; i++) {
        if (m_rmap[i] < N - M) continue;
        if ((m_rmap[i] - N + M)  >= m_nrsteps) {
            throw bad_symmetry(g_ns, k_clazz, "er_reduce(...)",
                    __FILE__, __LINE__, "rmap");
        }
    }
#endif

}


template<size_t N, size_t M>
er_reduce<N, M>::~er_reduce() {

    product_table_container::get_instance().ret_table(m_pt.get_id());
}


template<size_t N, size_t M>
void er_reduce<N, M>::perform(evaluation_rule<N - M> &to) const {

    er_reduce<N, M>::start_timer();

    to.clear();

    const eval_sequence_list<N> &slist = m_rule.get_sequences();

    // Collect the rsteps present in each sequence
    std::vector<size_t> rsteps_in_seq(slist.size() * m_nrsteps, 0);
    for (size_t i = 0, pos = 0; i < slist.size(); i++, pos += m_nrsteps) {

        const sequence<N, size_t> &seq = slist[i];
        for (size_t j = 0; j < N; j++) {
            if (seq[j] == 0 || m_rmap[j] < N - M) continue;

            rsteps_in_seq[pos + m_rmap[j] - (N - M)] += seq[j];
        }
    }


    // Loop over products
    for (typename evaluation_rule<N>::iterator it = m_rule.begin();
            it != m_rule.end(); it++) {

        const product_rule<N> &pra = m_rule.get_product(it);

        if (! reduce_product(pra, slist,  rsteps_in_seq, to)) {
            to.clear();
            product_rule<N - M> &pr = to.new_product();
            pr.add(sequence<N - M, size_t>(1), product_table_i::k_invalid);
            return;
        }
    } // End for it

    er_reduce<N, M>::stop_timer();
}


template<size_t N, size_t M>
bool er_reduce<N, M>::reduce_product(const product_rule<N> &pr,
        const eval_sequence_list<N> &slist,
        const std::vector<size_t> &rsteps_in_seq,
        evaluation_rule<N - M> &to) const {

    // 1) Analyze product:
    //  - create a map of terms: seqno->intrinsic label(s)
    //  - determine # seq in the product which participate to a
    //    reduction step

    std::map<size_t, label_set_t> term_map;
    sequence< M, std::set<size_t> > seq_in_rstep;
    for (typename product_rule<N>::iterator it = pr.begin();
            it != pr.end(); it++) {

        size_t seqno = pr.get_seqno(it);
        term_map[seqno].insert(pr.get_intrinsic(it));

        for (size_t i = 0, pos = seqno * m_nrsteps;
                i < m_nrsteps; i++, pos++) {

            if (m_rdims[i].size() != m_pt.get_n_labels() ||
                    rsteps_in_seq[pos] == 0) continue;

            seq_in_rstep[i].insert(seqno);
        }
    }

    // 2) Analyse reduction steps in this product
    //  - Mark reduction steps which have to be performed explicitly
    //  - Populate graph of sequences participating in the reduction steps

    adjacency_list adj_list;
    mask<M> rsteps_to_do;

    for (size_t i = 0; i < m_nrsteps; i++) {
        if (m_rdims[i].size() != m_pt.get_n_labels() ||
                seq_in_rstep[i].size() != 2) {

            rsteps_to_do[i] = true;
            continue;
        }

        size_t seqno1 = *(seq_in_rstep[i].begin());
        size_t seqno2 = *(seq_in_rstep[i].rbegin());

        if (seqno1 == seqno2) continue;

        if (rsteps_in_seq[seqno1 * m_nrsteps + i] +
                rsteps_in_seq[seqno2 * m_nrsteps + i] != 2) {
            rsteps_to_do[i] = true;
            continue;
        }

        adj_list.add(seqno1, seqno2);
    } // End for i

    // 3) Combine terms where possible

    std::vector< sequence<N - M, size_t> > pr_seqs;
    std::vector< sequence<M, size_t> > red_seqs;
    std::vector< bool > zero_seqs;
    std::list< std::vector<label_set_t> > intr_list;

    std::set<size_t> terms_done;
    for (typename product_rule<N>::iterator it = pr.begin();
            it != pr.end(); it++) {

        size_t seqno = pr.get_seqno(it);
        if (terms_done.count(seqno) != 0) continue;

        // Get list of terms which connect to the current sequence due to some
        // reduction step
        std::vector<size_t> conn_list;
        adj_list.get_connected(seqno, conn_list);
        if (conn_list.empty()) conn_list.push_back(seqno);

        // Combine the sequences in the list of connected terms
        // and append them to the sequence lists for the current product
        size_t nidx = append_seq(slist, conn_list, pr_seqs, red_seqs);
        zero_seqs.push_back(nidx == 0);

        // Get the max number reduction steps performed on the same two
        // sequences
        size_t nmrsteps = get_rstep_multiplicity(adj_list, conn_list);

        // Collect the intrinsic labels which have to be combined
        std::vector<label_set_t> cilist;
        for (std::vector<size_t>::iterator ic = conn_list.begin();
                ic != conn_list.end(); ic++) {
            cilist.push_back(term_map[*ic]);
        }

        append_intr(cilist, nmrsteps, intr_list);

        // Mark terms done
        for (std::vector<size_t>::iterator ic = conn_list.begin();
                ic != conn_list.end(); ic++) {
            terms_done.insert(*ic);
        }
    } // End for it

    // 4) Loop over the remaining rsteps
    // Loop over all remaining reduction indexes

    size_t nrsteps_to_do = 0;
    index<M> idx1, idx2;
    for (size_t i = 0; i < m_nrsteps; i++) {
        if (! rsteps_to_do[i]) continue;

        nrsteps_to_do++;
        idx2[i] = m_rdims[i].size() - 1;
    }

    if (nrsteps_to_do != 0) {

        abs_index<M> aridx(dimensions<M>(index_range<M>(idx1, idx2)));
        do {

            const index<M> &ridx = aridx.get_index();

            // Create a vector of labels by which the intrisic labels have to
            // be amended
            std::vector<label_group_t> rstep_labels(red_seqs.size());
            for (size_t i = 0; i < red_seqs.size(); i++) {

                const sequence<M, size_t> &rseq = red_seqs[i];

                for (size_t j = 0; j < m_nrsteps; j++) {
                    if (! rsteps_to_do[j]) continue;

                    rstep_labels[i].insert(rstep_labels[i].end(),
                            rseq[j], m_rdims[j][ridx[j]]);
                }
            }

            for (std::list< std::vector<label_set_t> >::iterator il =
                    intr_list.begin(); il != intr_list.end(); il++) {

                std::list<label_group_t> rlist;
                {
                    std::vector<label_set_t> rvec;
                    for (size_t i = 0; i < red_seqs.size(); i++) {

                        label_group_t &intr = rstep_labels[i];
                        for (label_set_t::iterator it = il->at(i).begin();
                                it != il->at(i).end(); it++) {

                            label_set_t ls;
                            intr.push_back(*it);
                            m_pt.product(intr, ls);
                            intr.pop_back();
                            rvec.push_back(ls);
                        }
                    }

                    create_list(rvec, rlist);
                }

                for (std::list<label_group_t>::iterator ir = rlist.begin();
                        ir != rlist.end(); ir++) {

                    std::vector<label_set_t> cur(red_seqs.size());
                    size_t length = 0;

                    for (size_t i = 0, j = 0; i < red_seqs.size(); i++) {

                        if (zero_seqs[i]) {
                            label_set_t::iterator it = il->at(i).begin();
                            for (; it != il->at(i).end(); it++, j++) {
                                if (ir->at(j) != product_table_i::k_identity &&
                                        ir->at(j) != product_table_i::k_invalid)
                                    break;
                            }
                            if (it != il->at(i).end()) {
                                cur.clear();
                                break;
                            }

                            continue;
                        }

                        for (label_set_t::iterator it = il->at(i).begin();
                                it != il->at(i).end(); it++, j++, length++) {

                            cur[i].insert(ir->at(j));
                        }
                    }

                    // All forbidden product
                    if (cur.empty()) continue;
                    // All allowed product
                    if (length == 0) return false;


                    product_rule<N - M> &pr = to.new_product();
                    for (size_t i = 0; i < cur.size(); i++) {
                        for (label_set_t::iterator it = cur[i].begin();
                                it != cur[i].end(); it++) {
                            pr.add(pr_seqs[i], *it);
                        }
                    }
                }
            }

        } while (aridx.inc());
    }
    else {

        for (std::list< std::vector<label_set_t> >::iterator il =
                intr_list.begin(); il != intr_list.end(); il++) {

            std::vector<label_set_t> cur(il->size());
            size_t length = 0;

            for (size_t i = 0; i < il->size(); i++) {

                label_set_t &ls = il->at(i);

                if (zero_seqs[i]) {
                    if (ls.count(product_table_i::k_identity) == 0 &&
                            ls.count(product_table_i::k_invalid) == 0) {

                        cur.clear();
                        break;
                    }
                    continue;
                }

                cur[i].insert(ls.begin(), ls.end());
                length += ls.size();
            }

            if (cur.empty()) continue;
            if (length == 0) return false;

            product_rule<N - M> &pr = to.new_product();
            for (size_t i = 0; i < cur.size(); i++) {
                for (label_set_t::iterator it = cur[i].begin();
                        it != cur[i].end(); it++) {
                    pr.add(pr_seqs[i], *it);
                }
            }
        }
    }

    return true;
}


template<size_t N, size_t M>
size_t er_reduce<N, M>::append_seq(const eval_sequence_list<N> &slist,
        const std::vector<size_t> &clist,
        std::vector< sequence<N - M, size_t> > &seq_list,
        std::vector< sequence<M, size_t> > &rseq_list) const {

    seq_list.push_back(sequence<N - M, size_t>());
    sequence<N - M, size_t> &seq = seq_list.back();

    rseq_list.push_back(sequence<M, size_t>());
    sequence<M, size_t> &rseq = rseq_list.back();

    size_t nidx = 0;
    for (std::vector<size_t>::const_iterator ic = clist.begin();
            ic != clist.end(); ic++) {

        const sequence<N, size_t> &seqx = slist[*ic];
        for (size_t i = 0; i < N; i++) {
            if (m_rmap[i] < N - M) {
                nidx += seqx[i];
                seq[m_rmap[i]] += seqx[i];
            }
            else {
                rseq[m_rmap[i] - (N - M)] += seqx[i];
            }
        }
    }

    return nidx;
}


template<size_t N, size_t M>
void er_reduce<N, M>::append_intr(
        const std::vector<label_set_t> &cilist, size_t nmrsteps,
        std::list< std::vector<label_set_t> > &ilist) const {

    if (ilist.empty()) ilist.push_back(std::vector<label_set_t>());

    if (nmrsteps == 0) {
        if (cilist.size() != 1)
            throw bad_symmetry(g_ns, k_clazz, "append_intr(...)",
                    __FILE__, __LINE__, "nmrsteps");

        for (std::list< std::vector<label_set_t> >::iterator il =
                ilist.begin(); il != ilist.end(); il++) {
            il->push_back(*(cilist.begin()));
        }

        return;
    }

    // Convert cilist
    std::list<label_group_t> cilist2;
    create_list(cilist, cilist2);

    // Get the result labels of the product
    // \sigma_1\times\cdots\times\sigma_{n-1}
    label_set_t product_set;
    get_product_labels(nmrsteps - 1, product_set);

    // Combine intrinsic labels
    std::list<label_group_t> rilist;
    {
        std::vector<label_set_t> rivec;
        if (product_set.size() != 0) {

            for (std::list<label_group_t>::iterator ic = cilist2.begin();
                    ic != cilist2.end(); ic++) {

                label_set_t ls;
                for (label_set_t::iterator ip = product_set.begin();
                        ip != product_set.end(); ip++) {

                    label_set_t lsx;
                    ic->push_back(*ip);
                    m_pt.product(*ic, lsx);
                    ic->pop_back();

                    ls.insert(lsx.begin(), lsx.end());
                }
                rivec.push_back(ls);
            }
        }
        else {

            for (std::list<label_group_t>::iterator ic = cilist2.begin();
                    ic != cilist2.end(); ic++) {

                label_set_t ls;
                m_pt.product(*ic, ls);
                rivec.push_back(ls);
            }
        }

        create_list(rivec, rilist);
    }

    // Append the combined labels to ilist
    std::list< std::vector<label_set_t> >::iterator il = ilist.begin();
    while (il != ilist.end()) {

        for (std::list<label_group_t>::iterator ir = rilist.begin();
                ir != rilist.end(); ir++) {

            label_set_t ls;
            for (label_group_t::iterator it = ir->begin();
                    it != ir->end(); it++) ls.insert(*it);

            ilist.insert(il, *il)->push_back(ls);
        }

        il = ilist.erase(il);
    }
}

template<size_t N, size_t M>
size_t er_reduce<N, M>::get_rstep_multiplicity(const adjacency_list &alist,
        const std::vector<size_t> &clist) const {

    size_t nrm = 0;
    for (std::vector<size_t>::const_iterator it = clist.begin();
            it != clist.end(); it++) {

        std::vector<size_t> nlist;
        alist.get_next_neighbours(*it, nlist);

        for (std::vector<size_t>::const_iterator in = nlist.begin();
                in != nlist.end(); in++) {

            nrm = std::max(nrm, alist.weight(*it, *in));
        }
    }

    return nrm;
}


template<size_t N, size_t M>
void er_reduce<N, M>::create_list(const std::vector<label_set_t> &in,
        std::list<label_group_t> &out) const {

    std::vector<label_set_t::const_iterator> itlst;
    for (std::vector<label_set_t>::const_iterator it = in.begin();
            it != in.end(); it++) {
        itlst.push_back(it->begin());
    }

    while (itlst.back() != in.back().end()) {

        label_group_t lg;
        for (size_t i = 0; i < itlst.size(); i++)
            lg.push_back(*(itlst[i]));

        out.push_back(lg);

        for (size_t i = 0; i < itlst.size(); i++) {
            for (size_t j = 0; j < i; j++) itlst[j] = in[j].begin();

            itlst[i]++;
            if (itlst[i] != in[i].end()) break;
        }
    }
}


template<size_t N, size_t M>
void er_reduce<N, M>::get_product_labels(size_t n, label_set_t &ls) const {

    ls.clear();
    if (n == 0) return;

    for (product_table_i::label_t l = 0; l != m_pt.get_n_labels(); l++) {
        label_group_t lg(2, l);

        label_set_t lsx;
        m_pt.product(lg, lsx);

        ls.insert(lsx.begin(), lsx.end());
    }

    if (n == 1) return;

    std::vector<label_set_t::const_iterator> vls(n);
    for (size_t i = 0; i < n; i++) vls[i] = ls.begin();

    label_set_t ls2;
    while (vls[n - 1] != ls.end()) {

        label_group_t lg(n);
        for (size_t i = 0; i < n; i++) lg[i] = *(vls[i]);

        label_set_t lsx;
        m_pt.product(lg, lsx);
        ls2.insert(lsx.begin(), lsx.end());

        for (size_t i = 0; i < n; i++) {
            vls[i]++;
            if (vls[i] == ls.end() && i != n - 1) vls[i] = ls.begin();
            else break;
        }
    }

    ls.clear();
    ls.insert(ls2.begin(), ls2.end());
}

} // namespace libtensor


#endif // LIBTENSOR_ER_REDUCE_IMPL_H
