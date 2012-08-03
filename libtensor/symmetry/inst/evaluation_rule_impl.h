#ifndef LIBTENSOR_EVALUATION_RULE_IMPL_H
#define LIBTENSOR_EVALUATION_RULE_IMPL_H

#include <list>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/permutation_generator.h>
#include <libtensor/core/sequence_generator.h>
#include "../bad_symmetry.h"


namespace libtensor {


template<size_t N>
const char *evaluation_rule<N>::k_clazz = "evaluation_rule<N>";


template<size_t N>
void evaluation_rule<N>::optimize() {

    typedef typename eval_sequence_list<N>::eval_sequence_t eval_sequence_t;

    std::list<product_rule_t> new_rules;
    eval_sequence_list<N> new_slist;

    // Loop over all products
    typename std::list<product_rule_t>::iterator it1 = m_rules.begin();
    for (; it1 != m_rules.end(); it1++) {

        product_rule_t &pr = *it1;
        // Delete empty products
        if (pr.empty()) continue;

        product_rule_t prx(&new_slist);

        // Look for products with 'all allowed' rules
        size_t nallowed = 0;
        typename product_rule_t::iterator ip1 = pr.begin();
        for (; ip1 != pr.end(); ip1++) {

            if (pr.get_intrinsic(ip1) == product_table_i::k_invalid) {
                nallowed++;
                continue;
            }

            const eval_sequence_t &seq = pr.get_sequence(ip1);
            size_t nidx = 0;
            for (register size_t j = 0; j < N; j++) nidx += seq[j];
            if (nidx == 0) {
                if (pr.get_intrinsic(ip1) == product_table_i::k_identity) {
                    nallowed++;
                    continue;
                }
                else {
                    break;
                }
            }

            prx.add(pr.get_sequence(ip1), pr.get_intrinsic(ip1));
        }

        // If there was one forbidden term the product can be deleted
        if (ip1 != it1->end()) continue;

        if (prx.empty()) {
            if (nallowed != 0) break;
            else continue;
        }

        new_rules.push_back(prx);
    }

    // All blocks are allowed by this rule
    if (it1 != m_rules.end()) {
        m_slist.clear();
        m_rules.clear();

        eval_sequence_t seq(1);
        product_rule_t pr(&m_slist);
        pr.add(seq, product_table_i::k_invalid);
        m_rules.push_back(pr);
        return;
    }

    // Remove duplicate products
    for (it1 = new_rules.begin(); it1 != new_rules.end(); it1++) {

        typename std::list<product_rule_t>::iterator it2 = it1;
        it2++;
        while (it2 != new_rules.end()) {
            if (*it1 == *it2) it2 = new_rules.erase(it2);
            else it2++;
        }
    }

    // Clear the member lists
    m_rules.clear();
    m_slist.clear();

    // Copy from new lists to member lists
    for (it1 = new_rules.begin(); it1 != new_rules.end(); it1++) {

        product_rule<N> &pr = new_product();
        for (typename product_rule<N>::iterator ip1 = it1->begin();
                ip1 != it1->end(); ip1++) {
            pr.add(it1->get_sequence(ip1), it1->get_intrinsic(ip1));
        }
    }
}


template<size_t N> template<size_t M>
void evaluation_rule<N>::reduce(evaluation_rule<N - M> &res,
        const sequence<N, size_t> &rmap,
        const sequence<M, label_group_t> &rdims,
        const product_table_i &pt) const {

    typedef typename eval_sequence_list<N>::eval_sequence_t eval_sequence_t;
    typedef typename product_rule_t::iterator term_iterator;

#ifdef LIBTENSOR_DEBUG
    size_t nrsteps = 0;
    for (; nrsteps < M && rdims[nrsteps].size() > 0; nrsteps++) ;
    for (size_t i = 0; i < N; i++) {
        if (rmap[i] < N - M) continue;
        if (rmap[i] - (N - M) >= nrsteps) {
            throw bad_symmetry(g_ns, k_clazz, "reduce(...)",
                    __FILE__, __LINE__, "rmap");
        }
    }
#endif
    res.clear();

    // Loop over products
    for (const_iterator it = m_rules.begin(); it != m_rules.end(); it++) {

        const product_rule_t &pra = *it;

        // Determine rsteps present in product and sequences contributing to
        // rstep
        sequence<M, size_t> rsteps_in_pr(0);
        sequence<M, std::set<size_t> > terms_in_rstep;
        std::vector<term_iterator> terms;

        size_t iterm = 0;
        for (term_iterator ip = pra.begin(); ip != pra.end(); ip++, iterm++) {

            const eval_sequence_t &seq = pra.get_sequence(ip);
            terms.push_back(ip);

            for (size_t i = 0; i < N; i++) {

                if (seq[i] == 0 || rmap[i] < N - M) continue;

                size_t rstep = rmap[i] - (N - M);
                rsteps_in_pr[rstep] += seq[i];
                terms_in_rstep[rstep].insert(iterm);
            }
        }

        // Loop over all rsteps and find lists of terms that can be merged
        mask<M> rsteps_to_do;
        std::vector< std::set<size_t> > s2c;
        for (size_t i = 0; i < M && ! rdims[i].empty(); i++) {
            if (rsteps_in_pr[i] == 0) continue;
            if (rsteps_in_pr[i] != 2 || terms_in_rstep[i].size() != 2 ||
                    rdims[i].size() != pt.get_n_labels()) {
                rsteps_to_do[i] = true;
                continue;
            }

            std::set<size_t>::iterator it1, it2;
            it1 = it2 = terms_in_rstep[i].begin();
            it2++;

            typename std::vector< std::set<size_t> >::iterator j1 = s2c.end();
            typename std::vector< std::set<size_t> >::iterator j2 = s2c.end();
            for (typename std::vector< std::set<size_t> >::iterator j =
                    s2c.begin(); j != s2c.end(); j++) {
                if (j1 != s2c.end() && j2 != s2c.end()) break;
                if (j1 == s2c.end() && j->count(*it1) != 0) { j1 = j; }
                if (j2 == s2c.end() && j->count(*it2) != 0) { j2 = j; }
            }

            if (j1 == s2c.end() && j2 == s2c.end()) {
                std::set<size_t> its;
                its.insert(*it1);
                its.insert(*it2);
                s2c.push_back(its);
            }
            else if (j2 == s2c.end()) {
                j1->insert(*it2);
            }
            else if (j1 == s2c.end()) {
                j2->insert(*it1);
            }
            else if (j1 != j2) {
                j1->insert(j2->begin(), j2->end());
                s2c.erase(j2);
            }
        }

        // Combine terms
        eval_sequence_list<N> seq_list;
        std::list<label_group_t> intr_list;
        intr_list.push_back(label_group_t());

        std::set<size_t> terms_done;
        for (typename std::vector< std::set<size_t> >::iterator itsl =
                s2c.begin(); itsl != s2c.end(); itsl++) {

            eval_sequence_t seq2(0);

            label_group_t lg;
            for (typename std::set<size_t>::iterator its = itsl->begin();
                    its != itsl->end(); its++) {

                term_iterator itx = terms[*its];
                const eval_sequence_t &seq1 = pra.get_sequence(itx);
                for (size_t i = 0; i < N; i++) seq2[rmap[i]] += seq1[i];

                lg.push_back(pra.get_intrinsic(itx));
                terms_done.insert(*its);
            }
            seq_list.add(seq2);

            label_set_t ls(pt.product(lg));
            std::list<label_group_t>::iterator iti = intr_list.begin();
            while (iti != intr_list.end()) {
                if (ls.size() != 1) {
                    std::list<label_group_t>::iterator itj = iti;
                    itj++;
                    intr_list.insert(itj, ls.size() - 1, *iti);
                }

                for (label_set_t::iterator il =  ls.begin();
                        il != ls.end(); il++, iti++) {
                    iti->push_back(*il);
                }
            }
        }

        // Add remaining terms to lists
        iterm = 0;
        for (term_iterator ip = pra.begin(); ip != pra.end(); ip++, iterm++) {

            if (terms_done.count(iterm) != 0) continue;

            eval_sequence_t seq2(0);
            const eval_sequence_t &seq1 = pra.get_sequence(ip);
            for (size_t i = 0; i < N; i++) seq2[rmap[i]] += seq1[i];

            seq_list.add(seq2);
            for (std::list<label_group_t>::iterator iti = intr_list.begin();
                    iti != intr_list.end(); iti++) {
                iti->push_back(pra.get_intrinsic(ip));
            }
        }

        // Loop over all remaining reduction indexes
        index<M> idx1, idx2;
        for (size_t i = 0; i < M; i++) {
            if (! rsteps_to_do[i]) continue;

            idx2[i] = rdims[i].size() - 1;
        }


        abs_index<M> aridx(dimensions<M>(index_range<M>(idx1, idx2)));
        std::list<label_group_t> intr2_list;
        do {

            const index<M> &ridx = aridx.get_index();

            // Loop over all reduction sequences
            std::list<label_group_t> new_intr(intr_list);
            for (size_t i = 0; i < seq_list.size(); i++) {

                // Get current sequence
                const sequence<N, size_t> &cur_seq = seq_list[i];

                // Create label group from reduction sequence at current index
                label_group_t lg;
                for (size_t j = 0; j < M; j++) {
                    if (! rsteps_to_do[j]) continue;
                    lg.insert(lg.end(), cur_seq[j + N - M], rdims[j][ridx[j]]);
                }

                // Loop over all intrinsic label lists
                std::list<label_group_t>::iterator iti = new_intr.begin();
                while (iti != new_intr.end()) {

                    lg.push_back(iti->at(i));
                    label_set_t ls(pt.product(lg));
                    lg.pop_back();

                    if (ls.size() != 1) {
                        std::list<label_group_t>::iterator itj = iti;
                        itj++;
                        new_intr.insert(itj, ls.size() - 1, *iti);
                    }
                    for (label_set_t::iterator is = ls.begin();
                            is != ls.end(); is++, iti++) {
                        iti->at(i) = *is;
                    }
                }
            }
            intr2_list.insert(intr2_list.end(),
                    new_intr.begin(), new_intr.end());

        } while (aridx.inc());

        intr_list.clear();

        eval_sequence_list<N - M> seq2_list;
        for (size_t i = 0; i < seq_list.size(); i++) {
            sequence<N - M, size_t> seq2(0);
            for (register size_t j = 0; j < N - M; j++)
                seq2[j] = seq_list[i][j];
            seq2_list.add(seq2);
        }
        for (std::list<label_group_t>::const_iterator it = intr2_list.begin();
                it != intr2_list.end(); it++) {

            product_rule<N - M> &pr = res.new_product();
            for (size_t i = 0; i < seq2_list.size(); i++) {
                pr.add(seq2_list[i], it->at(i));
            }
        }
    }
    res.optimize();
}


template<size_t N> template<size_t M>
void evaluation_rule<N>::merge(evaluation_rule<M> &res,
        const sequence<N, size_t> &mmap,
        const mask<M> &smsk) const {

    // Loop over products
    for (const_iterator it = m_rules.begin(); it != m_rules.end(); it++) {

        const product_rule_t &pra = *it;
        product_rule<M> &prb = res.new_product();

        for (typename product_rule_t::iterator ip = pra.begin();
                ip != pra.end(); ip++) {

            const sequence<N, size_t> &seq1 = pra.get_sequence(ip);
            sequence<M, size_t> seq2(0);
            for (register size_t i = 0; i < N; i++) {
                seq2[mmap[i]] += seq1[i];
            }

            size_t nidx = 0;
            for (register size_t i = 0; i < M; i++) {
                if (! smsk[i]) continue;

                seq2[i] = seq2[i] % 2;
            }

            prb.add(seq2, pra.get_intrinsic(ip));
        }
    }
    res.optimize();
}


template<size_t N>
bool evaluation_rule<N>::is_allowed(const sequence<N, label_t> &blk_labels,
        const product_table_i &pt) const {

    // Loop over all sequences in rule and determine result labels
    std::vector<label_set_t> ls(m_slist.size());
    for (size_t i = 0; i < m_slist.size(); i++) {
        const sequence<N, size_t> &seq = m_slist[i];
        label_group_t lg;
        register size_t j = 0;
        for (; j < N; j++) {
            if (seq[j] == 0) continue;
            if (blk_labels[j] == product_table_i::k_invalid) break;

            lg.insert(lg.end(), seq[j], blk_labels[j]);
        }
        if (j != N) {
            for (label_t ll = 0; ll < pt.get_n_labels(); ll++)
                ls[i].insert(ll);
        }
        else if (lg.size() != 0) {
            ls[i] = pt.product(lg);
        }
    }

    // Loop over all products in the evaluation rule
    for (const_iterator it = m_rules.begin(); it != m_rules.end(); it++) {

        // Loop over all terms in the current product
        typename product_rule<N>::iterator ip = it->begin();
        for (; ip != it->end(); ip++) {

            // Invalid intrinsic label
            if (ip->second == product_table_i::k_invalid) continue;

            if (ls[ip->first].count(ip->second) == 0) break;
        }

        if (ip == it->end()) {
            return true;
        }
    }

    return false;
}


} // namespace libtensor


#endif // LIBTENSOR_EVALUATION_RULE_IMPL_H
