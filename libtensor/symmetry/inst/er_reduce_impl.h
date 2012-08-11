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
        const sequence<M, label_group_t> rdims, const std::string &id) :
        m_rule(rule), m_rmap(rmap), m_rdims(rdims),
        m_pt(product_table_container::get_instance().req_const_table(id)) {

#ifdef LIBTENSOR_DEBUG
    size_t nrsteps = 0;
    for (; nrsteps < M && m_rdims[nrsteps].size() > 0; nrsteps++) ;
    for (size_t i = 0; i < N; i++) {
        if (m_rmap[i] < N - M) continue;
        if (m_rmap[i] - (N - M) >= nrsteps) {
            throw bad_symmetry(g_ns, k_clazz, "er_reduce(...)",
                    __FILE__, __LINE__, "rmap");
        }
    }
#endif

}



template<size_t N, size_t M>
void er_reduce<N, M>::perform(evaluation_rule<N - M> &rule) const {

    er_reduce<N, M>::start_timer();

    rule.clear();

    // Loop over products
    for (typename evaluation_rule<N>::const_iterator it = m_rule.begin();
            it != m_rule.end(); it++) {

        const product_rule<N> &pra = *it;

        // Determine rsteps present in product and sequences contributing to
        // rstep
        sequence<M, size_t> rsteps_in_pr(0);
        sequence<M, std::set<size_t> > terms_in_rstep;
        std::vector<typename product_rule<N>::iterator> terms;

        size_t iterm = 0;
        for (typename product_rule<N>::iterator ip = pra.begin();
                ip != pra.end(); ip++, iterm++) {

            const sequence<N, size_t> &seq = pra.get_sequence(ip);
            terms.push_back(ip);

            for (size_t i = 0; i < N; i++) {

                if (seq[i] == 0 || m_rmap[i] < N - M) continue;

                size_t rstep = m_rmap[i] - (N - M);
                rsteps_in_pr[rstep] += seq[i];
                terms_in_rstep[rstep].insert(iterm);
            }
        }

        // Loop over all rsteps and find lists of terms that can be merged
        mask<M> rsteps_to_do;
        std::vector< std::set<size_t> > s2c;
        for (size_t i = 0; i < M && ! m_rdims[i].empty(); i++) {
            if (rsteps_in_pr[i] == 0) continue;
            if (rsteps_in_pr[i] != 2 || terms_in_rstep[i].size() != 2 ||
                    m_rdims[i].size() != m_pt.get_n_labels()) {
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
            else if (j1 == j2) {
                rsteps_to_do[i] = true;
                continue;
            }
            else {
                j1->insert(j2->begin(), j2->end());
                s2c.erase(j2);
            }
        }

        // Combine terms
        std::vector< sequence<N - M, size_t> > seq_list;
        std::vector< sequence<M, size_t> > rseq_list;
        std::vector<bool> zero_seq;
        std::list<label_group_t> intr_list;
        intr_list.push_back(label_group_t());

        std::set<size_t> terms_done;
        for (typename std::vector< std::set<size_t> >::iterator itsl =
                s2c.begin(); itsl != s2c.end(); itsl++) {

            seq_list.push_back(sequence<N - M, size_t>(0));
            sequence<N - M, size_t> &seq = seq_list.back();

            rseq_list.push_back(sequence<M, size_t>(0));
            sequence<M, size_t> &rseq = rseq_list.back();

            size_t nidx = 0;

            label_group_t lg;
            for (typename std::set<size_t>::iterator its = itsl->begin();
                    its != itsl->end(); its++) {

                typename product_rule<N>::iterator itx = terms[*its];
                const sequence<N, size_t> &seqx = pra.get_sequence(itx);

                for (size_t i = 0; i < N; i++)
                    if (m_rmap[i] < N - M) {
                        nidx += seqx[i];
                        seq[m_rmap[i]] += seqx[i];
                    }
                    else
                        rseq[m_rmap[i] - (N - M)] += seqx[i];

                lg.push_back(pra.get_intrinsic(itx));
                terms_done.insert(*its);
            }
            zero_seq.push_back(nidx == 0);

            product_table_i::label_set_t ls;
            m_pt.product(lg, ls);
            std::list<label_group_t>::iterator iti = intr_list.begin();
            while (iti != intr_list.end()) {
                if (ls.size() != 1) {
                    std::list<label_group_t>::iterator itj = iti;
                    itj++;
                    intr_list.insert(itj, ls.size() - 1, *iti);
                }

                for (product_table_i::label_set_t::iterator il =  ls.begin();
                        il != ls.end(); il++, iti++) {
                    iti->push_back(*il);
                }
            }
        }

        // Add remaining terms to lists
        iterm = 0;
        for (typename product_rule<N>::iterator ip = pra.begin();
                ip != pra.end(); ip++, iterm++) {

            if (terms_done.count(iterm) != 0) continue;

            seq_list.push_back(sequence<N - M, size_t>(0));
            sequence<N - M, size_t> &seq = seq_list.back();

            rseq_list.push_back(sequence<M, size_t>(0));
            sequence<M, size_t> &rseq = rseq_list.back();

            size_t nidx = 0;

            const sequence<N, size_t> &seqx = pra.get_sequence(ip);
            for (size_t i = 0; i < N; i++) {
                if (m_rmap[i] < N - M) {
                    nidx += seqx[i];
                    seq[m_rmap[i]] += seqx[i];
                }
                else
                    rseq[m_rmap[i] - (N - M)] += seqx[i];
            }
            zero_seq.push_back(nidx == 0);

            for (std::list<label_group_t>::iterator iti = intr_list.begin();
                    iti != intr_list.end(); iti++) {
                iti->push_back(pra.get_intrinsic(ip));
            }
        }

        // Loop over all remaining reduction indexes
        index<M> idx1, idx2;
        for (size_t i = 0; i < M; i++) {
            if (! rsteps_to_do[i]) continue;

            idx2[i] = m_rdims[i].size() - 1;
        }


        abs_index<M> aridx(dimensions<M>(index_range<M>(idx1, idx2)));
        std::list<label_group_t> intr2_list;
        do {

            const index<M> &ridx = aridx.get_index();

            // Loop over all reduction sequences
            std::list<label_group_t> new_intr(intr_list);
            for (size_t i = 0; i < seq_list.size(); i++) {

                // Get current sequence
                const sequence<M, size_t> &rseq = rseq_list[i];

                // Create label group from reduction sequence at current index
                label_group_t lg;
                for (size_t j = 0; j < M; j++) {
                    if (! rsteps_to_do[j]) continue;
                    lg.insert(lg.end(), rseq[j], m_rdims[j][ridx[j]]);
                }

                // Loop over all intrinsic label lists
                std::list<label_group_t>::iterator iti = new_intr.begin();
                while (iti != new_intr.end()) {

                    lg.push_back(iti->at(i));
                    product_table_i::label_set_t ls;
                    m_pt.product(lg, ls);
                    lg.pop_back();

                    if (zero_seq[i]) {
                        if (ls.count(product_table_i::k_identity) == 0)
                            iti = new_intr.erase(iti);
                        else {
                            iti->at(i) = product_table_i::k_identity;
                            iti++;
                        }
                    }
                    else {
                        if (ls.size() != 1) {
                            std::list<label_group_t>::iterator itj = iti;
                            itj++;
                            new_intr.insert(itj, ls.size() - 1, *iti);
                        }
                        for (product_table_i::label_set_t::iterator is =
                                ls.begin(); is != ls.end(); is++, iti++) {
                            iti->at(i) = *is;
                        }
                    }
                }
            }
            intr2_list.insert(intr2_list.end(),
                    new_intr.begin(), new_intr.end());

        } while (aridx.inc());

        intr_list.clear();

        for (std::list<label_group_t>::const_iterator it = intr2_list.begin();
                it != intr2_list.end(); it++) {

            product_rule<N - M> &pr = rule.new_product();
            for (size_t i = 0; i < seq_list.size(); i++) {

                pr.add(seq_list[i], it->at(i));
            }
        }
    }
    rule.optimize();

    er_reduce<N, M>::stop_timer();
}


} // namespace libtensor


#endif // LIBTENSOR_ER_REDUCE_IMPL_H
