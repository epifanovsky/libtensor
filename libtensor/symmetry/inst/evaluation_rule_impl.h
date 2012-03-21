#ifndef LIBTENSOR_EVALUATION_RULE_IMPL_H
#define LIBTENSOR_EVALUATION_RULE_IMPL_H

namespace libtensor {

template<size_t N>
const char *evaluation_rule<N>::k_clazz = "evaluation_rule<N>";

template<size_t N>
size_t evaluation_rule<N>::add_sequence(const sequence<N, size_t> &seq) {

    size_t seqno = 0;
    for (; seqno < m_sequences.size(); seqno++) {
        const sequence<N, size_t> &ref = m_sequences[seqno];

        register size_t i = 0;
        for (; i < N; i++) {
            if (seq[i] != ref[i]) break;
        }
        if (i == N) return seqno;
    }

    m_sequences.push_back(seq);
    return m_sequences.size() - 1;
}

template<size_t N>
size_t evaluation_rule<N>::add_product(size_t seq_no,
        label_t intr, label_t target) {
#ifdef LIBTENSOR_DEBUG
    if (seq_no >= m_sequences.size())
        throw bad_parameter(g_ns, k_clazz,
                "add_product(size_t, label_t, label_t)",
                __FILE__, __LINE__, "seq_no.");
#endif

    m_setup.push_back(product_t());
    product_t &pr = m_setup.back();


    pr.insert(add_term(seq_no, intr, target));
    return m_setup.size() - 1;
}

template<size_t N>
void evaluation_rule<N>::add_to_product(size_t no,
        size_t seq_no, label_t intr, label_t target) {

    static const char *method =
            "add_to_product(size_t, size_t, label_t, label_t)";

#ifdef LIBTENSOR_DEBUG
    if (no >= m_setup.size())
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "no");
    if (seq_no >= m_sequences.size())
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "seq_no");
#endif

    product_t &pr = m_setup[no];
    pr.insert(add_term(seq_no, intr, target));
}

template<size_t N>
void evaluation_rule<N>::optimize() {

    // Determine zero sequences
    std::vector<bool> marked_seqs(m_sequences.size(), false);
    for (size_t i = 0; i < m_sequences.size(); i++) {
        const sequence<N, size_t> &seq = m_sequences[i];
        size_t nidx = 0;
        for (register size_t j = 0; j < N; j++) nidx += seq[j];

        marked_seqs[i] = (nidx == 0);
    }

    // Determine allowed and forbidden terms
    std::vector<bool> marked_terms(m_term_list.size(), false);
    for (size_t i = 0; i < m_term_list.size(); i++) {
        const term &ct = m_term_list[i];
        marked_terms[i] = (marked_seqs[ct.seqno] ||
                ct.target == product_table_i::k_invalid ||
                ct.intr == product_table_i::k_invalid);
    }
    marked_seqs.clear();

    // Loop over all products
    typename std::vector<product_t>::iterator itp = m_setup.begin();
    while (itp != m_setup.end()) {

        product_t &pr = *itp;
        bool has_allowed = false;
        typename product_t::iterator itt = pr.begin();
        while (itt != pr.end()) {

            // Term is forbidden or allowed
            if (marked_terms[*itt]) {
                if (m_term_list[*itt].intr != product_table_i::k_invalid)
                    break;

                product_t::iterator itt2 = itt;
                itt++;
                pr.erase(itt2);
                has_allowed = true;
                continue;
            }

            itt++;
        }
        // If there was one forbidden term the product can be deleted
        if (itt != pr.end()) {
            itp = m_setup.erase(itp);
            continue;
        }

        // If there were only allowed terms in the product the whole rule
        // can be simplified
        if (has_allowed && pr.size() == 0) break;

        // Check if this is a duplicate product
        typename std::vector<product_t>::iterator itp2 = m_setup.begin();
        for (; itp2 != itp; itp2++) {
            if (itp2->size() != itp->size()) continue;

            typename product_t::iterator it1 = itp->begin(),
                    it2 = itp2->begin();
            for (; it1 != itp->end(); it1++, it2++) {
                if (*it1 != *it2) break;
            }
            if (it1 == itp->end()) break;
        }
        if (itp2 != itp) {
            itp = m_setup.erase(itp);
            continue;
        }

        itp++;
    }
    marked_terms.clear();

    // All blocks are allowed by this rule
    if (itp != m_setup.end()) {
        m_sequences.clear();
        m_term_list.clear();
        m_setup.clear();

        sequence<N, size_t> seq(1);
        term t(0, product_table_i::k_invalid, 0);
        m_sequences.push_back(seq);
        m_term_list.push_back(t);
        m_setup.push_back(product_t());
        m_setup.back().insert(0);
        return;
    }

    // Loop over all products and collect used terms
    std::set<size_t> used_terms;
    for (size_t i = 0; i < m_setup.size(); i++) {
        used_terms.insert(m_setup[i].begin(), m_setup[i].end());
    }

    // Loop over term list and delete unused terms (if there are any)
    if (used_terms.size() != m_term_list.size()) {
        std::vector<term> term_list(used_terms.size(), term(0, 0, 0));
        std::map<size_t, size_t> term_map;
        size_t ii = 0;
        for (std::set<size_t>::const_iterator it = used_terms.begin();
                it != used_terms.end(); it++, ii++) {
            term_list[ii] = m_term_list[*it];
            term_map[*it] = ii;
        }
        m_term_list.assign(term_list.begin(), term_list.end());

        // Loop again over product and modify the term ids accordingly
        for (size_t i = 0; i < m_setup.size(); i++) {
            product_t new_pr;
            for (typename product_t::iterator it = m_setup[i].begin();
                    it != m_setup[i].end(); it++) {
                new_pr.insert(term_map[*it]);
            }
            m_setup[i] = new_pr;
        }
    }

    // Loop over term list and collect used sequences
    std::set<size_t> used_seqs;
    for (size_t i = 0; i < m_term_list.size(); i++) {
        used_seqs.insert(m_term_list[i].seqno);
    }

    // Loop over sequence list and delete unused sequences (if there are any)
    if (used_seqs.size() != m_sequences.size()) {
        std::vector< sequence<N, size_t> > seq_list(used_seqs.size());
        std::map<size_t, size_t> seq_map;
        size_t ii = 0;
        for (std::set<size_t>::const_iterator it = used_seqs.begin();
                it != used_seqs.end(); it++, ii++) {
            seq_list[ii] = m_sequences[*it];
            seq_map[*it] = ii;
        }
        m_sequences.assign(seq_list.begin(), seq_list.end());

        // Loop again over term list and modify the seq nos accordingly
        for (size_t i = 0; i < m_term_list.size(); i++) {
            m_term_list[i].seqno = seq_map[m_term_list[i].seqno];
        }
    }
}


template<size_t N>
size_t evaluation_rule<N>::add_term(
        size_t seq_no, label_t intr, label_t target) {

    for (size_t i = 0; i < m_term_list.size(); i++) {
        const term &ct = m_term_list[i];
        if (seq_no == ct.seqno && intr == ct.intr && target == ct.target)
            return i;
    }

    m_term_list.push_back(term(seq_no, intr, target));
    return m_term_list.size() - 1;
}

template<size_t N>
bool evaluation_rule<N>::is_valid(iterator it) const {

    for (std::vector<product_t>::const_iterator it1 = m_setup.begin();
            it1 != m_setup.end(); it1++) {

        const product_t &pr = *it1;
        for (iterator it2 = pr.begin(); it2 != pr.end(); it2++) {

            if (it == it2) return true;
        }
    }

    return false;
}


} // namespace libtensor

#endif // LIBTENSOR_EVALUATION_RULE_IMPL_H
