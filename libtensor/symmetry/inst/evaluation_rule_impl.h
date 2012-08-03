#ifndef LIBTENSOR_EVALUATION_RULE_IMPL_H
#define LIBTENSOR_EVALUATION_RULE_IMPL_H

#include <list>
#include <libtensor/core/permutation_generator.h>
#include <libtensor/core/sequence_generator.h>
#include "../bad_symmetry.h"


namespace libtensor {


template<size_t N>
const char *evaluation_rule<N>::k_clazz = "evaluation_rule<N>";


template<size_t N>
void evaluation_rule<N>::add_to_product(iterator it,
        const sequence_t &seq, label_t intr) {

    // Ignore all allowed rules in non-empty products
    if (intr == product_table_i::k_invalid && ! it->empty()) return;

    // Add sequence if not available yet
    size_t seqno = has_sequence(seq);
    if (seqno == m_sequences.size()) m_sequences.push_back(seq);

    // Check if sequence already exists in product
    product_t::iterator ip = it->find(seqno);
    if (ip == it->end())
        it->insert(product_t::value_type(seqno, intr));
    else
        throw bad_symmetry(g_ns, k_clazz,
                "add_to_product(iterator, const sequence_t &, label_t)",
                __FILE__, __LINE__, "Adding an already existing sequence.");
}


template<size_t N>
void evaluation_rule<N>::optimize() {

    // Determine zero sequences
    std::vector<bool> zero_seqs(m_sequences.size(), false);
    for (size_t i = 0; i < m_sequences.size(); i++) {
        const sequence_t &seq = m_sequences[i];
        size_t nidx = 0;
        for (register size_t j = 0; j < N; j++) nidx += seq[j];
        zero_seqs[i] = (nidx == 0);
    }

    // Loop over all products
    iterator it1 = m_setup.begin();
    while (it1 != m_setup.end()) {
        // Delete empty products
        if (it1->empty()) {
            it1 = m_setup.erase(it1);
            continue;
        }

        bool has_allowed = false;
        typename product_t::iterator ip1 = it1->begin();
        while (ip1 != it1->end()) {
            if (ip1->second == product_table_i::k_invalid) {
                product_t::iterator ip2 = ip1++;
                it1->erase(ip2);
                has_allowed = true;
            }

            if (zero_seqs[ip1->first]) {
                if (ip1->second == product_table_i::k_identity) {
                    product_t::iterator ip2 = ip1++;
                    it1->erase(ip2);
                    has_allowed = true;
                }
                else {
                    break;
                }
            }
        }

        // If there was one forbidden term the product can be deleted
        if (ip1 != it1->end()) {
            it1 = m_setup.erase(it1);
            continue;
        }

        // If there were only allowed terms in the product the whole rule
        // can be simplified
        if (has_allowed && it1->size() == 0) break;

        // Check if this is a duplicate product
        iterator it2 = m_setup.begin();
        for (; it2 != it1; it2++) {
            if (it2->size() != it1->size()) continue;

            typename product_t::iterator ip1 = it1->begin();
            typename product_t::iterator ip2 = it2->begin();

            for (; ip1 != ip1->end(); ip1++, ip2++) {
                if (ip1->first != ip2->first ||
                        ip1->second != ip2->second) break;
            }
            if (ip1 == it1->end()) break;
        }
        if (it2 != it1) {
            it1 = m_setup.erase(it1);
            continue;
        }

        it1++;
    }
    zero_seqs.clear();

    // All blocks are allowed by this rule
    if (it1 != m_setup.end()) {
        m_sequences.clear();
        m_setup.clear();

        sequence_t seq(1);
        m_sequences.push_back(seq);
        iterator itn = new_product();
        itn->insert(product_t::value_type(0, product_table_i::k_invalid));
        return;
    }
}


template<size_t N>
size_t evaluation_rule<N>::has_sequence(const sequence_t &seq) const {

    size_t seqno = 0;
    for (; seqno < m_sequences.size(); seqno++) {
        const sequence<N, size_t> &ref = m_sequences[seqno];

        register size_t i = 0;
        for (; i < N; i++) {
            if (seq[i] != ref[i]) break;
        }
        if (i == N) return seqno;
    }

    return m_sequences.size();
}


} // namespace libtensor


#endif // LIBTENSOR_EVALUATION_RULE_IMPL_H
