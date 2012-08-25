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
evaluation_rule<N>::evaluation_rule(const evaluation_rule<N> &other) {

    m_slist = new eval_sequence_list<N>();
    for (const_iterator it = other.begin(); it != other.end(); it++) {
        const product_rule<N> &pr = other.get_product(it);

        m_rules.push_back(product_rule<N>(m_slist));
        product_rule<N> &prx = m_rules.back();
        for (typename product_rule<N>::iterator ip = pr.begin();
                ip != pr.end(); ip++) {
            prx.add(pr.get_sequence(ip), pr.get_intrinsic(ip));
        }
    }
}


template<size_t N>
const evaluation_rule<N> &
evaluation_rule<N>::operator=(const evaluation_rule<N> &other) {

    m_slist->clear();
    m_rules.clear();

    for (const_iterator it = other.begin(); it != other.end(); it++) {

        const product_rule<N> &pr = other.get_product(it);

        m_rules.push_back(product_rule<N>(m_slist));
        product_rule<N> &prx = m_rules.back();
        for (typename product_rule<N>::iterator ip = pr.begin();
                ip != pr.end(); ip++) {
            prx.add(pr.get_sequence(ip), pr.get_intrinsic(ip));
        }
    }

    return *this;
}


template<size_t N>
bool evaluation_rule<N>::is_allowed(const sequence<N, label_t> &blk_labels,
        const product_table_i &pt) const {

    bool allowed = false;

    evaluation_rule<N>::start_timer("is_allowed");

    // Loop over all sequences in rule and determine result labels
    std::vector<label_set_t> ls(m_slist->size());
    for (size_t i = 0; i < m_slist->size(); i++) {
        const sequence<N, size_t> &seq = (*m_slist)[i];
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
            pt.product(lg, ls[i]);
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
            allowed = true;
            break;
        }
    }

    evaluation_rule<N>::stop_timer("is_allowed");

    return allowed;
}


} // namespace libtensor


#endif // LIBTENSOR_EVALUATION_RULE_IMPL_H
