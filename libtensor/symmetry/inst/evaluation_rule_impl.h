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
    for (iterator it = other.begin(); it != other.end(); it++) {
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

    for (iterator it = other.begin(); it != other.end(); it++) {

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

    // Loop over all products in the evaluation rule
    for (iterator it = m_rules.begin(); it != m_rules.end(); it++) {

    	const product_rule<N> &pr = *it;

        // Loop over all terms in the current product
        typename product_rule<N>::iterator ip = pr.begin();
        for (; ip != pr.end(); ip++) {

            if (pr.get_intrinsic(ip) == product_table_i::k_invalid) continue;

        	// Construct product
            const sequence<N, size_t> &seq = pr.get_sequence(ip);

            label_group_t lg;
        	register size_t i = 0;
        	for (; i < N; i++) {
        		if (seq[i] == 0) continue;
        		if (blk_labels[i] == product_table_i::k_invalid) break;
        		lg.insert(lg.end(), seq[i], blk_labels[i]);
        	}
        	if (i != N) continue;

            label_set_t ls;
            pt.product(lg, ls);

            if (ls.count(pr.get_intrinsic(ip)) == 0) break;

        }

        if (ip == it->end()) { return true; }
    }

    return false;
}


} // namespace libtensor


#endif // LIBTENSOR_EVALUATION_RULE_IMPL_H
