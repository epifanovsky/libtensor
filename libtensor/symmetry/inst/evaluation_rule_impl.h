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

    m_slist.clear();
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


} // namespace libtensor


#endif // LIBTENSOR_EVALUATION_RULE_IMPL_H
