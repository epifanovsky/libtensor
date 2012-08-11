#ifndef LIBTENSOR_ER_OPTIMIZE_IMPL_H
#define LIBTENSOR_ER_OPTIMIZE_IMPL_H

#include "../bad_symmetry.h"

namespace libtensor {


template<size_t N>
const char *er_optimize<N>::k_clazz = "er_optimize<N>";


template<size_t N>
er_optimize<N>::er_optimize(const evaluation_rule<N> &from) : m_rule(from) {
}


template<size_t N>
void er_optimize<N>::perform(evaluation_rule<N> &to) const {

    er_optimize<N>::start_timer();

    to.clear();

    std::list< product_rule<N> > plst;
    eval_sequence_list<N> slst;

    // Loop over all products
    typename evaluation_rule<N>::const_iterator it = m_rule.begin();
    for (; it != m_rule.end(); it++) {

        const product_rule<N> &pr = m_rule.get_product(it);

        // Delete empty products
        if (pr.empty()) continue;

        product_rule<N> prx(&slst);

        // Look for products with 'all allowed' rules
        size_t nallowed = 0;
        typename product_rule<N>::iterator ip = pr.begin();
        for (; ip != pr.end(); ip++) {

            if (pr.get_intrinsic(ip) == product_table_i::k_invalid) {
                nallowed++;
                continue;
            }

            const sequence<N, size_t> &seq = pr.get_sequence(ip);
            size_t nidx = 0;
            for (register size_t j = 0; j < N; j++) nidx += seq[j];
            if (nidx == 0) {
                if (pr.get_intrinsic(ip) == product_table_i::k_identity) {
                    nallowed++; continue;
                }
                else break;
            }

            prx.add(pr.get_sequence(ip), pr.get_intrinsic(ip));
        }

        // If there was one forbidden term the product can be deleted
        if (ip != pr.end()) continue;

        if (prx.empty()) {
            if (nallowed != 0) break;
            else continue;
        }

        plst.push_back(prx);
    }

    // All blocks are allowed by this rule
    if (it != m_rule.end()) {

        sequence<N, size_t> seq(1);
        product_rule<N> &pr = to.new_product();
        pr.add(seq, product_table_i::k_invalid);

        er_optimize<N>::stop_timer();
        return;
    }

    // Remove duplicate products
    for (typename std::list< product_rule<N> >::iterator it1 = plst.begin();
            it1 != plst.end(); it1++) {

        typename std::list< product_rule<N> >::iterator it2 = it1;
        it2++;
        while (it2 != plst.end()) {
            if (*it1 == *it2) it2 = plst.erase(it2);
            else it2++;
        }
    }

    // Copy from new lists to member lists
    for (typename std::list< product_rule<N> >::iterator it1 = plst.begin();
            it1 != plst.end(); it1++) {

        product_rule<N> &pr = to.new_product();
        for (typename product_rule<N>::iterator ip1 = it1->begin();
                ip1 != it1->end(); ip1++) {
            pr.add(it1->get_sequence(ip1), it1->get_intrinsic(ip1));
        }
    }

    er_optimize<N>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_ER_OPTIMIZE_IMPL_H
