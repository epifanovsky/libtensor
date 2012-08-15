#ifndef LIBTENSOR_ER_OPTIMIZE_IMPL_H
#define LIBTENSOR_ER_OPTIMIZE_IMPL_H

#include "../bad_symmetry.h"
#include "../product_table_container.h"

namespace libtensor {


template<size_t N>
const char *er_optimize<N>::k_clazz = "er_optimize<N>";


template<size_t N>
er_optimize<N>::er_optimize(const evaluation_rule<N> &from,
        const std::string &id) :
        m_rule(from), m_mergable(true) {

    product_table_container &ptc = product_table_container::get_instance();
    const product_table_i &pt = ptc.req_const_table(id);

    for (product_table_i::label_t l = 0; l < pt.get_n_labels(); l++) {
        product_table_i::label_group_t lg(2, l);
        product_table_i::label_set_t ls;
        pt.product(lg, ls);
        if (ls.size() != 1 || ls.count(product_table_i::k_identity) != 1) {
            m_mergable = false; break;
        }
    }

    ptc.ret_table(id);
}


template<size_t N>
void er_optimize<N>::perform(evaluation_rule<N> &to) const {

    typedef std::multimap<size_t, product_table_i::label_t> product_map_t;

    er_optimize<N>::start_timer();

    to.clear();

    // Find zero sequences
    const eval_sequence_list<N> &slist = m_rule.get_sequences();
    std::vector< sequence<N, size_t> > new_seq;

    std::vector<const sequence<N, size_t> *> seq_ptr(slist.size(), 0);

    for (size_t i = 0; i < slist.size(); i++) {

        const sequence<N, size_t> &seq = slist[i];

        if (m_mergable) {
            size_t nidx = 0, nidx0 = 0;
            for (register size_t j = 0; j < N; j++) {
                nidx0 += seq[j]; nidx += (seq[j] % 2);
            }
            if (nidx != 0) {
                if (nidx0 == nidx) seq_ptr[i] = &seq;
                else {
                    new_seq.push_back(sequence<N, size_t>());
                    sequence<N, size_t> &seq2 = new_seq.back();
                    for (register size_t j = 0; j < N; j++)
                        seq2[j] = seq[j] % 2;
                    seq_ptr[i] = &seq2;
                }
            }
        }
        else {
            size_t nidx = 0;
            for (register size_t j = 0; j < N; j++) nidx += seq[j];
            if (nidx != 0) seq_ptr[i] = &seq;
        }
    }

    // Loop over all products
    std::list<product_map_t> plst;

    typename evaluation_rule<N>::const_iterator it = m_rule.begin();
    for (; it != m_rule.end(); it++) {

        const product_rule<N> &pr = m_rule.get_product(it);

        // Delete empty products
        if (pr.empty()) continue;

        plst.push_back(product_map_t());
        product_map_t &pmap = plst.back();

        // Look for products with 'all allowed' rules
        size_t nallowed = 0;
        typename product_rule<N>::iterator ip = pr.begin();
        for (; ip != pr.end(); ip++) {

            if (pr.get_intrinsic(ip) == product_table_i::k_invalid) {
                nallowed++;
                continue;
            }

            if (seq_ptr[pr.get_seqno(ip)] == 0) {
                if (pr.get_intrinsic(ip) == product_table_i::k_identity) {
                    nallowed++;
                    continue;
                }
                else {
                    break;
                }
            }

            pmap.insert(product_map_t::value_type(pr.get_seqno(ip),
                    pr.get_intrinsic(ip)));
        }

        // If there was one forbidden term the product can be deleted
        if (ip != pr.end()) {
            plst.pop_back();
            continue;
        }

        if (pmap.empty()) {
            plst.pop_back();

            if (nallowed != 0) break;
            else continue;
        }
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
    for (typename std::list<product_map_t>::iterator it1 = plst.begin();
            it1 != plst.end(); it1++) {

        typename std::list<product_map_t>::iterator it2 = it1;
        it2++;
        while (it2 != plst.end()) {
            if (it1->size() == it2->size()) {
                product_map_t::iterator ip1 = it1->begin(), ip2 = it2->begin();
                for (; ip1 != it1->end(); ip1++, ip2++) {
                    if (ip1->first != ip2->first || ip1->second != ip2->second)
                        break;
                }
                if (ip1 == it1->end()) {
                    it2 = plst.erase(it2);
                    continue;
                }
            }
            it2++;
        }
    }

    // Copy from new lists to member lists
    for (typename std::list<product_map_t>::iterator it1 = plst.begin();
            it1 != plst.end(); it1++) {

        product_rule<N> &pr = to.new_product();
        for (typename product_map_t::iterator ip1 = it1->begin();
                ip1 != it1->end(); ip1++) {
            pr.add(*(seq_ptr[ip1->first]), ip1->second);
        }
    }

    er_optimize<N>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_ER_OPTIMIZE_IMPL_H
