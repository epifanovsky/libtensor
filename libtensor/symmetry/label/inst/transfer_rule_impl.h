#ifndef LIBTENSOR_TRANSFER_RULE_IMPL_H
#define LIBTENSOR_TRANSFER_RULE_IMPL_H

#include "../product_table_container.h"

namespace libtensor {

template<size_t N>
const char *transfer_rule<N>::k_clazz = "transfer_rule<N>";

template<size_t N>
transfer_rule<N>::transfer_rule(
        const evaluation_rule<N> &from, const std::string &id) :
        m_from(from), m_mergeable(true),
        m_pt(product_table_container::get_instance().req_const_table(id)) {

    label_set_t complete_set(m_pt.get_complete_set());
    label_set_t::const_iterator it = complete_set.begin();

    m_merge_set = m_pt.product(*it, *it);
    it++;
    for (; it != complete_set.end(); it++) {
        label_set_t setx = m_pt.product(*it, *it);

        if (setx.size() != m_merge_set.size()) {
            m_mergeable = false;
            break;
        }

        label_set_t::const_iterator ita = m_merge_set.begin(),
                itb = setx.begin();
        for (; ita != m_merge_set.end(); ita++, itb++)
            if (*ita != *itb) break;

        if (ita != m_merge_set.end()) {
            m_mergeable = false;
            break;
        }
    }
    if (! m_mergeable) m_merge_set.empty();
}

template<size_t N>
transfer_rule<N>::~transfer_rule() {

    product_table_container::get_instance().ret_table(m_pt.get_id());
}

template<size_t N>
void transfer_rule<N>::perform(evaluation_rule<N> &to) {

    static const char *method = "perform(evaluation_rule<N> &)";

    transfer_rule<N>::start_timer();

    to.clear_all();

    // Optimize rules in from
    rule_list_t optimized;
    std::map<rule_id_t, bool> trivial;
    optimize_basic(optimized, trivial);

    // Find duplicates and similar rules in the optimized rules
    std::map<rule_id_t, rule_id_t> dupl, sim;
    find_similar(optimized, sim, dupl);

    // Create a list of products
    std::vector< std::set<rule_id_t> > products;
    for (size_t i = 0; i < m_from.get_n_products(); i++) {

        std::set<rule_id_t> cur_product;

        bool found_ta = false, found_tf = false;
        for (typename evaluation_rule<N>::product_iterator it
                = m_from.begin(i); it != m_from.end(i); it++) {

            rule_id_t rid = m_from.get_rule_id(it);

            // Check if the current rule is trivial
            typename std::map<rule_id_t, bool>::const_iterator irt =
                    trivial.find(rid);
            if (irt != trivial.end()) {
                // Mark if trivially allowed
                if (irt->second) {
                    found_ta = true; continue;
                }
                // If trivially forbidden, product is unnecessary
                else {
                    found_tf = true; break;
                }
            }

            // If the rule is duplicate, add the original to product
            typename std::map<rule_id_t, rule_id_t>::const_iterator ird =
                    dupl.find(rid);

            if (ird != dupl.end())
                cur_product.insert(ird->second);
            else
                cur_product.insert(rid);
        }

        // If there was a trivially forbidden rule ignore this product
        if (found_tf) continue;

        // If the only rule in product is trivially allowed, the whole
        // evaluation rule is trivially allowed and we are done.
        if (cur_product.empty()) {
            if (found_ta) {
                basic_rule<N> br(m_pt.get_complete_set());
                for (size_t i = 0; i < N; i++) br[i] = 1;
                rule_id_t rid = to.add_rule(br);
                to.add_product(rid);

                return;
            }

            continue;
        }

        // At last check if there are similar rules in the product
        for (typename std::set<rule_id_t>::iterator it1 = cur_product.begin();
                it1 != cur_product.end(); it1++) {

            rule_id_t rid_cur = *it1;

            std::set<rule_id_t> sim2cur;
            typename std::set<rule_id_t>::iterator it2 = it1; it2++;
            while (it2 != cur_product.end()) {

                typename std::map<rule_id_t, rule_id_t>::const_iterator itx =
                        sim.find(*it2);
                if (itx != sim.end() && itx->second == rid_cur)
                    sim2cur.insert(itx->first);

                it2++;
            }

            if (sim2cur.empty()) continue;

            const basic_rule<N> &br_cur = optimized[rid_cur];
            label_set_t lsn(br_cur.get_target());

            bool is_orig = true;
            for (typename std::set<rule_id_t>::const_iterator ir =
                    sim2cur.begin(); ir != sim2cur.end(); ir++) {

                const label_set_t &lsx = optimized[*ir].get_target();
                for (label_set_t::iterator itn = lsn.begin();
                        itn != lsn.end(); itn++) {

                    label_t ll = *itn;
                    if (lsx.count(ll) == 0) {
                        lsn.erase(ll);
                        itn = lsn.lower_bound(ll);
                        is_orig = false;
                    }
                }

                cur_product.erase(*ir);
            }

            if (! is_orig) {
                cur_product.erase(rid_cur);
                it1 = cur_product.lower_bound(rid_cur);

                basic_rule<N> brn(lsn);
                for (size_t i = 0; i < N; i++) brn[i] = br_cur[i];

                rule_id_t rid = optimized.rbegin()->first + 1;
                optimized.insert(typename rule_list_t::value_type(rid, brn));
                cur_product.insert(rid);
            }
        }

        products.push_back(cur_product);
    }

    // Now look for duplicate products
    for (typename std::vector< std::set<rule_id_t> >::iterator it1 =
            products.begin(); it1 != products.end(); it1++) {

        typename std::vector< std::set<rule_id_t> >::iterator it2 = it1;
        it2++;
        while (it2 != products.end()) {

            if (it2->size() != it1->size()) { it2++; continue; }

            typename std::set<rule_id_t>::const_iterator ir1 = it1->begin();
            typename std::set<rule_id_t>::const_iterator ir2 = it2->begin();
            for (; ir1 != it1->end(); ir1++, ir2++) {
                if (*ir1 != *ir2) break;
            }
            if (ir1 == it1->end())
                it2 = products.erase(it2);
            else
                it2++;
        }
    }

    // Check all "products" consisting of one basic_rule
    for (typename std::vector< std::set<rule_id_t> >::iterator it1 =
            products.begin(); it1 != products.end(); it1++) {

        if (it1->size() != 1) continue;

        rule_id_t rid_cur = *(it1->begin());

        std::set<rule_id_t> sim2cur;
        typename std::vector< std::set<rule_id_t> >::iterator it2 = it1;
        it2++;
        while (it2 != products.end()) {

            if (it2->size() != 1) { it2++; continue; }

            typename std::map<rule_id_t, rule_id_t>::const_iterator itx =
                    sim.find(*(it2->begin()));
            if (itx == sim.end() || itx->second != rid_cur) {
                it2++; continue;
            }

            sim2cur.insert(itx->first);
            it2 = products.erase(it2);
        }

        if (sim2cur.empty()) continue;

        basic_rule<N> brx = optimized[rid_cur];
        for (typename std::set<rule_id_t>::iterator its = sim2cur.begin();
                its != sim2cur.end(); its++) {

            const basic_rule<N> &br2 = optimized[*its];
            const label_set_t &t2 = br2.get_target();
            for (label_set_t::const_iterator ils = t2.begin();
                    ils != t2.end(); ils++)
                brx.set_target(*ils);
        }

        // If the resulting rule is trivial, we are done again
        if (brx.get_target().size() == m_pt.get_complete_set().size()) {

            basic_rule<N> br(m_pt.get_complete_set());
            for (size_t i = 0; i < N; i++) br[i] = 1;
            rule_id_t rid = to.add_rule(br);
            to.add_product(rid);

            return;
        }

        rule_id_t next_rid = optimized.rend()->first;
        next_rid++;
        optimized.insert(typename rule_list_t::value_type(next_rid, brx));
        it1->clear();
        it1->insert(next_rid);
    }

    // Now transfer the products to new evaluation rule
    std::map<rule_id_t, rule_id_t> map;
    for (typename std::vector< std::set<rule_id_t> >::const_iterator it =
            products.begin(); it != products.end(); it++) {

        typename std::set<rule_id_t>::const_iterator pit = it->begin();

        rule_id_t rid;
        typename std::map<rule_id_t, rule_id_t>::const_iterator ir =
                map.find(*pit);
        if (ir == map.end()) {
            rid = to.add_rule(optimized[*pit]);
            map[*pit] = rid;
        }
        else {
            rid = ir->second;
        }

        size_t pid = to.add_product(rid);

        pit++;
        for (; pit != it->end(); pit++) {

            ir = map.find(*pit);
            if (ir == map.end()) {
                rid = to.add_rule(optimized[*pit]);
                map[*pit] = rid;
            }
            else {
                rid = ir->second;
            }
            to.add_to_product(pid, rid);
        }
    }

    transfer_rule<N>::stop_timer();
}

template<size_t N>
void transfer_rule<N>::optimize_basic(
        rule_list_t &opt, std::map<rule_id_t, bool> &triv) const {

    label_set_t complete_set = m_pt.get_complete_set();

    opt.clear();
    triv.clear();

    // Loop over all basic rules in m_from
    for (typename evaluation_rule<N>::rule_iterator it = m_from.begin();
            it != m_from.end(); it++) {

        rule_id_t rid = m_from.get_rule_id(it);
        const basic_rule<N> &br = m_from.get_rule(it);

        // If target is empty, rule is trivially forbidden
        if (br.get_target().size() == 0) {
            triv[rid] = false; continue;
        }
        // If target is the complete set, rule is trivially allowed
        if (br.get_target().size() == complete_set.size()) {
            triv[rid] = true; continue;
        }

        bool is_zero = true;
        basic_rule<N> brx(br);
        // If labels can be merged
        if (m_mergeable) {
            for (size_t i = 0; i < N; i++) {
                if (brx[i] != 0) is_zero = false;
                if (brx[i] < 2) { continue; }

                if (brx[i] % 2 == 0)
                    brx[i] = 0;
                else
                    brx[i] = 1;

                label_set_t nt = m_pt.product(brx.get_target(), m_merge_set);
                brx.reset_target();
                for (label_set_t::const_iterator ils = nt.begin();
                        ils != nt.end(); ils++)
                    brx.set_target(*ils);
            }
        }
        else {
            for (size_t i = 0; i < N; i++)
                if (brx[i] != 0) { is_zero = false; break; }
        }

        // If no block labels are evaluated, the rule is trivially forbidden.
        if (is_zero) {
            triv[rid] = false; continue;
        }

        opt.insert(typename rule_list_t::value_type(rid, brx));
    }
}

template<size_t N>
void transfer_rule<N>::find_similar(const rule_list_t &rules,
        rule_map_t &sim, rule_map_t &dupl) {

    for (typename rule_list_t::const_iterator it1 = rules.begin();
            it1 != rules.end(); it1++) {

        const basic_rule<N> &br1 = it1->second;

        typename rule_list_t::const_iterator it2 = it1;
        it2++;
        for (; it2 != rules.end(); it2++) {
            const basic_rule<N> &br2 = it2->second;

            if (br1 == br2) {
                dupl[it2->first] = it1->first;
                continue;
            }

            size_t i = 0;
            for (; i < N; i++)
                if (br1[i] != br2[i]) break;
            if (i == N) sim[it2->first] = it1->first;
        }
    }
}



} // namespace libtensor

#endif // LIBTENSOR_TRANSFER_RULE_IMPL_H
