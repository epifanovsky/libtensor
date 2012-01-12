#include "../bad_symmetry.h"
#include "product_table_container.h"
#include "transfer_rule.h"

namespace libtensor {

const char *transfer_rule::k_clazz = "transfer_rule";

transfer_rule::transfer_rule(const evaluation_rule &rule, size_t ndims,
        const std::string &id) : m_ndims(ndims),
        m_pt(product_table_container::get_instance().req_const_table(id)) {

    analyze(rule);
    optimize_products();
}

transfer_rule::~transfer_rule() {

    product_table_container::get_instance().ret_table(m_pt.get_id());
}

void transfer_rule::perform(evaluation_rule &to) {

    static const char *method = "perform(evaluation_rule &)";

    transfer_rule::start_timer();

    to.clear_all();

    std::map<rule_id, rule_id> map;
    for (std::list<product_t>::const_iterator it = m_products.begin();
            it != m_products.end(); it++) {

        product_t::const_iterator pit = it->begin();
        rule_id rid;
        std::map<rule_id, rule_id>::iterator ir = map.find(pit->first);
        if (ir == map.end()) {
            const basic_rule &br = pit->second->second;
            rid = to.add_rule(br.intr, br.order);
            map[pit->first] = rid;
        }
        else {
            rid = ir->second;
        }
        size_t pid = to.add_product(rid);
        pit++;

        for (; pit != it->end(); it++) {

            ir = map.find(pit->first);
            if (ir == map.end()) {
                const basic_rule &br = pit->second->second;
                rid = to.add_rule(br.intr, br.order);
            }
            else {
                rid = ir->second;
            }
            to.add_to_product(pid, rid);
        }
    }

    transfer_rule::stop_timer();
}

void transfer_rule::analyze(const evaluation_rule &rule) {

    static const char *method = "analyze(const evaluation_rule &)";

    transfer_rule::start_timer(method);

    // First optimize all the basic rules
    rule_list optimized;
    std::map<rule_id, bool> found_intr;
    for (evaluation_rule::rule_iterator it = rule.begin();
            it != rule.end(); it++) {

        rule_id rid = rule.get_rule_id(it);
        found_intr[rid] = transfer_basic(rule.get_rule(it),
                optimized[rid]);
    }

    rule_id next_rid = optimized.rbegin()->first + 1;

    // Then look at the products and test what optimized rules are actually
    // required, which can be merged, and which are trivial
    for (size_t i = 0; i < rule.get_n_products(); i++) {

        bool forbidden = false;
        std::map<rule_id, bool> done;

        product_t cur_pr;
        for (evaluation_rule::product_iterator pit = rule.begin(i);
                pit != rule.end(i); pit++) {

            rule_id rid = rule.get_rule_id(pit);
            if (done[rid]) continue;
            done[rid] = true;

            const basic_rule &br = optimized[rid];

            // Identify trivial rules
            if (found_intr[rid]) {
                if (br.intr.size() == 0) { forbidden = true; break; }
                if (br.intr.size() == m_pt.nlabels()) { continue; }
                if (br.order.size() == 1) {
                    if (br.intr[0] != 0) { forbidden = true; break; }
                    else { continue; }
                }
            }
            else {
                if (br.order.size() == 0) { forbidden = true; break; }
            }

            // Since the current basic rule is non-trivial, look at all
            // subsequent basic rules and check if we can merge
            std::list<rule_id> merge;
            merge.push_back(rid);
            evaluation_rule::product_iterator pit2 = pit;
            pit2++;
            for (; pit2 != rule.end(i); pit2++) {
                rule_id rid2 = rule.get_rule_id(pit2);

                if (done[rid2]) continue;
                if (equal_order(br.order, optimized[rid2].order)) {
                    merge.push_back(rid2);
                    done[rid2] = true;
                }
            }

            // Multiple rules can be merged
            if ((merge.size() != 0) && found_intr[rid]) {

                std::vector<bool> valid(m_pt.nlabels(), true);
                for (std::list<rule_id>::iterator it = merge.begin();
                        it != merge.end(); it++) {

                    const label_group &intrx = optimized[*it].intr;

                    size_t i = 0, j = 0;
                    for (; j < m_pt.nlabels() && i < intrx.size(); j++) {
                        if (j == intrx[i]) { i++; }
                        else { valid[j] = false; }
                    }
                    for (; j < m_pt.nlabels(); j++) { valid[j] = false; }
                }

                label_group intr;
                for (size_t i = 0; i < m_pt.nlabels(); i++) {
                    if (valid[i]) intr.push_back(i);
                }

                if (intr.size() == 0) { forbidden = true; break; }
                if (intr.size() == br.intr.size()) {
                    rule_list::iterator it = m_req_rules.find(rid);
                    if (it == m_req_rules.end()) {
                        m_req_rules[rid] = br;
                        it = m_req_rules.find(rid);
                    }
                    cur_pr[rid] = it;
                }
                else {
                    m_req_rules[next_rid] = basic_rule(intr, br.order);
                    rule_list::iterator it = m_req_rules.find(next_rid);
                    cur_pr[next_rid] = it;
                    next_rid++;
                }
            }
            else {
                rule_list::iterator it = m_req_rules.find(rid);
                if (it == m_req_rules.end()) {
                    m_req_rules[rid] = br;
                    it = m_req_rules.find(rid);
                }

                cur_pr[rid] = it;
            }
        }

        // Forbidden products can savely be ignored
        if (forbidden) continue;

        // Empty product and not forbidden, the product allows all blocks
        // i.e. everything can be simplified to one rule
        if (cur_pr.size() == 0) {
            m_req_rules.clear();
            m_products.clear();

            m_req_rules[next_rid] = basic_rule(label_group(1, 0),
                    std::vector<size_t>(1, evaluation_rule::k_intrinsic));
            cur_pr[next_rid] = m_req_rules.find(next_rid);
            m_products.push_back(cur_pr);

            return;
        }

        m_products.push_back(cur_pr);
    }

    // Now, loop over the required rules and check, if any duplicate rules have
    // been newly introduced
    std::map<rule_id, rule_id> idmap;
    for (rule_list::iterator ir1 = m_req_rules.begin();
            ir1 != m_req_rules.end(); ir1++) {

         // Skip if the rule is already marked as identical to another one
        if (idmap.find(ir1->first) != idmap.end()) continue;

        basic_rule &br1 = ir1->second;

        // Loop over all later rules
        rule_list::iterator ir2 = ir1; ir2++;
        for (; ir2 != m_req_rules.end(); ir2++) {

            // First compare the evaluation order
            const basic_rule &br2 = ir2->second;
            if (equal_order(br2.order, br1.order) &&
                    equal_intr(br2.intr, br1.intr)) {
                idmap[ir2->first] = ir1->first;
            }
        }
    }

    // At last, loop over all products and replace the identical copies
    for (std::list<product_t>::iterator it = m_products.begin();
            it != m_products.end(); it++) {

        product_t &cur_pr = *it;
        std::list<rule_id> remove, add;
        for (product_t::iterator ipr = cur_pr.begin();
                ipr != cur_pr.end(); ipr++) {

            std::map<rule_id, rule_id>::iterator ir = idmap.find(ipr->first);
            if (ir == idmap.end()) continue;

            remove.push_back(ir->first);
            add.push_back(ir->second);
        }

        for (std::list<rule_id>::iterator ix = remove.begin();
                ix != remove.end(); ix++) {

            cur_pr.erase(*ix);
        }

        for (std::list<rule_id>::iterator ix = add.begin();
                ix != add.end(); ix++) {

            product_t::iterator ipr = cur_pr.find(*ix);
            if (ipr != cur_pr.end()) continue;

            cur_pr[*ix] = m_req_rules.find(*ix);
        }
    }

    transfer_rule::stop_timer(method);
}

void transfer_rule::optimize_products() {

    static const char *method = "optimize_products()";

    transfer_rule::start_timer(method);

    typedef std::list<product_t> product_list;

    // Find identical products and remove them
    for (product_list::iterator it1 = m_products.begin();
            it1 != m_products.end(); it1++) {

        product_list::iterator it2 = it1;
        it2++;
        while (it2 != m_products.end()) {

            if (equal_product(*it1, *it2)) { it2 = m_products.erase(it2); }
            else { it2++; }
        }
    }

    // Then look for "products" with only one element and identical evaluation
    // order, and merge them
    for (product_list::iterator it1 = m_products.begin();
            it1 != m_products.end(); it1++) {

        if (it1->size() != 1) continue;

        std::list<rule_list::iterator> merge;
        rule_list::iterator ir1 = it1->begin()->second;
        merge.push_back(ir1);

        product_list::iterator it2 = it1;
        it2++;
        while (it2 != m_products.end()) {

            if (it2->size() == 1) {
                rule_list::iterator ir2 = it2->begin()->second;
                if (equal_order(ir1->second.order, ir2->second.order)) {
                    it2 = m_products.erase(it2);
                    merge.push_back(ir2);
                    continue;
                }
            }
            it2++;
        }

        if (merge.size() == 1) continue;

        // OK, there is something to merge, so delete the existing product...
        it1->clear();

        size_t max_size = 0;
        rule_list::iterator ir;
        std::map<label_t, bool> lmap;
        for (std::list<rule_list::iterator>::iterator irm = merge.begin();
                irm != merge.end(); irm++) {

            const basic_rule &br = (*irm)->second;
            if (br.intr.size() > max_size) {
                max_size = br.intr.size(); ir = *irm;
            }

            for (size_t k = 0; k < br.intr.size(); k++) {
                lmap[br.intr[k]] = true;
            }
        }

        // But the result of the merge is identical to one of the element
        if (lmap.size() == max_size) {
            (*it1)[ir->first] = ir; continue;
        }

        // Otherwise create a new basic rule
        label_group intr;
        for (std::map<label_t, bool>::iterator il = lmap.begin();
                il != lmap.end(); il++) {
            intr.push_back(il->first);
        }

        rule_id next_rid = m_req_rules.rbegin()->first + 1;
        m_req_rules[next_rid] = basic_rule(intr, ir->second.order);
        rule_list::iterator irx = m_req_rules.find(next_rid);
        (*it1)[next_rid] = irx;
    }

    transfer_rule::stop_timer(method);
}

bool transfer_rule::transfer_basic(
        const basic_rule &from, basic_rule &to) {

    static const char *method =
            "transfer_basic(const basic_rule &, basic_rule &) const";

    to.order.clear();
    to.intr.clear();

    // First try to optimize the order
    bool found_intr = false;
    for (size_t i = 0; i < from.order.size(); i++) {

        if (from.order[i] == evaluation_rule::k_intrinsic) {
            to.order.push_back(evaluation_rule::k_intrinsic);
            found_intr = true;
            continue;
        }

        if (from.order[i] >= m_ndims) {
            throw bad_symmetry(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Invalid order.");
        }

        size_t j = i + 1;
        for (; j < from.order.size(); j++) {
            if (from.order[j] != from.order[i]) break;
        }

        size_t neq = j - i;
        // Nothing to optimize here
        if (neq == 1) {
            to.order.push_back(from.order[i]); continue;
        }

        // Check if we can reduce identical dims to zero or one
        std::map<label_t, label_t> map;
        for (label_t l = 0; l < m_pt.nlabels(); l++) {

            label_group lg(neq, l);
            label_t lt = 0;
            while (! m_pt.is_in_product(lg, lt)) lt++;
            map[l] = lt;
            for (; lt < m_pt.nlabels(); lt++) {
                if (m_pt.is_in_product(lg, lt)) break;
            }

            if (lt != m_pt.nlabels()) { map.erase(l); break; }
        }

        // May be we can reduce the dims
        if (map.size() == m_pt.nlabels()) {
            bool is_zero = true, is_ident = true;
            for (label_t l = 0; l < m_pt.nlabels(); l++) {
                is_zero = is_zero && (map[l] == 0);
                is_ident = is_ident && (map[l] == l);
            }
            if (is_ident || is_zero) {
                if (is_ident) to.order.push_back(from.order[i]);
                i = j - 1;
                continue;
            }
        }

        // Ok we cannot reduce them => just copy
        for (size_t k = i; k < j; k++) to.order.push_back(from.order[k]);
        i = j - 1;
    }

    if (found_intr) {
        // Get a list of unique labels
        std::map<label_t, bool> lmap;
        for (size_t i = 0; i < from.intr.size(); i++) {
            if (! m_pt.is_valid(from.intr[i])) {
                throw bad_symmetry(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "Invalid label.");
            }

            lmap[from.intr[i]] = true;
        }

        for (std::map<label_t, bool>::iterator it = lmap.begin();
                it != lmap.end(); it++) {
            to.intr.push_back(it->first);
        }
    }

    return found_intr;
}

bool transfer_rule::equal_order(const std::vector<size_t> &o1,
        const std::vector<size_t> &o2) {

    if (o1.size() != o2.size()) return false;

    size_t i = 0;
    for (; i < o1.size(); i++) {
        if (o1[i] != o2[i]) break;
    }
    if (i != o1.size()) return false;

    return true;
}

bool transfer_rule::equal_intr(const label_group &i1, const label_group &i2) {

    if (i1.size() != i2.size()) return false;

    size_t i = 0;
    for (; i < i1.size(); i++) {
        size_t j = 0;
        for (; j < i2.size(); j++) {
            if (i1[i] == i2[j]) break; }
        if (j == i2.size()) break;
    }
    if (i != i1.size()) return false;

    return true;
}

bool transfer_rule::equal_product(const product_t &pr1, const product_t &pr2) {

    if (pr1.size() != pr2.size()) return false;

    product_t::const_iterator it1 = pr1.begin(), it2 = pr2.begin();
    for (; it1 != pr1.end(); it1++, it2++) {
        if (it1->first != it2->first) break;
    }
    return it1 == pr1.end();
}


} // namespace libtensor
