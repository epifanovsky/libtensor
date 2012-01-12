#include "../bad_symmetry.h"
#include "product_table_container.h"
#include "transfer_rule.h"

namespace libtensor {

const char *transfer_rule::k_clazz = "transfer_rule";

transfer_rule::transfer_rule(const evaluation_rule &rule, size_t ndims,
        const std::string &id) : m_ndims(ndims),
        m_pt(product_table_container::get_instance().req_const_table(id)) {

    analyze(rule);
}

transfer_rule::~transfer_rule() {

    product_table_container::get_instance().ret_table(m_pt.get_id());
}

void transfer_rule::perform(evaluation_rule &to) {

    static const char *method = "perform(evaluation_rule &)";

}

void transfer_rule::analyze(const evaluation_rule &rule) {

    static const char *method = "analyze(const evaluation_rule &)";

    typedef evaluation_rule::rule_id rule_id;

    // First check which basic rules are actually in the product and
    // non-trivial
    std::map<rule_id, bool> trivial;
    for (size_t i = 0; i < rule.get_n_products(); i++) {

        for (evaluation_rule::product_iterator pit = rule.begin(i);
                pit != rule.end(i); pit++) {

            rule_id rid = rule.get_rule_id(pit);
            if (m_req_rules.find(rid) != m_req_rules.end())
                continue;

            const basic_rule &br = rule.get_rule(pit);
            if (! check(br.order, m_ndims)) {
                std::vector<size_t> order;
                transfer_order(br.order, order, m_pt);
                if (order.size() == 0) { trivial[rid] = false; continue; }

                basic_rule &brx = m_req_rules[rid];
                brx.order = order;
                continue;
            }

            // Check intrinsic labels for invalid ones
            check(br.intr, m_pt);

            // Get a list of unique labels
            std::map<label_t, bool> lmap;
            for (size_t i = 0; i < br.intr.size(); i++) {
                lmap[br.intr[i]] = true;
            }

            // Is trivial rule (all blocks allowed)
            if (lmap.size() == m_pt.nlabels()) {
                trivial[rid] = true;
                continue;
            }

            // Is trivial rule (all blocks forbidden)
            if (lmap.size() == 0) {
                trivial[rid] = false;
                continue;
            }

            // Transfer labels
            basic_rule &brx = m_req_rules[rid];
            for (std::map<label_t, bool>::iterator it = lmap.begin();
                    it != lmap.end(); it++) {
                brx.intr.push_back(it->first);
            }

            // At last optimize the order
            transfer_order(br.order, brx.order, m_pt);

            // If the order contains only the ref to the intrinsic label
            if (brx.order.size() == 1) {
                // If this does not contain 0, all blocks are forbidden
                if (brx.intr[0] != 0) {
                    trivial[rid] = false;
                    m_req_rules.erase(rid);
                    continue;
                }
                if (brx.intr.size() != 1) {
                    brx.intr.clear();
                    brx.intr.push_back(0);
                }
            }
        }
    }

    // Look in the required rules for duplicates
    std::map<rule_id, rule_id> ident, sim;
    for (rule_list::iterator ir1 = m_req_rules.begin();
            ir1 != m_req_rules.end(); ir1++) {

        // Skip if the rule is already marked as identical to another one
        if (ident.find(ir1->first) != ident.end()) continue;

        basic_rule &br1 = ir1->second;

        // Loop over all later rules
        rule_list::iterator ir2 = ir1; ir2++;
        for (; ir2 != m_req_rules.end(); ir2++) {

            // First compare the evaluation order
            const basic_rule &br2 = ir2->second;
            if (! (br2.order == br1.order)) continue;
            if (! (br2.intr == br1.intr)) {
                sim[ir2->first] = ir1->first;
            }
            else {
                ident[ir2->first] = ir1->first;
            }
        }
    }

    // Then look in the trivial rules for duplicates
    std::map<bool, rule_id> tr;
    for (std::map<rule_id, bool>::iterator it = trivial.begin();
            it != trivial.end(); it++) {

        std::map<bool, rule_id>::iterator itr = tr.find(it->second);
        if (itr == tr.end())
            tr[it->second] = it->first;
        else
            ident[it->first] = itr->second;
    }

    // Now, loop over all products
    for (size_t i = 0; i < rule.get_n_products(); i++) {

        // Add the rules to the current product
        std::map<rule_id, rule_list::iterator> cur_pr;

        bool has_trivial = false, allowed = true;
        for (evaluation_rule::product_iterator pit = rule.begin(i);
                pit != rule.end(i); pit++) {

            // Get the correct rule id
            rule_id rid = rule.get_rule_id(pit);
            std::map<rule_id, rule_id>::iterator iid = ident.find(rid);
            if (iid != ident.end()) { rid = iid->second; }

            // Check if the rule belongs to required rules
            rule_list::iterator ir = m_req_rules.find(rid);
            if (ir == m_req_rules.end()) {
                // Otherwise it should be a trivial rule
                if (tr[true] == rid) { // trivial allowed
                    has_trivial = true; allowed = true;
                    continue;
                }
                else if (tr[false] == rid) { // trivial forbidden
                    has_trivial = true; allowed = false;
                    break;
                }
                else {
                    throw bad_symmetry(g_ns, k_clazz, method,
                            __FILE__, __LINE__, "");
                }
            }

            cur_pr[rid] = ir;
        }

        // Is there a trivial rule in the product?
        if (has_trivial) {
            // Does it allow all blocks?
            if (allowed) {
                // A trivial rule that allows all blocks as the only rule in
                // the product allows all blocks for the whole evaluation rule
                if (cur_pr.size() == 0) {
                    m_req_rules.clear();
                    m_req_rules[tr[true]] = basic_rule(label_group(1, 0),
                            std::vector<size_t>(0));

                    cur_pr[tr[true]] = m_req_rules.find(tr[true]);
                    m_products.clear();
                    m_products.push_back(cur_pr);
                    return;
                }
            }
            // A trivial rule that forbids all blocks in the product results
            // in all blocks being forbidden for the product, thus the product
            // is unnecessary
            else continue;
        }

        if (cur_pr.size() == 0) continue;

        // Loop over existing products and see if the current one already
        // exists
        std::list<product_t>::iterator it = m_products.begin();
        for (; it != m_products.end(); it++) {
            if (cur_pr == *it) break;
        }

        if (it == m_products.end())
            m_products.push_back(cur_pr);
    }

    // The last step is to optimize if there are similar rules
    optimize_similars(sim);
}

void transfer_rule::optimize_similars(const std::map<rule_id, rule_id> &sim) {

    // TO BE IMPLEMENTED....

//    if (sim.empty()) return;
//
//    // Loop over all similar rules
//    for (std::map<rule_id, rule_id>::iterator is = sim.begin();
//            is != sim.end(); is++) {
//
//        rule_id rid1 = is->first, rid2 = is->second;
//
//        // Find all products containing rid1 and rid2
//        std::map<product_list::iterator> p1, p2;
//        for (std::list<product>::iterator it = m_setup.begin();
//                    it != m_setup.end(); it++) {
//
//                if (it->find(rid1) != it->end()) p1.push_back(it);
//                if (it->find(rid2) != it->end()) p2.push_back(it);
//            }
//
//            // Do they occur equally often?
//            if (p1.size() != p2.size()) continue;
//
//            // Do they both occur alone in products
//            if (p1.size() == 1 && p1[0]->size() == 1 && p2[0]->size() == 1) {
//
//                // Transfer the intrinsic labels from 2 to 1
//                rule_list::iterator ir1 = m_rules.find(rid1);
//                rule_list::iterator ir2 = m_rules.find(rid2);
//
//                basic_rule &br1 = ir1->second;
//                const basic_rule &br2 = ir2->second;
//
//                label_group intr(br1.intr);
//                for (size_t j = 0; j < br2.intr.size(); j++) {
//                    size_t k = 0;
//                    for (; k < br1.intr.size(); k++) {
//                        if (br1.intr[k] == br2.intr[j]) break;
//                    }
//                    if (k == br1.intr.size()) intr.push_back(br2.intr[j]);
//                }
//
//                br1.intr = intr;
//                m_rules.erase(rid2);
//                m_setup.erase(p2[0]);
//
//                continue;
//            }
//
//            // Do they occur in the same products?
//            size_t i = 0;
//            for (; i < p1.size(); i++) { if (p1[i] != p2[i]) break; }
//            if (i == p1.size()) {
//
//                // Determine the common intrinsic labels in both rules
//                rule_list::iterator ir1 = m_rules.find(rid1);
//                rule_list::iterator ir2 = m_rules.find(rid2);
//
//                label_group intr;
//                basic_rule &br1 = ir1->second;
//                const basic_rule &br2 = ir2->second;
//                for (size_t j = 0; j < br1.intr.size(); j++) {
//                    size_t k = 0;
//                    for (; k < br2.intr.size(); k++) {
//                        if (br1.intr[j] == br2.intr[k]) break;
//                    }
//                    if (k == br2.intr.size()) continue;
//
//                    intr.push_back(br1.intr[k]);
//                }
//                br1.intr = intr;
//
//                m_rules.erase(rid2);
//                for (size_t j = 0; j < p1.size(); j++) m_setup[i].erase(rid2);
//            }
//        }
//    }
}

bool transfer_rule::is_equal(const std::vector<size_t> &o1,
        const std::vector<size_t> &o2) {

    if (o1.size() != o2.size()) return false;

    size_t i = 0;
    for (; i < o1.size(); i++) {
        if (o1[i] != o2[i]) break;
    }
    if (i != o1.size()) return false;

    return true;
}

bool transfer_rule::is_eaual(const transfer_rule::label_group &i1,
        const transfer_rule::label_group &i2) {

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

bool transfer_rule::is_equal(const product_t &pr1, const product_t &pr2) {

    if (pr1.size() != pr2.size()) return false;

    product_t::const_iterator it1 = pr1.begin(), it2 = pr2.begin();
    for (; it1 != pr1.end(); it1++, it2++) {
        if (it1->first != it2->first) break;
    }
    return it1 == pr1.end();
}


void transfer_rule::check(const transfer_rule::label_group &intr,
        const product_table_i &pt) {

#ifdef LIBTENSOR_DEBUG
    static const char *method =
            "check(const label_group &, const product_table_i &)";

    for (size_t i = 0; i < intr.size(); i++) {
        if (! pt.is_valid(intr[i]))
            throw bad_symmetry(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Invalid label.");
    }
#endif
}

bool transfer_rule::check(const std::vector<size_t> &order, size_t ndims) {

#ifdef LIBTENSOR_DEBUG

    static const char *method =
            "check(const std::vector<size_t> &order, size_t ndims)";

    bool has_intrinsic = false;
    for (size_t i = 0; i < order.size(); i++) {
        if (order[i] == evaluation_rule::k_intrinsic) {
            has_intrinsic = true; continue;
        }
        if (order[i] >= ndims)
            throw bad_symmetry(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Invalid order.");
    }

    return has_intrinsic;

#else

    for (size_t i = 0; i < order.size(); i++) {
        if (order[i] == evaluation_rule::k_intrinsic)
            return true;
    }
    return false;
#endif
}

void transfer_rule::transfer_order(const std::vector<size_t> &from,
        std::vector<size_t> &to, const product_table_i &pt) {

    for (size_t i = 0; i < from.size(); i++) {

        if (from[i] == evaluation_rule::k_intrinsic) {
            to.push_back(evaluation_rule::k_intrinsic);
            continue;
        }

        size_t j = i + 1;
        for (; j < from.size(); j++) { if (from[j] != from[i]) break; }

        size_t neq = j - i;
        if (neq == 1) {
            to.push_back(from[i]); continue;
        }

        // Check if we can reduce identical dims to zero or one
        std::map<label_t, label_t> res;
        label_group lg(neq);
        for (label_t l = 0; l < pt.nlabels(); l++) {
            res[l] = pt.invalid();

            for (size_t k = 0; k < neq; k++) lg[k] = l;

            label_t &rr = res[l];
            label_t lt = 0;
            for (; lt < pt.nlabels(); lt++) {
                if (pt.is_in_product(lg, lt)) {
                    if (rr != pt.invalid()) break;
                    rr = lt;
                }
            }
            if (lt != pt.nlabels()) break;
        }

        // May be we can reduce the dims
        if (res.size() == pt.nlabels()) {
            bool is_zero = true, is_ident = true;
            for (label_t l = 0; l < pt.nlabels(); l++) {
                is_zero = is_zero && (res[l] == 0);
                is_ident = is_ident && (res[l] == l);
            }
            if (is_ident) { to.push_back(from[i]); i = j - 1; continue; }
            if (is_zero) { i = j - 1; continue; }
        }

        // Ok we cannot reduce them => just copy
        for (size_t k = i; k < j; k++) to.push_back(from[k]);
        i = j - 1;
    }

}

} // namespace libtensor
