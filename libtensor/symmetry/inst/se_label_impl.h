#ifndef LIBTENSOR_SE_LABEL_IMPL_H
#define LIBTENSOR_SE_LABEL_IMPL_H

#include <libtensor/defs.h>
//#include "../core/dimensions.h"
//#include "../core/mask.h"
#include "../bad_symmetry.h"
#include "../label/product_table_container.h"

namespace libtensor {

template<size_t N, typename T>
const char *se_label<N, T>::k_clazz = "se_label<N, T>";

template<size_t N, typename T>
const char *se_label<N, T>::k_sym_type = "se_label";

template<size_t N, typename T>
se_label<N, T>::se_label(const dimensions<N> &bidims, const std::string &id) :
    m_blk_labels(bidims),
    m_pt(product_table_container::get_instance().req_const_table(id)) {
}

template<size_t N, typename T>
se_label<N, T>::se_label(const se_label<N, T> &el) :
    m_blk_labels(el.m_blk_labels), m_rule(el.m_rule),
    m_pt(product_table_container::get_instance().req_const_table(
            el.m_pt.get_id())) {
}

template<size_t N, typename T>
se_label<N, T>::~se_label() {

    product_table_container::get_instance().ret_table(m_pt.get_id());
}

template<size_t N, typename T>
void se_label<N, T>::set_rule(label_t intr, label_t target) {

    static const char *method = "set_rule(label_t, label_t)";

    m_rule.clear_all();
    if (target == product_table_i::k_invalid) return;

    size_t seqno = m_rule.add_sequence(sequence<N, size_t>(1));
    m_rule.add_product(seqno, intr, target);
}
template<size_t N, typename T>
void se_label<N, T>::set_rule(const label_set_t &intr, label_t target) {

    static const char *method = "set_rule(const label_set &)";

    // Now start updating the evaluation rule by clearing the old
    m_rule.clear_all();

    if (intr.empty()) return;

    size_t seqno = m_rule.add_sequence(sequence<N, size_t>(1));
    for (label_set_t::const_iterator it = intr.begin(); it != intr.end(); it++)
        m_rule.add_product(seqno, *it, target);
}

template<size_t N, typename T>
void se_label<N, T>::set_rule(const evaluation_rule<N> &rule) {

    static const char *method = "set_rule(const evaluation_rule<N> &)";

    m_rule = rule;
    m_rule.optimize();
}


template<size_t N, typename T>
void se_label<N, T>::permute(const permutation<N> &p) {

    m_blk_labels.permute(p);

    for (size_t i = 0; i < m_rule.get_n_sequences(); i++) {

        p.apply(m_rule[i]);
    }
}

template<size_t N, typename T>
bool se_label<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

    const dimensions<N> &bidims = m_blk_labels.get_block_index_dims();
    return bidims.equals(bis.get_block_index_dims());
}

template<size_t N, typename T>
bool se_label<N, T>::is_allowed(const index<N> &idx) const {

    static const char *method = "is_allowed(const index<N> &)";

#ifdef LIBTENSOR_DEBUG
    const dimensions<N> &bidims = m_blk_labels.get_block_index_dims();
    // Test, if index is valid block index
    for (size_t i = 0; i < N; i++) {
        if (idx[i] >= bidims[i]) {
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "idx.");
        }
    }
#endif

    // Construct the block label
    sequence<N, label_t> blk(product_table_i::k_invalid);
    for (register size_t i = 0; i < N; i++) {
        blk[i] = m_blk_labels.get_label(m_blk_labels.get_dim_type(i), idx[i]);
    }

    // Construct label groups for every sequence in rule
    std::vector<product_table_i::label_group_t> lgs(m_rule.get_n_sequences());
    std::vector<bool> invalid(m_rule.get_n_sequences(), false);
    for (size_t i = 0; i < m_rule.get_n_sequences(); i++) {
        const sequence<N, size_t> &seq = m_rule[i];
        product_table_i::label_group_t &lg = lgs[i];
        for (size_t j = 0; j < N; j++) {
            if (seq[j] == 0) continue;

            lg.insert(lg.end(), seq[j], blk[j]);
            if (blk[j] == product_table_i::k_invalid) {
                invalid[i] = true; break;
            }
        }
    }

    // Loop over all products in the evaluation rule
    for (size_t i = 0; i < m_rule.get_n_products(); i++) {

        // Loop over all terms in the current product
        bool is_allowed = true;
        for (typename evaluation_rule<N>::iterator it =
                m_rule.begin(i); it != m_rule.end(i); it++) {

            // Sequence is empty or target is invalid label ?
            product_table_i::label_group_t &lg = lgs[m_rule.get_seq_no(it)];
            if (lg.size() == 0 ||
                    m_rule.get_target(it) == product_table_i::k_invalid) {
                is_allowed = false; break;
            }

            // block label in sequence or intrinsic label is the invalid label
            if (invalid[m_rule.get_seq_no(it)] ||
                    m_rule.get_intrinsic(it) == product_table_i::k_invalid) {
               continue;
            }

            lg.push_back(m_rule.get_intrinsic(it));
            is_allowed = is_allowed &&
                    m_pt.is_in_product(lg, m_rule.get_target(it));
            lg.pop_back();

            if (! is_allowed) break;
        }

        if (is_allowed) return true;
    }

    return false;
}

} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_IMPL_H

