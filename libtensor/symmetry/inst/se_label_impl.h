#ifndef LIBTENSOR_SE_LABEL_IMPL_H
#define LIBTENSOR_SE_LABEL_IMPL_H

#include "../label/transfer_rule.h"

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
void se_label<N, T>::set_rule(const label_set_t &lts) {

    static const char *method = "set_rule(const label_set &)";

#ifdef LIBTENSOR_DEBUG
    // Check the intrinsic labels for invalid labels
    for (label_set_t::const_iterator it = lts.begin(); it != lts.end(); it++) {
        if (! m_pt.is_valid(*it))
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "lts.");
    }
#endif

    // Now start updating the evaluation rule by clearing the old
    m_rule.clear_all();

    if (lts.empty()) return;

    basic_rule<N> br(lts);
    for (size_t i = 0; i < N; i++) br[i] = 1;

    typename evaluation_rule<N>::rule_id_t id = m_rule.add_rule(br);
    m_rule.add_product(id);
}

template<size_t N, typename T>
void se_label<N, T>::set_rule(const evaluation_rule<N> &rule) {

    static const char *method = "set_rule(const evaluation_rule<N> &)";

    m_rule = rule;
    transfer_rule<N>(rule, m_pt.get_id()).perform(m_rule);
}


template<size_t N, typename T>
void se_label<N, T>::permute(const permutation<N> &p) {

    m_blk_labels.permute(p);

    for (typename evaluation_rule<N>::rule_iterator it = m_rule.begin();
            it != m_rule.end(); it++) {

        basic_rule<N> &br = m_rule.get_rule(m_rule.get_rule_id(it));
        p.apply(br);
    }
}

template<size_t N, typename T>
bool se_label<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

    const dimensions<N> &bidims = m_blk_labels.get_block_index_dims();
    return bidims.equals(bis.get_block_index_dims());
}

template<size_t N, typename T>
bool se_label<N, T>::is_allowed(const index<N> &idx) const {

    typedef typename evaluation_rule<N>::rule_id_t rule_id_t;

    static const char *method = "is_allowed(const index<N> &)";

#ifdef LIBTENSOR_DEBUG
    const dimensions<N> &bidims = m_blk_labels.get_block_index_dims();
    // Test, if index is valid block index
    for (size_t i = 0; i < N; i++) {
        if (idx[i] >= bidims[i]) {
            throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "idx.");
        }
    }
#endif

    // Construct the block label
    sequence<N, label_t> blk(product_table_i::k_invalid);
    for (size_t i = 0; i < N; i++) {
        size_t dim_type = m_blk_labels.get_dim_type(i);
        blk[i] = m_blk_labels.get_label(dim_type, idx[i]);
    }

    // Determine which basic rules are allowed.
    label_set_t complete_set = m_pt.get_complete_set();

    std::map<rule_id_t, bool> allowed;
    for (typename evaluation_rule<N>::rule_iterator it = m_rule.begin();
            it != m_rule.end(); it++) {

        rule_id_t rid = m_rule.get_rule_id(it);
        const basic_rule<N> &br = m_rule.get_rule(it);

        product_table_i::label_group_t lg;
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < br[i]; j++) lg.insert(blk[i]);
        }

        const label_set_t &target = br.get_target();
        // No dimensions in sequence or empty target == not allowed
        if (lg.size() == 0 || target.size() == 0) {
            allowed[rid] = false;
            continue;
        }
        // Label of one or more dimensions invalid or target contains all labels
        // == allowed
        if (lg.count(product_table_i::k_invalid) != 0 ||
                target.size() == complete_set.size()) {
            allowed[rid] = true;
            continue;
        }

        // Loop over all target labels and see if one is contained in the
        label_set_t::const_iterator ii = target.begin();
        for (; ii != target.end(); ii++) {
            if (m_pt.is_in_product(lg, *ii)) break;
        }
        allowed[rid] = (ii != target.end());
    }

    // Loop over all products in the evaluation rule
    for (size_t i = 0; i < m_rule.get_n_products(); i++) {

        // Loop over all terms in the current product
        bool is_allowed = true;
        for (typename evaluation_rule<N>::product_iterator itr =
                m_rule.begin(i); itr != m_rule.end(i); itr++) {

            is_allowed = is_allowed && allowed[m_rule.get_rule_id(itr)];
        }

        if (is_allowed) return true;
    }

    return false;
}

} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_IMPL_H

