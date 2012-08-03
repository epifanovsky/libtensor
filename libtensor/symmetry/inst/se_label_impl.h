#ifndef LIBTENSOR_SE_LABEL_IMPL_H
#define LIBTENSOR_SE_LABEL_IMPL_H

#include <libtensor/defs.h>
#include <libtensor/core/abs_index.h>
#include "../bad_symmetry.h"
#include "../product_table_container.h"


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
void se_label<N, T>::set_rule(label_t intr) {

    label_set_t ls;
    ls.insert(intr);
    set_rule(ls);
}


template<size_t N, typename T>
void se_label<N, T>::set_rule(const label_set_t &intr) {

    m_rule.clear();
    if (intr.empty()) return;

    sequence<N, size_t> seq(1);
    for (label_set_t::const_iterator it = intr.begin();
            it != intr.end(); it++) {
        product_rule<N> &pr = m_rule.new_product();
        pr.add(seq, *it);
    }
}

template<size_t N, typename T>
void se_label<N, T>::set_rule(const evaluation_rule<N> &rule) {

    m_rule = rule;
    m_rule.optimize();
}


template<size_t N, typename T>
void se_label<N, T>::permute(const permutation<N> &p) {

    m_blk_labels.permute(p);
    eval_sequence_list<N> &sl = m_rule.get_sequences();

    for (size_t i = 0; i < sl.size(); i++) {
        p.apply(sl[i]);
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

    se_label<N, T>::start_timer(method);

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

    bool allowed = m_rule.is_allowed(blk, m_pt);

    se_label<N, T>::stop_timer(method);
    return allowed;
}


} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_IMPL_H

