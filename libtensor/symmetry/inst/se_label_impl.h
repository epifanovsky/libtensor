#ifndef LIBTENSOR_SE_LABEL_IMPL_H
#define LIBTENSOR_SE_LABEL_IMPL_H

#include <libtensor/defs.h>
#include <libtensor/core/abs_index.h>
#include "../bad_symmetry.h"


namespace libtensor {


template<size_t N, typename T>
const char *se_label<N, T>::k_clazz = "se_label<N, T>";


template<size_t N, typename T>
const char *se_label<N, T>::k_sym_type = "label";


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
void se_label<N, T>::permute(const permutation<N> &p) {

    m_blk_labels.permute(p);
    eval_sequence_list<N> &sl = m_rule.get_sequences();

    for (size_t i = 0; i < sl.size(); i++) {
        p.apply(sl[i]);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_IMPL_H

