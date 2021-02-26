#ifndef LIBTENSOR_SE_LABEL_IMPL_H
#define LIBTENSOR_SE_LABEL_IMPL_H

#include <libutil/threads/tls.h>
#include <libtensor/defs.h>
#include <libtensor/core/abs_index.h>
#include "../bad_symmetry.h"


namespace libtensor {


class se_label_buffer {
private:
    product_table_i::label_group_t m_lg;

public:
    se_label_buffer() {
        m_lg.reserve(32);
    }

    static product_table_i::label_group_t &get_lg() {
        return libutil::tls<se_label_buffer>::get_instance().get().m_lg;
    }

};


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


template<size_t N, typename T>
bool se_label<N, T>::is_allowed(const index<N> &idx) const {

    product_table_i::label_group_t &lg = se_label_buffer::get_lg();

    // Loop over all products in the evaluation rule
    for(typename evaluation_rule<N>::iterator it = m_rule.begin();
        it != m_rule.end(); ++it) {

        const product_rule<N> &pr = m_rule.get_product(it);
        if(pr.empty()) return false;

        // Loop over all terms in the current product
        typename product_rule<N>::iterator ip = pr.begin();
        for(; ip != pr.end(); ++ip) {

            if(pr.get_intrinsic(ip) == product_table_i::k_invalid) continue;

            // Construct product
            const sequence<N, size_t> &seq = pr.get_sequence(ip);

            lg.clear();
            size_t i = 0;
            for(; i < N; i++) {
                if(seq[i] == 0) continue;
                label_t l = m_blk_labels.get_label(m_blk_labels.get_dim_type(i),
                    idx[i]);
                if(l == product_table_i::k_invalid) break;
                lg.insert(lg.end(), seq[i], l);
            }
            if(i != N) continue;

            if(!m_pt.is_in_product(lg, pr.get_intrinsic(ip))) break;
        }

        if(ip == pr.end()) return true;
    }

    return false;
}


} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_IMPL_H

