#ifndef LIBTENSOR_ER_MERGE_IMPL_H
#define LIBTENSOR_ER_MERGE_IMPL_H

#include "../bad_symmetry.h"

namespace libtensor {


template<size_t N, size_t M>
const char *er_merge<N, M>::k_clazz = "er_merge<N, M>";


template<size_t N, size_t M>
er_merge<N, M>::er_merge(const evaluation_rule<N> &rule,
        const sequence<N, size_t> &mmap, const mask<M> &smsk) :
        m_rule(rule), m_mmap(mmap), m_smsk(smsk) {

#ifdef LIBTENSOR_DEBUG
    mask<M> msk;
    for (size_t i = 0; i < N; i++) {

        if (mmap[i] > M) {
            throw bad_symmetry(g_ns, k_clazz, "er_merge(...)",
                    __FILE__, __LINE__, "mmap");
        }

        msk[mmap[i]] = true;
    }

    for (size_t i = 0; i < M; i++) {
        if (! msk[i]) {
            throw bad_symmetry(g_ns, k_clazz, "er_merge(...)",
                    __FILE__, __LINE__, "mmap");
        }
    }
#endif
}


template<size_t N, size_t M>
void er_merge<N, M>::perform(evaluation_rule<M> &to) const {

    er_merge<N, M>::start_timer();

    to.clear();

    // Merge sequences
    const eval_sequence_list<N> &slist1 = m_rule.get_sequences();
    eval_sequence_list<M> slist2;
    std::vector<size_t> smap(slist1.size(), 0);
    for (size_t i = 0; i < slist1.size(); i++) {

        const sequence<N, size_t> &seq1 = slist1[i];
        sequence<M, size_t> seq2(0);

        for (register size_t j = 0; j < N; j++) {
            seq2[m_mmap[j]] += seq1[j];
        }

        size_t nidx = 0;
        for (register size_t j = 0; j < M; j++) {
            if (m_smsk[j]) seq2[j] = seq2[j] % 2;

            nidx += seq2[j];
        }
        smap[i] = (nidx == 0 ? slist1.size() : slist2.add(seq2));
    }

    // Loop over products
    typename evaluation_rule<N>::iterator it = m_rule.begin();
    for (; it != m_rule.end(); it++) {

        const product_rule<N> &pra = *it;

        bool all_allowed = true;

        typename product_rule<N>::iterator ip = pra.begin();
        for (; ip != pra.end(); ip++) {

            // Zero sequence
            if (smap[pra.get_seqno(ip)] != slist1.size()) {
                all_allowed = false; continue;
            }

            if (pra.get_intrinsic(ip) != product_table_i::k_identity) break;
        }
        if (ip != pra.end()) continue;

        if (all_allowed) break;

        product_rule<M> &prb = to.new_product();
        for (typename product_rule<N>::iterator ip = pra.begin();
                    ip != pra.end(); ip++) {
            prb.add(slist2[smap[pra.get_seqno(ip)]], pra.get_intrinsic(ip));
        }
    }
    if (it != m_rule.end()) {
        to.clear();
        product_rule<M> &pr = to.new_product();
        pr.add(sequence<M, size_t>(1), product_table_i::k_invalid);
    }

    er_merge<N, M>::stop_timer();
}


} // namespace libtensor


#endif // LIBTENSOR_ER_MERGE_IMPL_H
