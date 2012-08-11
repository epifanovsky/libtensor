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
void er_merge<N, M>::perform(evaluation_rule<M> &rule) const {

    er_merge<N, M>::start_timer();

    // Loop over products
    for (typename evaluation_rule<N>::const_iterator it = m_rule.begin();
            it != m_rule.end(); it++) {

        const product_rule<N> &pra = *it;
        product_rule<M> &prb = rule.new_product();

        for (typename product_rule<N>::iterator ip = pra.begin();
                ip != pra.end(); ip++) {

            const sequence<N, size_t> &seq1 = pra.get_sequence(ip);
            sequence<M, size_t> seq2(0);
            for (register size_t i = 0; i < N; i++) {
                seq2[m_mmap[i]] += seq1[i];
            }

            size_t nidx = 0;
            for (register size_t i = 0; i < M; i++) {
                if (! m_smsk[i]) continue;

                seq2[i] = seq2[i] % 2;
            }

            prb.add(seq2, pra.get_intrinsic(ip));
        }
    }
    rule.optimize();

    er_merge<N, M>::stop_timer();
}


} // namespace libtensor


#endif // LIBTENSOR_ER_MERGE_IMPL_H
