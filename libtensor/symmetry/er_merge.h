#ifndef LIBTENSOR_ER_MERGE_H
#define LIBTENSOR_ER_MERGE_H

#include <libtensor/timings.h>
#include "evaluation_rule.h"


namespace libtensor {


/** \brief Reduce dimension of evaluation rule by merge.


    \ingroup libtensor_symmetry
 **/
template<size_t N, size_t M>
class er_merge : public timings< er_merge<N, M> > {
public:
    static const char *k_clazz; //!< Class name

private:
    const evaluation_rule<N> &m_rule; //!< Input rule
    sequence<N, size_t> m_mmap; //!< Merge map
    mask<M> m_smsk; //!< Mask of dimensions that can be simplified.

public:
    er_merge(const evaluation_rule<N> &rule,
            const sequence<N, size_t> &mmap, const mask<M> &smsk);

    void perform(evaluation_rule<M> &rule) const;
};


} // namespace libtensor


#endif // LIBTENSOR_ER_MERGE_H
