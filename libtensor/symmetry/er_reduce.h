#ifndef LIBTENSOR_ER_REDUCE_H
#define LIBTENSOR_ER_REDUCE_H

#include <libtensor/timings.h>
#include "evaluation_rule.h"
#include "product_table_i.h"


namespace libtensor {


/** \brief Reduce dimension of evaluation rule by summation.

    \ingroup libtensor_symmetry
 **/
template<size_t N, size_t M>
class er_reduce : public timings< er_reduce<N, M> > {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename product_table_i::label_group_t label_group_t;

private:
    const evaluation_rule<N> &m_rule; //!< Input rule
    const sequence<N, size_t> m_rmap; //!< Reduction map
    const sequence<M, label_group_t> m_rdims; //!< Reduction dimension
    const product_table_i &m_pt; //!< Product table

public:
    er_reduce(const evaluation_rule<N> &rule, const sequence<N, size_t> &rmap,
            const sequence<M, label_group_t> rdims, const std::string &id);

    void perform(evaluation_rule<N - M> &rule) const;
};


} // namespace libtensor


#endif // LIBTENSOR_ER_REDUCE_H
