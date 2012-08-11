#ifndef LIBTENSOR_ER_REDUCE_H
#define LIBTENSOR_ER_REDUCE_H

#include <libtensor/timings.h>
#include "evaluation_rule.h"
#include "product_table_i.h"


namespace libtensor {


/** \brief Reduce dimension of evaluation rule by summation over
        certain dimensions.

    The reduction is performed according to \c m_rmap and \c m_rdim.
    \c m_rmap specifies which input dimensions end up at which output
    dimensions, as well as the input dimensions which are going to be
    reduced. The latter dimensions are given by values larger than N - M
    in \c m_rmap. Since the reduction can be performed in multiple steps
    the value also indicates the reduction step (i.e. value \f$ N - M \f$
    refers to the first reduction step, \f$ N - M + 1 \f$ to the second, ...).

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
    /** \brief Constructor
        \param rule Input rule
        \param rmap Index map input->output
        \param rdims Labels of the reduction steps
        \param id Product table ID
     **/
    er_reduce(const evaluation_rule<N> &rule, const sequence<N, size_t> &rmap,
            const sequence<M, label_group_t> rdims, const std::string &id);

    /** \brief Destructor
     **/
    ~er_reduce();

    /** \brief Perform reduction
        \param rule Output rule
     **/
    void perform(evaluation_rule<N - M> &rule) const;
};


} // namespace libtensor


#endif // LIBTENSOR_ER_REDUCE_H
