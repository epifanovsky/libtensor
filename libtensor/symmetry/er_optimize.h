#ifndef LIBTENSOR_ER_OPTIMIZE_H
#define LIBTENSOR_ER_OPTIMIZE_H

#include <libtensor/timings.h>
#include "evaluation_rule.h"


namespace libtensor {


/** \brief Optimizes an evaluation rule.

    Optimization is attempted by the following steps:
    - Find always forbidden and always allowed terms in each product
    - Delete always allowed terms in a product (unless it is the only term)
    - Delete products comprising always forbidden terms
    - Find duplicate products and delete them
    - Find unused sequences and delete them

    \ingroup libtensor_symmetry
 **/
template<size_t N>
class er_optimize : public timings< er_optimize<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    const evaluation_rule<N> &m_rule; //!< Input rule
    bool m_mergable;

public:
    /** \brief Constructor
        \param from Input rule
        \param id Product table ID
     **/
    er_optimize(const evaluation_rule<N> &from, const std::string &id);

    /** \brief Perform optimization
        \param to Optimized copy of input rule.
     **/
    void perform(evaluation_rule<N> &to) const;
};


} // namespace libtensor


#endif // LIBTENSOR_ER_OPTIMIZE_H
