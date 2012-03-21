#ifndef LIBTENSOR_SIMPLIFY_RULE_H
#define LIBTENSOR_SIMPLIFY_RULE_H

#include "../../timings.h"
#include "evaluation_rule.h"
#include "product_table_i.h"

namespace libtensor {

/** \brief Transfers the contents of an evalutaion_rule to another thereby
        performing checks and optimizations

    The target evaluation_rule is overridden in this process. Basic checks
    w.r.t to the intrinsic labels and evaluation orders are performed.
    The evaluation rule is optimized by removing duplicate and trivial basic
    rules, by merging similar basic rules, by removing duplicate and trivial
    products, and by minimizing the number of products and the number of basic
    rules in a product

    \ingroup libtensor_symmetry
 **/
template<size_t N>
class transfer_rule : public timings< transfer_rule<N> > {
public:
    static const char *k_clazz;

private:
    typedef typename product_table_i::label_t label_t;
    typedef typename product_table_i::label_set_t label_set_t;

    typedef typename evaluation_rule<N>::rule_id_t rule_id_t;
    typedef std::map< rule_id_t, basic_rule<N> > rule_list_t;
    typedef std::map<rule_id_t, rule_id_t> rule_map_t;

    const evaluation_rule<N> &m_from; //!< Evaluation rule to transfer
    const product_table_i &m_pt; //!< Product table

    bool m_mergeable; //!< Flag if labels can be merged
    label_set_t m_merge_set; //!< Result of a merge

public:
    /** \brief Constructor
        \param rule Evaluation rule to be transfered
        \param ndim # dimension allowed in the evaluation order
        \param id ID of product table
     **/
    transfer_rule(const evaluation_rule<N> &rule, const std::string &id);

    /** \brief Destructor
     **/
    ~transfer_rule();

    /** \brief Perform checks and simplifications on rule
        \param rule Target evaluation rule
     **/
    void perform(evaluation_rule<N> &rule);

private:
    /** \brief Optimize basic rules.
        \param[out] opt Optimized rules.
        \param[out] triv Trivial rules (true: allowed, false: forbidden).
     **/
    void optimize_basic(rule_list_t &opt,
            std::map<rule_id_t, bool> &triv) const;

    /** \brief Find similar or duplicate rules in list.
        \param rules List of rules.
        \param[out] sim Similar rules (i.e. sequences are identical).
        \param[out] dupl Duplicate rules (i.e. also targets are identical).
     **/
    static void find_similar(const rule_list_t &rules,
            rule_map_t &sim, rule_map_t &dupl);
};


} // namespace libtensor

#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

    extern template class transfer_rule<1>;
    extern template class transfer_rule<2>;
    extern template class transfer_rule<3>;
    extern template class transfer_rule<4>;
    extern template class transfer_rule<5>;
    extern template class transfer_rule<6>;

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "inst/transfer_rule_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_TRANSFER_RULE_H
