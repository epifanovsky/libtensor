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
class transfer_rule : public timings<transfer_rule> {
public:
    static const char *k_clazz;
    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_group label_group;

private:
    typedef evaluation_rule::rule_id rule_id;
    typedef evaluation_rule::basic_rule basic_rule;
    typedef std::map<rule_id, basic_rule> rule_list;
    typedef std::map<rule_id, rule_list::iterator> product_t;

    const product_table_i &m_pt; //!< Product table
    std::map<size_t, size_t> m_reduce; //!< Reduce ndims to less

    rule_list m_req_rules;
    std::list<product_t> m_products;
    size_t m_ndims; //!< # dimensions

public:
    /** \brief Constructor
        \param rule Evaluation rule to be transfered
        \param ndim # dimension allowed in the evaluation order
        \param id ID of product table
     **/
    transfer_rule(const evaluation_rule &rule,
            size_t max_dim, const std::string &id);

    /** \brief Destructor
     **/
    ~transfer_rule();

    /** \brief Perform checks and simplifications on rule
        \param rule Target evaluation rule
     **/
    void perform(evaluation_rule &rule);

private:
    /** \brief Take the evaluation rule apart and split it into its minimal
            components
        \param from Evaluation rule
     **/
    void analyze(const evaluation_rule &from);

    /** \brief Optimize the products w.r.t duplicate products
     **/
    void optimize_products();

    /** \brief Transfers and optimizes a basic rule
        \param from Source
        \param to Target
        \param pt Product table
     **/
    bool transfer_basic(const basic_rule &from, basic_rule &to);

    /** \brief Compare to evaluation orders
     **/
    static bool equal_order(const std::vector<size_t> &o1,
            const std::vector<size_t> &o2);

    /** \brief Compare to intrinsic labels
     **/
    static bool equal_intr(const label_group &i1, const label_group &i2);

    /** \brief Compare to rules products
     **/
    static bool equal_product(const product_t &pr1, const product_t &pr2);

};


} // namespace libtensor

#endif // LIBTENSOR_SIMPLIFY_RULE_H
