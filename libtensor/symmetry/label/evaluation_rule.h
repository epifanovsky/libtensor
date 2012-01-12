#ifndef LIBTENSOR_EVALUATION_RULE_H
#define LIBTENSOR_EVALUATION_RULE_H

#include <map>
#include "../../exception.h"
#include "product_table_i.h"

namespace libtensor {

/** \brief Evaluation rule for a sum of products of basic rules.

    Every possible label evaluation rule can be expressed as a sum of products
    of basic rules. This is a container class for such composite label
    evaluation rules.

    TODO:
    - extend documentation

    \ingroup libtensor_symmetry
 **/
class evaluation_rule {
public:
    static const char *k_clazz; //!< Class name
    static const size_t k_intrinsic; //!< Dimension refering to intrinsic label

public:
    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_group label_group;

public:
    struct basic_rule {
        std::vector<size_t> order; //!< Evaluation order of dimensions
        label_group intr; //!< Intrinsic labels

        basic_rule(const label_group &intr_ = label_group(),
                const std::vector<size_t> order_ = std::vector<size_t>()) :
            intr(intr_), order(order_) { }
    };

    typedef size_t rule_id;

private:
    typedef std::map<rule_id, basic_rule> rule_list;

public:
    typedef rule_list::const_iterator rule_iterator;

private:
    typedef std::map<rule_id, rule_iterator> rules_product;
    typedef std::vector<rules_product> product_list;

public:
    typedef rules_product::const_iterator product_iterator;

private:
    rule_list m_rules; //!< List of basic rules
    rule_id m_next_rule_id; //!< Next rule ID
    product_list m_setup; //!< Rules setup

public:
    evaluation_rule() : m_next_rule_id(0) { }

    //! \name Manipulation functions
    //@{

    /** \brief Add a new basic rule to the end of the list
        \param intr Intrinsic labels
        \param order Evaluation order
        \return Returns the ID of the new rule
     **/
    rule_id add_rule(const label_group &intr, const std::vector<size_t> &order);

    /** \brief Add a new product to the list of products
        \param rule ID of the rule to be part of the new product
        \return Number of the new product
     **/
    size_t add_product(rule_id rule);

    /** \brief Add another rule to a product

        \param no Number of the product to add to
        \param rule ID of the rule to be added
     **/
    void add_to_product(size_t no, rule_id rule);

    //@}

    //! \name Functions to access the basic rules (read access)
    //@{

    /** \brief STL-style iterator to the start of the list of basic rules
     **/
    rule_iterator begin() const { return m_rules.begin(); }

    /** \brief STL-style iterator to the end of the list of basic rules
     **/
    rule_iterator end() const { return m_rules.end(); }

    /** \brief Returns the ID of the rule given by iterator
        \param it Iterator to rule
        \return Rule ID
     **/
    rule_id get_rule_id(rule_iterator it) const {
#ifdef LIBTENSOR_DEBUG
        if (! is_valid_rule(it))
            throw bad_parameter(g_ns, k_clazz,
                    "get_rule_intrinsic(rule_iterator)",
                    __FILE__, __LINE__, "it");
#endif

        return it->first;
    }

    /** \brief Return a basic rule
        \param id Rule ID
     **/
    basic_rule &get_rule(rule_id id) {

        rule_list::iterator it = m_rules.find(id);
#ifdef LIBTENSOR_DEBUG
        if (it == m_rules.end()) {
            throw bad_parameter(g_ns, k_clazz,
                                "get_rule(rule_id)", __FILE__, __LINE__, "it");
        }
#endif
        return it->second;
    }

    /** \brief Return a basic rule (const version
        \param id Rule ID
     **/
    const basic_rule &get_rule(rule_id id) const {

        rule_iterator it = m_rules.find(id);
        return get_rule(it);
    }



    /** \brief Return a basic rule
        \param it Iterator pointing to a rule
     **/
    const basic_rule &get_rule(rule_iterator it) const {
#ifdef LIBTENSOR_DEBUG
        if (! is_valid_rule(it))
            throw bad_parameter(g_ns, k_clazz,
                    "get_rule(rule_iterator)", __FILE__, __LINE__, "it");
#endif

        return it->second;
    }

    //@}

    //! \name Functions to access the rule setup (read access)
    //@{

    /** \brief Return the number of products
     **/
    size_t get_n_products() const { return m_setup.size(); }

    /** \brief STL-style iterator to the 1st rule in a product
        \param no Product number.
     **/
    product_iterator begin(size_t no) const {

#ifdef LIBTENSOR_DEBUG
        if (no >= m_setup.size())
            throw bad_parameter(g_ns, k_clazz,
                    "begin(size_t)", __FILE__, __LINE__, "no");
#endif

        return m_setup[no].begin();
    }

    /** \brief STL-style iterator to the end of a product
        \param no Product number
     **/
    product_iterator end(size_t no) const {

#ifdef LIBTENSOR_DEBUG
        if (no >= m_setup.size())
            throw bad_parameter(g_ns, k_clazz,
                    "end(size_t)", __FILE__, __LINE__, "no");
#endif

        return m_setup[no].end();
    }

    /** \brief Return the ID of a basic rule
        \param pit Iterator pointing to a rule
        \return Rule ID
     **/
    rule_id get_rule_id(product_iterator pit) const {
#ifdef LIBTENSOR_DEBUG
        if (! is_valid_product_iterator(pit))
            throw bad_parameter(g_ns, k_clazz,
                    "get_rule_id(const_product_iterator)",
                    __FILE__, __LINE__, "pit");
#endif

        return pit->first;
    }

    /** \brief Return the ID of a basic rule
        \param pit Iterator pointing to a rule
        \return Rule ID
     **/
    const basic_rule &get_rule(product_iterator pit) const {
#ifdef LIBTENSOR_DEBUG
        if (! is_valid_product_iterator(pit))
            throw bad_parameter(g_ns, k_clazz,
                    "get_rule(const_product_iterator)",
                    __FILE__, __LINE__, "pit");
#endif

        return pit->second->second;
    }
    //@}

    /** \brief Delete the rule setup
     **/
    void clear_setup() { m_setup.clear(); }

    /** \brief Delete the rule setup as well as the list of rules
     **/
    void clear_all() { m_setup.clear(); m_rules.clear(); }

private:
    rule_id new_rule_id() { return m_next_rule_id++; }

    bool is_valid_rule(rule_iterator it) const;
    bool is_valid_product_iterator(product_iterator it) const;
};

} // namespace libtensor

#endif // LIBTENSOR_EVALUATION_RULE_H

