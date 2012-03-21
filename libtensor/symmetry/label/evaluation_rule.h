#ifndef LIBTENSOR_EVALUATION_RULE_H
#define LIBTENSOR_EVALUATION_RULE_H

#include <map>
#include <vector>
#include "../../exception.h"
#include "basic_rule.h"

namespace libtensor {

/** \brief Full evaluation rule to determine allowed blocks of a block %tensor.

    Every possible rule to determine allowed blocks of a block %tensor by
    its block labels can be represented as a sum of products of the elementary
    basic rules. The evaluation rule is the container class for any arbitrarily
    complex rule.

    For details on how the setup of the evaluation rule determines allowed
    blocks refer to the documentation of \sa se_label.

    \ingroup libtensor_symmetry
 **/
template<size_t N>
class evaluation_rule {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef size_t rule_id_t;
    typedef typename basic_rule<N>::label_t label_t;
    typedef typename basic_rule<N>::label_set_t label_set_t;

private:
    typedef std::map< rule_id_t, basic_rule<N> > rule_list_t;

public:
    typedef typename rule_list_t::const_iterator rule_iterator;

private:
    typedef std::map<rule_id_t, rule_iterator> rule_product_t;
    typedef std::vector<rule_product_t> product_list_t;

public:
    typedef typename rule_product_t::const_iterator product_iterator;

private:
    rule_list_t m_rules; //!< List of basic rules
    product_list_t m_setup; //!< Rules setup

    rule_id_t m_next_rule_id; //!< Next rule ID

public:
    /** \brief Default constructor
     **/
    evaluation_rule() : m_next_rule_id(0) { }

    //! \name Manipulation functions
    //@{

    /** \brief Add a new basic rule to the end of the list
        \param br Basic rule
        \return Returns the ID of the new rule
     **/
    rule_id_t add_rule(const basic_rule<N> &br);

    /** \brief Add a new product to the list of products
        \param rule ID of the rule to be part of the new product
        \return Number of the new product
     **/
    size_t add_product(rule_id_t rule);

    /** \brief Add another rule to a product

        \param no Number of the product to add to
        \param rule ID of the rule to be added
     **/
    void add_to_product(size_t no, rule_id_t rule);

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
    rule_id_t get_rule_id(rule_iterator it) const {
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
    basic_rule<N> &get_rule(rule_id_t id) {

        typename rule_list_t::iterator it = m_rules.find(id);
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
    const basic_rule<N> &get_rule(rule_id_t id) const {

        rule_iterator it = m_rules.find(id);
        return get_rule(it);
    }



    /** \brief Return a basic rule
        \param it Iterator pointing to a rule
     **/
    const basic_rule<N> &get_rule(rule_iterator it) const {
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
    rule_id_t get_rule_id(product_iterator pit) const {
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
    const basic_rule<N> &get_rule(product_iterator pit) const {
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
    rule_id_t new_rule_id() { return m_next_rule_id++; }

    bool is_valid_rule(rule_iterator it) const;
    bool is_valid_product_iterator(product_iterator it) const;
};

} // namespace libtensor


#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

    extern template class evaluation_rule<1>;
    extern template class evaluation_rule<2>;
    extern template class evaluation_rule<3>;
    extern template class evaluation_rule<4>;
    extern template class evaluation_rule<5>;
    extern template class evaluation_rule<6>;

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "inst/evaluation_rule_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES

#endif // LIBTENSOR_EVALUATION_RULE_H

