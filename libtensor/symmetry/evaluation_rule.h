#ifndef LIBTENSOR_EVALUATION_RULE_H
#define LIBTENSOR_EVALUATION_RULE_H

#include <map>
#include <vector>
#include <libtensor/core/sequence.h>
#include <libtensor/exception.h>
#include "product_table_i.h"


namespace libtensor {


/** \brief Evaluation rule to determine allowed blocks of a block %tensor.

    The evaluation rule is the container structure used by se_label to
    determine allowed blocks of a N-dim block %tensor by its block labels.
    It comprises a list of unique N-dim sequences and a list of "products"
    where each product is a list of index-label pairs. In each pair the index
    refers to one of the N-dim sequences, while the label refers to the
    "intrinsic" label.

    The function \c new_product() allows to create a new empty list of
    index-label tuples. The return value is an iterator to the new list which
    has to be used to add further index-label pairs to this list via
    \c add_to_product().

    For details on how the evaluation rule is used to determine allowed
    blocks please refer to the documentation of \sa se_label.

    \ingroup libtensor_symmetry
 **/
template<size_t N>
class evaluation_rule {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename product_table_i::label_t label_t;
    typedef typename product_table_i::label_set_t label_set_t;

    typedef sequence<N, size_t> sequence_t;
    typedef std::map<size_t, label_t> product_t;
    typedef std::list<product_t>::iterator iterator;
    typedef std::list<product_t>::const_iterator const_iterator;

private:
    std::vector<sequence_t> m_sequences;
    std::list<product_t> m_setup;

public:
    //! \name Manipulation functions
    //@{

    /** \brief Create a new (empty) product in rule setup
     **/
    iterator new_product() {
        return m_setup.insert(m_setup.end(), product_t());
    }

    /** \brief Add index-label pair to a product
        \param it Iterator to the product to add to
        \param seq Sequence.
        \param intr Intrinsic label.
     **/
    void add_to_product(iterator it, const sequence_t &seq, label_t intr);

    /** \brief Tries to optimize the current evaluation rule.

        Optimization is attempted by the following steps:
        - Find always forbidden, always allowed, and duplicate terms in each
          product
        - Delete always allowed or duplicate terms in a product
        - Delete products comprising always forbidden terms
        - Find duplicate products and delete them
        - Find unused sequences and delete them
     **/
    void optimize();

    /** \brief Delete the list of lists
     **/
    void clear() { m_setup.clear(); m_sequences.clear(); }

    //@}

    //! \name Access functions (read only)
    //@{

    /** \brief Return the number of sequences in the rule.
     **/
    size_t get_n_sequences() const { return m_sequences.size(); }

    /** \brief Checks if the sequence exists and returns its index 
        
        If the sequence does not exist the number of sequences is returned.
     **/
    size_t has_sequence(const sequence_t &seq) const;

    /** \brief Access a sequence.
        \param n Sequence index.
     **/
    const sequence_t &operator[](size_t n) const {
#ifdef LIBTENSOR_DEBUG
        if (n >= m_sequences.size())
            throw bad_parameter(g_ns, k_clazz,
                    "operator[](size_t)", __FILE__, __LINE__, "n");
#endif
        return m_sequences[n];
    }

    /** \brief Access a sequence.
        \param n Sequence index.
     **/
    sequence_t &operator[](size_t n) {
#ifdef LIBTENSOR_DEBUG
        if (n >= m_sequences.size())
            throw bad_parameter(g_ns, k_clazz,
                    "operator[](size_t)", __FILE__, __LINE__, "n");
#endif
        return m_sequences[n];
    }

    /** \brief STL-style iterator to the 1st product in the setup
     **/
    iterator begin() { return m_setup.begin(); }

    /** \brief STL-style iterator to the 1st product in the setup
     **/
    const_iterator begin() const { return m_setup.begin(); }

    /** \brief STL-style iterator to the end of the product list
     **/
    iterator end() { return m_setup.end(); }

    /** \brief STL-style iterator to the end of the product list
     **/
    const_iterator end() const { return m_setup.end(); }

    /** \brief Return the n-th product
        \param n Number of product
     **/ 
    const product_t &get_product(const_iterator it) const { return *it; }

    //@}

};


} // namespace libtensor


#endif // LIBTENSOR_EVALUATION_RULE_H
