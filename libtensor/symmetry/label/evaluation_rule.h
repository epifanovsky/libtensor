#ifndef LIBTENSOR_EVALUATION_RULE_H
#define LIBTENSOR_EVALUATION_RULE_H

#include <map>
#include <vector>
#include "../../core/sequence.h"
#include "../../exception.h"
#include "product_table_i.h"

namespace libtensor {

/** \brief Evaluation rule to determine allowed blocks of a block %tensor.

    The evaluation rule is the container structure used by se_label to
    determine allowed blocks of a N-dim block %tensor by its block labels.
    It comprises
    - a list of unique N-dim sequences
    - a list of lists of index-label pairs where the index refers to one of
      the N-dim sequences.

    The list of unique sequences can be setup using the function
    \c add_sequence() and can be obtained by index access. \c add_sequence()
    returns the index of the added sequences which should be used when new
    index-label pairs are added to the list of lists.

    The function \c add_product() allows to add a new list of index-label pairs,
    thereby also adding the first pair to the list. The return value is the
    index of the newly added list which should be used to add further
    index-label pairs to this list via \c add_to_product().

    For details on how the evaluation rule is used to determine allowed
    blocks please refer to the documentation of \sa se_label.

    \ingroup libtensor_symmetry
 **/
template<size_t N>
class evaluation_rule {
public:
    static const char *k_clazz; //!< Class name

private:
    typedef std::multimap<size_t, product_table_i::label_t> product_t;

public:
    typedef typename product_table_i::label_t label_t;
    typedef typename product_t::const_iterator iterator;

private:
    std::vector< sequence<N, size_t> > m_sequences;
    std::vector<product_t> m_setup;

public:
//    /** \brief Default constructor
//
//        Creates an empty evaluation rule.
//     **/
//    evaluation_rule() { }

    //! \name Manipulation functions
    //@{

    /** \brief Add a new sequence to the list of sequences.
        \param seq Sequence to be added.
        \return Index of the sequence.

        The function checks, if an identical sequence is already present in
        the list. If this is the case, it returns the number of this sequence
        without adding a new one.
     **/
    size_t add_sequence(const sequence<N, size_t> &seq);

    /** \brief Add a new product to the list of products
        \param seq_no Sequence index.
        \param target Label.
        \return Index of the new product.
     **/
    size_t add_product(size_t seq_no, label_t target);

    /** \brief Add another index-label pair to a product
        \param no Number of the product to add to
        \param seq_no Sequence index.
        \param target Label.
     **/
    void add_to_product(size_t no, size_t seq_no, label_t target);

    /** \brief Delete the list of lists
     **/
    void clear_setup() { m_setup.clear(); }

    /** \brief Delete the list of lists and the sequences
     **/
    void clear_all() {
        m_setup.clear(); m_sequences.clear();
    }

    //@}

    //! \name Access functions (read only)
    //@{

    /** \brief Return the number of sequences.
     **/
    size_t get_n_sequences() const { return m_sequences.size(); }

    /** \brief Access a sequence.
        \param n Sequence index.
     **/
    const sequence<N, size_t> operator[](size_t n) const {
#ifdef LIBTENSOR_DEBUG
        if (n >= m_sequences.size())
            throw bad_parameter(g_ns, k_clazz,
                    "operator[](size_t)", __FILE__, __LINE__, "n");
#endif

        return m_sequences[n];
    }

    /** \brief Return the number of products.
     **/
    size_t get_n_products() const { return m_setup.size(); }

    /** \brief STL-style iterator to the 1st index-label pair in a product
        \param no Product number.
     **/
    iterator begin(size_t no) const {
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
    iterator end(size_t no) const {
#ifdef LIBTENSOR_DEBUG
        if (no >= m_setup.size())
            throw bad_parameter(g_ns, k_clazz,
                    "end(size_t)", __FILE__, __LINE__, "no");
#endif

        return m_setup[no].end();
    }

    /** \brief Return the sequence index of the current pair
        \param it Iterator
     **/
    size_t get_seq_no(iterator it) const {
#ifdef LIBTENSOR_DEBUG
        if (! is_valid(it))
            throw bad_parameter(g_ns, k_clazz, "get_seq_no(iterator)",
                    __FILE__, __LINE__, "it");
#endif

        return it->first;

    }

    /** \brief Return the sequence belonging to the current pair
        \param it Iterator.
     **/
     const sequence<N, size_t> &get_sequence(iterator it) const {
#ifdef LIBTENSOR_DEBUG
        if (! is_valid(it))
            throw bad_parameter(g_ns, k_clazz, "get_sequence(iterator)",
                    __FILE__, __LINE__, "it");
#endif

        return m_sequences[it->first];
    }

    /** \brief Return the label belonging to the current pair
        \param it Iterator pointing to a rule
        \return Rule ID
     **/
    label_t get_target(iterator it) const {
#ifdef LIBTENSOR_DEBUG
        if (! is_valid(it))
            throw bad_parameter(g_ns, k_clazz, "get_target(iterator)",
                    __FILE__, __LINE__, "it");
#endif

        return it->second;
    }
    //@}

private:
    bool is_valid(iterator it) const;
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

