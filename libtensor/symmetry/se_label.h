#ifndef LIBTENSOR_SE_LABEL_H
#define LIBTENSOR_SE_LABEL_H

#include <libtensor/core/symmetry_element_i.h>
#include "block_labeling.h"
#include "evaluation_rule.h"
#include "product_table_i.h"

namespace libtensor {

/** \brief Symmetry element for labels assigned to %tensor blocks
    \tparam N Symmetry cardinality (%tensor order).
    \tparam T Tensor element type.

    This %symmetry element establishes a set of allowed blocks on the basis of
    block labels along each dimension, an evaluation rule, and a product table.

    The construction of \c se_label using the default constructor initializes
    the class with a product table, but no block labels and no evaluation rule
    set. The labeling of the blocks is stored in the \c block_labeling object
    that can be obtained using the function \c get_labeling(). This object
    provides all functions necessary to set the labels. For details refer to
    the documentation of \sa block_labeling. The evaluation rule is also stored
    as its own object. It be set using one of the \c set_rule() functions
    provided:
    - \c set_rule(label_t)
    - \c set_rule(const label_set_t &)
    - \c set_rule(const evaluation_rule<N> &)

    Allowed blocks are determined from the evaluation rule as follows:
    - All blocks are forbidden, if the rule setup is empty.
    - A block is allowed, if it is allowed by any of the products in the rule
      setup.
    - A block is allowed by a product, if it is allowed by all terms in this
      product.
    - All blocks are allowed by a term, if the intrinsic label is the invalid
      label.
    - A block is forbidden by a term, if the respective sequence contains only
      zeroes.
    - A block is allowed by a term, if one of the block labels for which the
      sequence is non-zero is the invalid label.
    - A block is allowed by a term, if the product of labels specified
      by the sequence and the intrinsic label contains the target label.
    - The product of labels is a list of labels containing the intrinsic label
      once and the i-th block label n times, if the i-th entry of the sequence
      is n.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class se_label : public symmetry_element_i<N, T> {
public:
    static const char *k_clazz; //!< Class name
    static const char *k_sym_type; //!< Symmetry type

    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_set_t label_set_t;

private:
    block_labeling<N> m_blk_labels; //!< Block index labels
    evaluation_rule<N> m_rule; //!< Label evaluation rule

    const product_table_i &m_pt; //!< Product table

public:
    //!	\name Construction and destruction
    //@{
    /** \brief Initializes the %symmetry element
        \param bidims Block %index dimensions.
     **/
    se_label(const dimensions<N> &bidims, const std::string &id);

    /**	\brief Copy constructor
     **/
    se_label(const se_label<N, T> &elem);

    /**	\brief Virtual destructor
     **/
    virtual ~se_label();
    //@}

    //!	\name Manipulating functions
    //@{

    /** \brief Obtain the block index labeling
     **/
    block_labeling<N> &get_labeling() { return m_blk_labels; }

    /** \brief Obtain the block index labeling (const version)
     **/
    const block_labeling<N> &get_labeling() const { return m_blk_labels; }

    /** \brief Set the evaluation rule to consist of only one 1-term product.
        \param intr Intrinsic label.
        \param target Target label

        Replaces any existing rules.
     **/
    void set_rule(label_t intr, label_t target = 0);

    /** \brief Set the evaluation rule to consist of several 1-term products
            (one product for each intrinsic label in the set)
        \param intr Set of intrinsic labels.
        \param target Target label.

        Replaces any existing rule.
     **/
    void set_rule(const label_set_t &intr, label_t target = 0);

    /** \brief Set the evaluation rule to the given rule.
        \param rule Evaluation rule.

        The function checks the validity of the given rule and replaces any
        previously given rule.
     **/
    void set_rule(const evaluation_rule<N> &rule);
    //@}

    //! \name Access functions
    //@{

    /** \brief Return the current evaluation rule.
     **/
    const evaluation_rule<N> &get_rule() const {
        return m_rule;
    }

    /** \brief Returns the id of the product table
     **/
    const std::string &get_table_id() const {
        return m_pt.get_id();
    }

    //@}

    //!	\name Implementation of symmetry_element_i<N, T>
    //@{

    /**	\copydoc symmetry_element_i<N, T>::get_type()
     **/
    virtual const char *get_type() const {
        return k_sym_type;
    }

    /**	\copydoc symmetry_element_i<N, T>::clone()
     **/
    virtual symmetry_element_i<N, T> *clone() const {
        return new se_label<N, T>(*this);
    }

    /**	\copydoc symmetry_element_i<N, T>::permute
     **/
    void permute(const permutation<N> &perm);

    /**	\copydoc symmetry_element_i<N, T>::is_valid_bis
     **/
    virtual bool is_valid_bis(const block_index_space<N> &bis) const;

    /**	\copydoc symmetry_element_i<N, T>::is_allowed
     **/
    virtual bool is_allowed(const index<N> &idx) const;

    /**	\copydoc symmetry_element_i<N, T>::apply(index<N>&)
     **/
    virtual void apply(index<N> &idx) const { }

    /**	\copydoc symmetry_element_i<N, T>::apply(
			index<N>&, transf<N, T>&)
     **/
    virtual void apply(index<N> &idx, transf<N, T> &tr) const { }
    //@}

};

} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_H

