#ifndef LIBTENSOR_SE_LABEL_H
#define LIBTENSOR_SE_LABEL_H

#include "../defs.h"
#include "../core/dimensions.h"
#include "../core/mask.h"
#include "../core/symmetry_element_i.h"
#include "bad_symmetry.h"
#include "label/block_labeling.h"
#include "label/evaluation_rule.h"
#include "label/product_table_container.h"
#include "label/product_table_i.h"
#include "label/transfer_rule.h"

namespace libtensor {

/** \brief Symmetry element for labels assigned to %tensor blocks
    \tparam N Symmetry cardinality (%tensor order).
    \tparam T Tensor element type.

    This %symmetry element establishes a set of allowed blocks on the basis of
    block labels along each dimension, an evaluation rule, and a product table.

    The construction of \c se_label using the default constructor initializes
    the class with no block labels and no evaluation rule set. To set the
    labeling of the blocks in each dimension, first obtain the
    \c block_labeling object using the member function \c get_labeling(). Then,
    use the functions provided by the class \c block_labeling to set the labels
    (for details refer to the documentation of \sa block_labeling). To set the
    evaluation rule three functions \c set_rule() are provided:
    - \c set_rule(label_t) --
    - \c set_rule(const label_set_t &) --
    - \c set_rule(const evaluation_rule<N> &) --

    Allowed blocks are determined from the evaluation rule as follows:
    - All blocks are forbidden, if the rule setup of the evaluation rule is
      empty
    - A block is allowed, if it is allowed by any of the products in the rule
      setup
    - A block is allowed by a product, if it is allowed by all basic rules in
      this product
    - A block is allowed by a basic rule, if the product of labels specified
      by the rule contains the target label
    - The product of labels is formed from the block labels using the sequence
      of multiplicities of the basic rule.
    - If this is not possible since the evaluation order has zero length or
      no intrinsic labels are set, but the evaluation order comprises the
      intrinsic label index, all blocks are discarded by this rule.
    - If any product contains an invalid label the block is allowed.

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

    /** \brief Set the evaluation rule to consist of only one basic rule.
        \param lt Target label.

        Replaces any existing rule with the basic rule given by the parameters.
     **/
    void set_rule(const label_t &lt) {

        label_set_t tls; tls.insert(lt);
        set_rule(tls);
    }

    /** \brief Set the evaluation rule to consist of only one basic rule.
        \param tls Set of target labels.

        Replaces any existing rule with a basic rule.
     **/
    void set_rule(const label_set_t &tls);

    /** \brief Set the evaluation rule to a composite rule.
        \param rule Composite evaluation rule.

        The function checks the given rule on validity and replaces any
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

#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

    extern template class se_label<1, double>;
    extern template class se_label<2, double>;
    extern template class se_label<3, double>;
    extern template class se_label<4, double>;
    extern template class se_label<5, double>;
    extern template class se_label<6, double>;

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "inst/se_label_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES

#endif // LIBTENSOR_SE_LABEL_H

