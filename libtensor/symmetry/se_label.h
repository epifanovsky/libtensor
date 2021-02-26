#ifndef LIBTENSOR_SE_LABEL_H
#define LIBTENSOR_SE_LABEL_H

#include <libtensor/core/symmetry_element_i.h>
#include "block_labeling.h"
#include "evaluation_rule.h"
#include "product_table_container.h"

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
    as its own object. It can be set using one of the \c set_rule() functions
    provided:
    - \c set_rule(label_t)
    - \c set_rule(const label_set_t &)
    - \c set_rule(const evaluation_rule<N> &)

    The evaluation rule determines allowed blocks from the product table and
    the sequence of labels of a given block. For details please refer to the
    documentation of \sa evaluation_rule.

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class se_label : public symmetry_element_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name
    static const char k_sym_type[]; //!< Symmetry type

    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_set_t label_set_t;

private:
    block_labeling<N> m_blk_labels; //!< Block index labels
    evaluation_rule<N> m_rule; //!< Label evaluation rule

    const product_table_i &m_pt; //!< Product table

public:
    //! \name Construction and destruction
    //@{
    /** \brief Initializes the %symmetry element
        \param bidims Block %index dimensions.
        \param id Table ID
     **/
    se_label(const dimensions<N> &bidims, const std::string &id) :
        m_blk_labels(bidims), 
        m_pt(product_table_container::get_instance().req_const_table(id)) {
    }


    /** \brief Copy constructor
     **/
    se_label(const se_label<N, T> &elem) :
        m_blk_labels(elem.m_blk_labels), m_rule(elem.m_rule),
        m_pt(product_table_container::get_instance().req_const_table(
                elem.m_pt.get_id())) {
    }


    /** \brief Virtual destructor
     **/
    virtual ~se_label() {
        product_table_container::get_instance().ret_table(m_pt.get_id());
    }

    //@}

    //!    \name Manipulating functions
    //@{

    /** \brief Obtain the block index labeling
     **/
    block_labeling<N> &get_labeling() { return m_blk_labels; }

    /** \brief Obtain the block index labeling (const version)
     **/
    const block_labeling<N> &get_labeling() const { return m_blk_labels; }

    /** \brief Set the evaluation rule to consist of only one 1-term product.
        \param intr Intrinsic label.

        Replaces any existing rules.
     **/
    void set_rule(label_t intr);

    /** \brief Set the evaluation rule to consist of several 1-term products
            (one product for each intrinsic label in the set)
        \param intr Set of intrinsic labels.

        Replaces any existing rule.
     **/
    void set_rule(const label_set_t &intr);

    /** \brief Set the evaluation rule to the given rule.
        \param rule Evaluation rule.

        The function checks the validity of the given rule and replaces any
        previously given rule.
     **/
    void set_rule(const evaluation_rule<N> &rule) { m_rule = rule; }
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

    //!    \name Implementation of symmetry_element_i<N, T>
    //@{

    /** \copydoc symmetry_element_i<N, T>::get_type()
     **/
    virtual const char *get_type() const {
        return k_sym_type;
    }

    /** \copydoc symmetry_element_i<N, T>::clone()
     **/
    virtual symmetry_element_i<N, T> *clone() const {
        return new se_label<N, T>(*this);
    }

    /** \copydoc symmetry_element_i<N, T>::permute
     **/
    void permute(const permutation<N> &perm);

    /** \copydoc symmetry_element_i<N, T>::is_valid_bis
     **/
    virtual bool is_valid_bis(const block_index_space<N> &bis) const;

    /** \copydoc symmetry_element_i<N, T>::is_allowed
     **/
    virtual bool is_allowed(const index<N> &idx) const;

    /** \copydoc symmetry_element_i<N, T>::apply(index<N>&)
     **/
    virtual void apply(index<N> &idx) const { }

    /** \copydoc symmetry_element_i<N, T>::apply(
            index<N>&, transf<N, T>&)
     **/
    virtual void apply(index<N> &idx, tensor_transf<N, T> &tr) const { }
    //@}

};


template<size_t N, typename T>
const char se_label<N, T>::k_clazz[] = "se_label<N, T>";


template<size_t N, typename T>
const char se_label<N, T>::k_sym_type[] = "label";


template<size_t N, typename T>
inline
bool se_label<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

    const dimensions<N> &bidims = m_blk_labels.get_block_index_dims();
    return bidims.equals(bis.get_block_index_dims());
}


} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_H

