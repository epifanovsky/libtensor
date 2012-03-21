#ifndef LIBTENSOR_PRODUCT_TABLE_I_H
#define LIBTENSOR_PRODUCT_TABLE_I_H

#include <set>
#include "../../exception.h"

namespace libtensor {


/** \brief Interface for product tables


    - Finite set of labels
    - each label is of type size_t
    - maps direct products -> direct sums
    - no prefactors in direct sum, ony present or non-present
    - assumptions:
        - symmetric
        - special label: identity label \f$ l_x \f$
        - \f$ l_x \otimes l_i \rightarrow l_i \forall l_i \in S_L \f$
        - \f$ l_i \otimes \l_i \rightarrow l_x \oplus ... \forall l_i \in S_L \f$

	This interface specifies the functions which se_label and
	product_table_container use to operate. Any specific implementation of
	product_table_i needs to provide these functions to work with both
	classes.

    TODO:
    - documentation

    \ingroup libtensor_symmetry
 **/
class product_table_i {
public:
    typedef size_t label_t;
    typedef std::set<label_t> label_set_t;
    typedef std::multiset<label_t> label_group_t;

public:
    static const char *k_clazz; //!< Class name
    static const label_t k_invalid; //!< Invalid label


public:
    //! \name Constructors / destructors
    //@{

    /** \brief Virtual destructor
     **/
    virtual ~product_table_i() { }

    /** \brief Creates an identical copy of the product table.
     **/
    virtual product_table_i *clone() const = 0;

    //@}

    /** \brief Returns the id identifying the product table.
     **/
    virtual const std::string &get_id() const = 0;

    /** \brief Returns true if label is valid
        \param l Label to check
     **/
    virtual bool is_valid(label_t l) const = 0;

    /** \brief Returns the set of all valid labels
     **/
    virtual label_set_t get_complete_set() const = 0;

    /** \brief Returns the identity label
     **/
    virtual label_t get_identity() const = 0;

    /** \brief Determines if the label is in the product.

		\param lg Group of labels to take the product of.
		\param l Label to check against.
		\return True if label is in the product, else false.
     **/
    virtual bool is_in_product(const label_group_t &lg, label_t l) const = 0;

    /** \brief Compute the direct product of two labels.
        \param l1 First label
        \param l2 Second label
        \return Set of labels present in the result
     **/
    virtual label_set_t product(label_t l1, label_t l2) const = 0;

    /** \brief Compute the direct product of a label and a direct sum of
            multiple labels
        \param l1 Label
        \param ls2 Set of labels in the direct sum
        \return Set of labels present in result
     **/
    virtual label_set_t product(label_t l1, const label_set_t &ls2) const = 0;

    /** \brief Compute the direct product of a label and a direct sum of
            multiple labels
        \param ls1 Set of labels in the direct sum
        \param l2 Label
        \return Set of labels present in result
     **/
    virtual label_set_t product(const label_set_t &ls1, label_t l2) const = 0;

    /** \brief Computes the product of two direct sums of labels
        \param ls1 Set of labels in the first sum
        \param ls2 Set of labels in the second sum
        \retun Set of labels present in the result
     **/
    virtual label_set_t product(
            const label_set_t &ls1, const label_set_t &ls2) const = 0;

    /** \brief Does a consistency check on the table.
		\throw exception If product table is not set up properly.
     **/
    virtual void check() const throw(generic_exception);
};

}

#endif // LIBTENSOR_PRODUCT_TABLE_I_H
