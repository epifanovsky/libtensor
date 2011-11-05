#ifndef LIBTENSOR_PRODUCT_TABLE_I_H
#define LIBTENSOR_PRODUCT_TABLE_I_H

#include <vector>

namespace libtensor {


/** \brief Interface for general product tables

	A product table establishes a map
	\code
	l1 x l2 -> l3 + l4 + ...
	\endcode
	for a set of labels \code {l1, l2, ...} \endcode . Each label is
	represented as unsigned integers. Labels can represent any information for
	which a mapping of products of label pairs onto direct sums of labels
	are required. A typical example of labels would be the irreducible
	representations of a point symmetry group.

	This interface specifies the functions which se_label and
	product_table_container use to operate. Any specific implementation of
	product_table_i needs to provide these functions to work with both
	classes.

    \ingroup libtensor_symmetry
 **/
class product_table_i {
public:
    typedef unsigned int label_t;
    typedef std::vector<label_t> label_group;

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

    /** \brief Checks whether a label is valid.
     **/
    virtual bool is_valid(label_t l) const = 0;

    /** \brief Returns the number of valid labels
     **/
    virtual size_t nlabels() const = 0;

    /** \brief Returns an invalid label
     **/
    virtual label_t invalid() const = 0;

    /** \brief Determines if the label is in the product.

		\param lg Group of labels to take the product of.
		\param l Label to check against.
		\return True if label is in the product, else false.
     **/
    virtual bool is_in_product(const label_group &lg, label_t l) const = 0;

    /** \brief Does a consistency check on the table.
		\throw exception If product table is not set up properly.
     **/
    virtual void check() const throw(exception) = 0;
};

}

#endif // LIBTENSOR_PRODUCT_TABLE_I_H
