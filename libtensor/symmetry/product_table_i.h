#ifndef LIBTENSOR_PRODUCT_TABLE_I_H
#define LIBTENSOR_PRODUCT_TABLE_I_H

#include <vector>

namespace libtensor {


/** \brief Interface for general product tables

	A product table establishes a map l1 x l2 -> l3 + l4 + ...  for a set of
	labels {l1, l2, ...}. The set of labels is represented as unsigned
	integers. The labels can stand for all kinds of objects as long as there
	exists a mapping of a pair of objects onto a set of resulting objects. An
	example of a set of labels would be the irreducible representations of a
	point symmetry group.

	The interface provides functions for se_label and
	product_table_container operate on. Any implementation has to provide these
	to work with both classes.

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
