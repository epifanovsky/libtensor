#ifndef LIBTENSOR_PRODUCT_TABLE_I_H
#define LIBTENSOR_PRODUCT_TABLE_I_H

#include <vector>

namespace libtensor {


/** \brief Interface for general product tables

	This is intended to be used in combination with se_label and
	product_table_container.
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
	virtual const char *get_id() const = 0;

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

};

}

#endif // LIBTENSOR_PRODUCT_TABLE_I_H
