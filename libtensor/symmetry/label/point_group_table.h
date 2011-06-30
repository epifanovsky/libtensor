#ifndef LIBTENSOR_POINT_GROUP_TABLE_H
#define LIBTENSOR_POINT_GROUP_TABLE_H

#include "../../core/out_of_bounds.h"
#include "product_table_i.h"

namespace libtensor {

/** \brief Product tables for products of irreducible representations of a
		point group

	This implementation of product_table_i provides the product table for the
	irreducible representations of an arbitrary point symmetry group. The
	labels represent the different irreducible representations.

	The product table can be set up using the functions add_product() and
	delete_product(). Each product of two labels can have as many result labels
	as there are irreducible representations.

	The function \c is_in_product() computes the product of the label sequence
	given as first argument and compares the result to the label given as
	second argument. If the label is present in the result, true is returned.
	The product is evaluated from the right, i.e.
	\code
	l1 x (l2 x (l3 x ...))
	\endcode

    \ingroup libtensor_symmetry
 **/
class point_group_table : public product_table_i {
public:
	static const char *k_clazz; //!< Class name.
	static const label_t k_invalid; //!< Invalid label

private:
	const std::string m_id; //!< Table id

	size_t m_nirreps; //!< Number of irreducible representations
	std::vector<label_group> m_table; //!< The product table
public:
	//! \name Construction and destruction
	//@{

	/** \brief Constructor
		@param nirreps Number of irreducible representations
	 **/
	point_group_table(const std::string &id, size_t nirreps);

	/** \brief Copy constructor
		@param pt Other point group table
	 **/
	point_group_table(const point_group_table &pt);

	/** \brief Destructor
	 **/
	virtual ~point_group_table() { }

	//@}

	//!	\name Implementation of product_table_i
	//@{

	/** \copydoc product_table_i::clone
	 **/
	virtual point_group_table *clone() const {
		return new point_group_table(*this);
	}

	/** \copydoc product_table_i::get_id
	 **/
	virtual const std::string &get_id() const {
		return m_id;
	}

	/** \copydoc product_table_i::is_valid
	 **/
	virtual bool is_valid(label_t l) const {
		return l < m_nirreps;
	}

	/** \copydoc product_table_i::nlabels
	 **/
	virtual size_t nlabels() const {
		return m_nirreps;
	}

	/** \copydoc product_table_i::invalid
	 **/
	virtual label_t invalid() const {
		return k_invalid;
	}

	/** \copydoc product_table_i::is_in_product
	 **/
	virtual bool is_in_product(const label_group &lg, label_t l) const;

	/** \brief Does a consistency check on the table.
		\throw exception If product table is not set up properly.
	 **/
	virtual void check() const throw(exception);

	//@}

	//!	\name Manipulators
	//@{

	/** \brief Adds product of two irreps.
		\param l1 First label
		\param l2 Second label
		\param lr Result label
		\throw out_of_bounds If l1, l2 or lr are not valid.
	 **/
	void add_product(
			label_t l1, label_t l2, label_t lr) throw (out_of_bounds);

	/** \brief Deletes the product of two irreps.
		\param l1 First label
		\param l2 Second label
		\param lr Result label
		\throw out_of_bounds If l1, l2 or lr are not valid.
	 **/
	void delete_product(
			label_t l1, label_t l2) throw (out_of_bounds);
	//@}

private:
	size_t abs_index(label_t l1, label_t l2) const {
		if (l1 > l2)
			return l1 * (l1 + 1) / 2 + l2;
		else
			return l2 * (l2 + 1) / 2 + l1;
	}
};

}

#endif // LIBTENSOR_POINT_GROUP_TABLE_H
