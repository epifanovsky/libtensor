#ifndef LIBTENSOR_POINT_GROUP_TABLE_H
#define LIBTENSOR_POINT_GROUP_TABLE_H

#include "../core/out_of_bounds.h"
#include "product_table_i.h"

namespace libtensor {

/** \brief Product tables for products of irreducible representations of a
		point group

 **/
class point_group_table : public product_table_i {
public:
	static const char *k_clazz; //!< Class name.
	static const char *k_id; //!< Id of the product table

private:
	size_t m_nirreps; //!< Number of irreducible representations
	std::vector< std::vector<label_t> > m_table; //!< The product table

public:
	//! \name Construction and destruction
	//@{

	/** \brief Constructor
		@param nirreps Number of irreducible representations
	 **/
	point_group_table(size_t nirreps);

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
	virtual const char *get_id() const {
		return k_id;
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
	/** \copydoc product_table_i::is_valid
	 **/
	virtual label_t invalid() const {
		return m_nirreps;
	}

	/** \copydoc product_table_i::is_in_product
	 **/
	virtual bool is_in_product(const label_group &lg, label_t l) const;

	//@}

	//!	\name Manipulators
	//@{

	/** \brief Sets the i-th product of two irreps.
		\param l1 First label
		\param l2 Second label
		\param i Which label
		\param lr Result label
		\throw out_of_bounds If l1 or l2 are not valid irreps.
	 **/
	void set_product(label_t l1, label_t l2, size_t i, label_t lr) throw (out_of_bounds);

	/** \brief Does a consistency check on the table.
		\throw exception If product table is not set up properly.
	 **/
	void check() throw(exception);

	//@}

private:
	size_t abs_index(label_t l1, label_t l2) const {
		if (l1 > l2)
			return l2 * m_nirreps + l1;
		else
			return l1 * m_nirreps + l2;
	}
};

}

#endif // LIBTENSOR_POINT_GROUP_TABLE_H
