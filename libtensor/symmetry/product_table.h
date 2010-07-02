#ifndef LIBTENSOR_PRODUCT_TABLE_H
#define LIBTENSOR_PRODUCT_TABLE_H

namespace libtensor {

/** \brief General storage for product tables

	Stores a product table for a group of labels. The group is expected to be
	commutative. This is intended to be used in combination with se_label and
	product_table_container.

 **/
class product_table {
	friend class product_table_container;

public:
	typedef unsigned int label_t;

	const label_t unassigned; //!< Unassigned label

private:
	size_t m_size; //!< Size of the product table
	std::vector< std::vector<label_t> > m_table; //!< The product table

public:
	//! \name Access routines
	//@{
	/** \brief Returns the number of result labels in the product

		\param l1 First label
		\param l2 Second label
		\return Number of result labels
		\throw out_of_bounds If label is not allowed.
	 **/
	size_t product_size(label_t l1, label_t l2) const throw(out_of_bounds);

	/** \brief Returns the i-th label in the product

		\param l1 First label
		\param l2 Second label
		\param i Number of the result label
		\return Label
		\throw out_of_bounds If label is not allowed, or i exceeds number of
			labels in the product
	 **/
	label_t product(label_t l1, label_t l2,
			size_t i) const throw(out_of_bounds);

	//@}

	//! \name Manipulators
	//@{
	/** \brief Sets the i-th result label of the product

		\param l1 First input label
		\param l2 Second input label
		\param i  Number of the result label
		\param r  Result label
		\throw out_of_bounds If one of the labels is not allowed, or i exceeds
			number of labels in the product
	 **/
	void set_product(label_t l1, label_t l2,
			size_t i, label_t r) throw(out_of_bounds)

	//@}

private:
	product_table(size_t nlabels) : m_size(nlabels), unassigned(nlabels),
		m_table(nlabels * (nlabels - 1) / 2) {

	}

	product_table(const product_table &);
	product_table &operator=(const product_table &);
};

}

#endif // LIBTENSOR_PRODUCT_TABLE_H
