#ifndef LIBTENSOR_PRODUCT_TABLE_CONTAINER_H
#define LIBTENSOR_PRODUCT_TABLE_CONTAINER_H

#include <libvmm/libvmm.h>
#include "product_table.h"

namespace libtensor {

/** \brief Container for several product tables

	Singleton object to store and access several product tables.

 **/
class product_table_container :
	public libvmm::singleton<product_table_container> {

	friend class libvmm::singleton<product_table_container>;

public:
	typedef product_table::label_t label_t;
	typedef std::string id_t;

	static const id_t k_point_group; //!< Product table id for point groups

private:
	struct container {
		product_table m_table;
		size_t m_checked_out;
		size_t m_checked_out_r;

		container(size_t nlabels) :
			m_table(nlabels), m_checked_out(0), m_checked_out_r(0) { }
	};

	std::map<id_t, container> m_tables; //!< List of product tables

public:
	//! \name Manipulators
	//@{
	/** \brief Create a new product table

		\param id Table id
		\param nlabels Number of labels in the new product table
		\return Newly created product table
		\throw bad_parameter If table with id already exists
	 **/
	void create(id_t id, size_t nlabels) throw(bad_parameter);

	/** \brief Remove product table (if it exists)

		\param id Table id
		\throw bad_parameter If table does not exists.
		\throw exception If table has been checked out for reading or writing.
	 **/
	void erase(id_t id) throw(bad_parameter, exception);

	//@}

	/** \brief Request product table for writing

		\param id Table id
		\return Product table
		\throw bad_parameter If table does not exists.
		\throw exception If table has been checked out for reading or writing.
	 **/
	product_table &req_table(id_t id) throw(bad_parameter, exception);


	/** \brief Request product table for reading

		\param id Table id.
		\return Product table
		\throw bad_parameter If table does not exists.
		\throw exception If table has been checked out for writing.
	 **/
	const product_table &req_const_table(id_t id) throw (bad_parameter,
			exception);

	/** \brief Return checked out product table

		\param id Table id.
		\throw bad_parameter If table does not exists.
	 **/
	void ret_table(id_t id) throw(bad_parameter);


protected:
	product_table_container() { }

private:
	product_table_container(const product_table_container &cont);
	product_table_container &operator=(const product_table_container &cont);

};

}

#endif // LIBTENSOR_PRODUCT_TABLE_CONTAINER_H
