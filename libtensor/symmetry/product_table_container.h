#ifndef LIBTENSOR_PRODUCT_TABLE_CONTAINER_H
#define LIBTENSOR_PRODUCT_TABLE_CONTAINER_H

#include <libvmm/libvmm.h>
#include "../exception.h"
#include "product_table_i.h"

namespace libtensor {

/** \brief Container for several product tables

	Singleton object to store and access several product tables.

 **/
class product_table_container :
	public libvmm::singleton<product_table_container> {

	friend class libvmm::singleton<product_table_container>;

public:
	typedef product_table_i::label_t label_t;

private:
	struct container {
		product_table_i* m_pt; //!< Product table
		size_t m_co; //!< Checked out references of m_pt
		bool m_rw; //!< Checked out for reading and writing.

		container() : m_pt(0), m_co(0), m_rw(false) { }
	};

	typedef std::map<std::string, container> list_t;
	typedef std::pair<std::string, container> pair_t;

public:
	static const char *k_clazz; //!< Class name

private:
	list_t m_tables; //!< List of product tables

public:
	/** \brief Destructor
	 **/
	~product_table_container();

	//! \name Manipulators
	//@{
	/** \brief Create a new product table

		\param id Table id
		\param pt Product table to add
		\throw bad_parameter If table with id already exists
	 **/
	void add(const product_table_i &pt) throw(bad_parameter);

	/** \brief Remove product table (if it exists)

		\param id Table id
		\throw bad_parameter If table does not exists.
		\throw exception If table has been checked out for reading or writing.
	 **/
	void erase(const std::string &id) throw(bad_parameter, exception);

	/** \brief Request product table for writing

		\param id Table id
		\return Product table
		\throw bad_parameter If table does not exists.
		\throw exception If table has been checked out for reading or writing.
	 **/
	product_table_i &req_table(
			const std::string &id) throw(bad_parameter, exception);

	//@}


	/** \brief Request product table for reading

		\param id Table id.
		\return Product table
		\throw bad_parameter If table does not exists.
		\throw exception If table has been checked out for writing.
	 **/
	const product_table_i &req_const_table(
			const std::string &id) throw (bad_parameter, exception);

	/** \brief Return checked out product table

		\param id Table id.
		\throw bad_parameter If table does not exists.
	 **/
	void ret_table(const std::string &id) throw(bad_parameter);

	bool table_exists(const std::string &id);

protected:
	product_table_container() { }

private:
	product_table_container(const product_table_container &cont);
	product_table_container &operator=(const product_table_container &cont);

};

}

#endif // LIBTENSOR_PRODUCT_TABLE_CONTAINER_H
