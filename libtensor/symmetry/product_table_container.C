#include "product_table_container.h"

namespace libtensor {

const char *product_table_container::k_clazz = "product_table_container";


product_table_container::~product_table_container() {

	for (list_t::iterator it = m_tables.begin(); it != m_tables.end(); it++) {
		delete it->second.m_pt;
		it->second.m_pt = 0;
	}
}

void product_table_container::add(
		const product_table_i &pt) throw(bad_parameter) {

	static const char *method = "add(product_table_i &)";
	list_t::iterator it = m_tables.find(pt.get_id());
	if (it != m_tables.end())
		throw bad_parameter(g_ns, k_clazz, method,
				__FILE__, __LINE__, "Table already exists.");

	it = m_tables.insert(m_tables.begin(), pair_t(pt.get_id(), container()));

	it->second.m_pt = pt.clone();
}

void product_table_container::erase(
		const std::string &id) throw(bad_parameter, exception) {

	const char *method = "erase(const id_t &)";

	list_t::iterator it = m_tables.find(id);
	if (it == m_tables.end())
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
				"Table does not exist.");

	if (it->second.m_co != 0)
		throw_exc(k_clazz, method, "Table still checked out.");

	delete it->second.m_pt;
	it->second.m_pt = 0;
	m_tables.erase(it);
}

product_table_i &product_table_container::req_table(
		const std::string &id) throw(bad_parameter, exception) {

	const char *method = "req_table(const id_t&)";

	list_t::iterator it = m_tables.find(id);
	if (it == m_tables.end())
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
				"Table does not exist.");

	if (it->second.m_co > 0)
		throw_exc(k_clazz, method, "Table already checked out.");

	it->second.m_rw = true;
	it->second.m_co++;

	return *it->second.m_pt;
}

const product_table_i &product_table_container::req_const_table(
		const std::string &id) throw(bad_parameter, exception) {

	const char *method = "req_table(const id_t&)";

	list_t::iterator it = m_tables.find(id);
	if (it == m_tables.end())
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
				"Table does not exist.");

	if (it->second.m_co > 0 && it->second.m_rw)
		throw_exc(k_clazz, method, "Table already checked out for writing.");

	it->second.m_rw = false;
	it->second.m_co++;

	return *it->second.m_pt;
}

void product_table_container::ret_table(
		const std::string &id) throw(bad_parameter) {

	const char *method = "ret_table(const id_t&)";

	list_t::iterator it = m_tables.find(id);
	if (it == m_tables.end())
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
				"Table does not exist.");

	if (it->second.m_co == 0)
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
				"Returning table which has not been requested.");

	it->second.m_co--;
}

} // namespace libtensor

