#include "point_group_table.h"

#include <iostream>
#include <iomanip>

namespace libtensor {

const char *point_group_table::k_clazz = "point_group_table";

point_group_table::point_group_table(const std::string &id, size_t nirreps) :
	m_nirreps(nirreps), m_id(id),
	m_table(nirreps * (nirreps + 1) / 2, std::vector<label_t>(1, invalid()))
{ }

point_group_table::point_group_table(const point_group_table &pt) :
	m_id(pt.m_id), m_nirreps(pt.m_nirreps), m_table(pt.m_table) {

}

bool point_group_table::is_in_product(const label_group &lg, label_t l) const {

	typedef std::list<label_t> list_t;
	static const char *method = "is_in_product(const label_group &, label_t)";

	if (! is_valid(l))
		throw out_of_bounds(g_ns, k_clazz, method,
				__FILE__, __LINE__, "Irrep invalid.");

	for (size_t i = 0; i < lg.size(); i++)
		if (! is_valid(lg[i]))
			throw out_of_bounds(g_ns, k_clazz, method,
					__FILE__, __LINE__, "Irrep invalid.");

	list_t l1, l2, *ptr1, *ptr2;
	l1.push_back(lg[0]);
	ptr1 = &l1; ptr2 = &l2;

	for (size_t i = 1; i < lg.size(); i++) {
		for (list_t::iterator it = ptr1->begin(); it != ptr1->end(); it++) {
			size_t idx = abs_index(*it, lg[i]);
			const std::vector<label_t> &lr = m_table[idx];

			for (size_t j = 0; j < lr.size(); j++)
				ptr2->push_back(lr[j]);
		}
		ptr1->clear();

		std::swap(ptr1, ptr2);
	}

	for (list_t::iterator it = ptr1->begin(); it != ptr1->end(); it++) {
		if (*it == l) return true;
	}

	return false;
}

void point_group_table::set_product(
		label_t l1, label_t l2, size_t i, label_t lr) throw(out_of_bounds) {

	const char *method = "set_product(label_t, label_t, size_t, label_t)";

	if (! is_valid(l1) || ! is_valid(l2) || ! is_valid(lr))
		throw out_of_bounds(g_ns, k_clazz, method,
				__FILE__, __LINE__, "Label not allowed.");

	size_t idx = abs_index(l1, l2);

	if (i >= m_table[idx].size())
		m_table[idx].resize(i+1, invalid());

	m_table[idx][i] = lr;
}

void point_group_table::check() throw(exception) {

	const char *method = "check()";

	for (size_t i = 0; i < m_table.size(); i++) {
		for (size_t j = 0; j < m_table[i].size(); j++) {
			if (! is_valid(m_table[i][j]))
				throw_exc(k_clazz, method,
						"Invalid label found in product table.");
		}
	}
}

} // namespace libtensor


