#include "tod_add.h"

namespace libtensor {

/*
tod_add::tod_add(tensor_i<double> &t, const permutation &p, bool add)
	throw(exception) : m_out(t), m_perm_out(p), m_add(add),
	m_dims_out(t.get_dims()) {

	m_dims_out.permute(m_perm_out);
}

tod_add::~tod_add() {
}
*/

void tod_add::add_op(tensor_i<double> &t, const permutation &p, const double c)
	throw(exception) {

	dimensions d(t.get_dims());
	d.permute(p);
	
}

void tod_add::prefetch() throw(exception) {
}

void tod_add::perform(tensor_i<double> &t) throw(exception) {
}

void tod_add::perform(tensor_i<double> &t, double c) throw(exception) {
}

} // namespace libtensor

