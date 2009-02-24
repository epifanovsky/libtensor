#include "lehmer_code.h"
#include "symmetry.h"

namespace libtensor {

symmetry::symmetry(const symmetry_i &s, const dimensions &d) : m_dims(d) {
	syminfo si;
	index i(d.get_order());
	do {
		if(!s.is_unique(i)) {
			size_t iabs = d.abs_index(i);
			si.unique = d.abs_index(s.get_unique(i));
			si.perm = lehmer_code::get_instance().
				perm2code(s.get_perm(i));
			si.coeff = s.get_coeff(i);
			m_sym[iabs] = si;
		}
	} while(d.inc_index(i));
}

symmetry::~symmetry() {
}

bool symmetry::is_unique(const index &i) const throw(exception) {
	return m_sym.find(m_dims.abs_index(i)) == m_sym.end();
}

/**	\todo Implement symmetry::get_unique
**/
const index &symmetry::get_unique(const index &i) const throw(exception) {
	typename symmap::const_iterator iter = m_sym.find(m_dims.abs_index(i));
	if(iter == m_sym.end()) return i;

	return i;
}

const permutation &symmetry::get_perm(const index &i) const throw(exception) {
	typename symmap::const_iterator iter = m_sym.find(m_dims.abs_index(i));
	if(iter == m_sym.end()) {
		return lehmer_code::get_instance().code2perm(i.get_order(), 0);
	} else {
		return lehmer_code::get_instance().code2perm(
			i.get_order(), iter->second.perm);
	}
}

double symmetry::get_coeff(const index &i) const throw(exception) {
	typename symmap::const_iterator iter = m_sym.find(m_dims.abs_index(i));
	if(iter == m_sym.end()) return 1.0;
	else return iter->second.coeff;
}

} // namespace libtensor

