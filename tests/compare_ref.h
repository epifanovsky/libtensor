#ifndef LIBTENSOR_COMPARE_REF_H
#define LIBTENSOR_COMPARE_REF_H

#include <sstream>
#include <libtest.h>
#include <libtensor.h>

namespace libtensor {

template<size_t N>
class compare_ref {
public:
	static void compare(const char *test, tensor_i<N, double> &t,
		tensor_i<N, double> &t_ref, double thresh)
		throw(exception, libtest::test_exception);
	static void compare(const char *test, block_tensor_i<N, double> &t,
		block_tensor_i<N, double> &t_ref, double thresh)
		throw(exception, libtest::test_exception);
};

template<size_t N>
void compare_ref<N>::compare(const char *test, tensor_i<N, double> &t,
	tensor_i<N, double> &t_ref, double thresh)
	throw(exception, libtest::test_exception) {

	tod_compare<N> cmp(t, t_ref, thresh);
	if(!cmp.compare()) {
		std::ostringstream ss1, ss2;
		ss2 << "In " << test << ": ";
		ss2 << "Result does not match reference at element "
			<< cmp.get_diff_index() << ": "
			<< cmp.get_diff_elem_1() << " (act) vs. "
			<< cmp.get_diff_elem_2() << " (ref), "
			<< cmp.get_diff_elem_1() - cmp.get_diff_elem_2()
			<< " (diff)";
		throw libtest::test_exception("compare_ref::compare()",
			__FILE__, __LINE__, ss2.str().c_str());
	}
}

template<size_t N>
void compare_ref<N>::compare(const char *test, block_tensor_i<N, double> &t,
	block_tensor_i<N, double> &t_ref, double thresh)
	throw(exception, libtest::test_exception) {

	btod_compare<N> cmp(t, t_ref, thresh);
	if(!cmp.compare()) {
		std::ostringstream str;
		str << "In " << test << ": ";
		str << "Result does not match reference ";
		if ( ! cmp.get_diff().m_number_of_orbits )
			str << "symmetry";
		else if ( ! cmp.get_diff().m_similar_orbit ) {
			str << "symmetry: Orbits with canonical blocks "
				<< cmp.get_diff().m_canonical_block_index_1 << " (act) vs. "
				<< cmp.get_diff().m_canonical_block_index_2 << " (ref) differ.";
		}
		else if ( cmp.get_diff().m_zero_1 != cmp.get_diff().m_zero_2 ) {
			str << "at zero block " << cmp.get_diff().m_canonical_block_index_1
				<< ".";
		}
		else {
			str << "in block " << cmp.get_diff().m_canonical_block_index_1
				<< " at element "
				<< cmp.get_diff().m_inblock << ": "
				<< cmp.get_diff().m_diff_elem_1 << " (act) vs. "
				<< cmp.get_diff().m_diff_elem_2 << " (ref), "
				<< cmp.get_diff().m_diff_elem_1 - cmp.get_diff().m_diff_elem_2
				<< " (diff)";
		}
		throw libtest::test_exception("compare_ref::compare()",
			__FILE__, __LINE__, str.str().c_str());
	}
}


} // namespace libtensor

#endif // LIBTENSOR_COMPARE_REF_H
