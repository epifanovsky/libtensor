#ifndef LIBTENSOR_COMPARE_REF_H
#define LIBTENSOR_COMPARE_REF_H

#include <sstream>
#include <libtest/test_exception.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/tod/tod_compare.h>
#include <libtensor/btod/btod_compare.h>

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
	static void compare(const char *test, const symmetry<N, double> &s,
		const symmetry<N, double> &s_ref)
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


template<size_t N>
void compare_ref<N>::compare(const char *test, const symmetry<N, double> &s,
	const symmetry<N, double> &s_ref)
	throw(exception, libtest::test_exception) {

	if(!s_ref.get_bis().equals(s.get_bis())) {
		std::ostringstream ss;
		ss << "In " << test << ": Different block index spaces.";
		throw libtest::test_exception("compare_ref::compare()",
			__FILE__, __LINE__, ss.str().c_str());
	}

	orbit_list<N, double> ol(s), ol_ref(s_ref);
	for(typename orbit_list<N, double>::iterator io_ref = ol_ref.begin();
		io_ref != ol_ref.end(); io_ref++) {

		if(!ol.contains(ol_ref.get_abs_index(io_ref))) {
			std::ostringstream ss;
			ss << "In " << test << ": Canonical index "
				<< ol_ref.get_index(io_ref)
				<< " is absent from result.";
			throw libtest::test_exception("compare_ref::compare()",
				__FILE__, __LINE__, ss.str().c_str());
		}
	}
	for(typename orbit_list<N, double>::iterator io = ol.begin();
		io != ol.end(); io++) {

		if(!ol_ref.contains(ol.get_abs_index(io))) {
			std::ostringstream ss;
			ss << "In " << test << ": Canonical index "
				<< ol.get_index(io)
				<< " is absent from reference.";
			throw libtest::test_exception("compare_ref::compare()",
				__FILE__, __LINE__, ss.str().c_str());
		}
	}
}


} // namespace libtensor

#endif // LIBTENSOR_COMPARE_REF_H
