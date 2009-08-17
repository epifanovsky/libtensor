#include <sstream>
#include <libtensor.h>
#include "orbit_test.h"

namespace libtensor {

void orbit_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
	test_7();
	test_8();
}

void orbit_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "orbit_test::test_1()";
	try {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	dimensions<2> dims(index_range<2>(i1, i2));
	symmetry<2, double> sym(dims);

	index<2> io;
	do {
		orbit<2, double> orb(sym, io);
		if(orb.get_abs_canonical_index() != dims.abs_index(io)) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		const transf<2, double> &tr = orb.get_transf(io);
		permutation<2> pref;
		if(!tr.m_perm.equals(pref)) {
			std::ostringstream ss;
			ss << "Incorrect block permutation for " << io
				<< ": " << tr.m_perm << " vs. " << pref
				<< " (ref).";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
		if(tr.m_coeff != 1.0) {
			fail_test(testname, __FILE__, __LINE__,
				"Incorrect block transformation (coeff).");
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "orbit_test::test_2()";
	try {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	dimensions<2> dims(index_range<2>(i1, i2));
	symmetry<2, double> sym(dims);
	mask<2> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true;
	symel_cycleperm<2, double> cycle(cycle_msk, dims);
	sym.add_element(cycle);

	index<2> io;
	do {
		orbit<2, double> orb(sym, io);
		bool can = io[0] <= io[1];
		size_t abscanidx = orb.get_abs_canonical_index();
		if((can && abscanidx != dims.abs_index(io)) ||
			(!can && abscanidx == dims.abs_index(io))) {

			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		const transf<2, double> &tr = orb.get_transf(io);
		if(can) {
			if(!tr.m_perm.is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (1).");
			}
		} else {
			permutation<2> pref; pref.permute(0, 1);
			index<2> io2(io);
			io2.permute(pref);
			if(abscanidx != dims.abs_index(io2)) {
				fail_test(testname, __FILE__, __LINE__,
					"Inconsistent orbit composition (2).");
			}
			if(!tr.m_perm.equals(pref)) {
				std::ostringstream ss;
				ss << "Incorrect block permutation for " << io
					<< ": " << tr.m_perm << " vs. " << pref
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (2).");
			}
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "orbit_test::test_3()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	symmetry<4, double> sym(dims);
	mask<4> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true;
	symel_cycleperm<4, double> cycle(cycle_msk, dims);
	sym.add_element(cycle);

	index<4> io;
	do {
		orbit<4, double> orb(sym, io);
		bool can = io[0] <= io[1];
		size_t abscanidx = orb.get_abs_canonical_index();
		if((can && abscanidx != dims.abs_index(io)) ||
			(!can && abscanidx == dims.abs_index(io))) {

			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		const transf<4, double> &tr = orb.get_transf(io);
		if(can) {
			if(!tr.m_perm.is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (1).");
			}
		} else {
			permutation<4> pref; pref.permute(0, 1);
			index<4> io2(io);
			io2.permute(pref);
			if(abscanidx != dims.abs_index(io2)) {
				fail_test(testname, __FILE__, __LINE__,
					"Inconsistent orbit composition (2).");
			}
			if(!tr.m_perm.equals(pref)) {
				std::ostringstream ss;
				ss << "Incorrect block permutation for " << io
					<< ": " << tr.m_perm << " vs. " << pref
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (2).");
			}
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "orbit_test::test_4()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	symmetry<4, double> sym(dims);
	mask<4> cycle_msk;
	cycle_msk[1] = true; cycle_msk[2] = true;
	symel_cycleperm<4, double> cycle(cycle_msk, dims);
	sym.add_element(cycle);

	index<4> io;
	do {
		orbit<4, double> orb(sym, io);
		bool can = io[1] <= io[2];
		size_t abscanidx = orb.get_abs_canonical_index();
		if((can && abscanidx != dims.abs_index(io)) ||
			(!can && abscanidx == dims.abs_index(io))) {

			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		const transf<4, double> &tr = orb.get_transf(io);
		if(can) {
			if(!tr.m_perm.is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (1).");
			}
		} else {
			permutation<4> pref; pref.permute(1, 2);
			index<4> io2(io);
			io2.permute(pref);
			if(abscanidx != dims.abs_index(io2)) {
				fail_test(testname, __FILE__, __LINE__,
					"Inconsistent orbit composition (2).");
			}
			if(!tr.m_perm.equals(pref)) {
				std::ostringstream ss;
				ss << "Incorrect block permutation for " << io
					<< ": " << tr.m_perm << " vs. " << pref
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (2).");
			}
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "orbit_test::test_5()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	symmetry<4, double> sym(dims);
	mask<4> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true; cycle_msk[2] = true;
	symel_cycleperm<4, double> cycle(cycle_msk, dims);
	sym.add_element(cycle);

	index<4> io;
	do {
		orbit<4, double> orb(sym, io);
		bool can = (io[0] == io[1] && io[0] <= io[2]) ||
			(io[0] < io[1] && io[0] < io[2]);
		size_t abscanidx = orb.get_abs_canonical_index();
		if((can && abscanidx != dims.abs_index(io)) ||
			(!can && abscanidx == dims.abs_index(io))) {

			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		const transf<4, double> &tr = orb.get_transf(io);
		if(can) {
			if(!tr.m_perm.is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (1).");
			}
		} else {
			permutation<4> p2, p3;
			p2.permute(0, 1); p2.permute(1, 2);
			index<4> io2(io);
			while(!(io2[0] == io2[1] && io2[0] <= io2[2]) &&
				!(io2[0] < io2[1] && io2[0] < io2[2])) {
				io2.permute(p2);
				p3.permute(p2);
			}
			p3.invert();
			if(abscanidx != dims.abs_index(io2)) {
				std::ostringstream ss;
				ss << "Unexpected canonical index for " << io
					<< ": " << abscanidx << " vs. " << io2
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(!tr.m_perm.equals(p3)) {
				std::ostringstream ss;
				ss << "Incorrect block permutation for " << io
					<< ": " << tr.m_perm << " vs. " << p3
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (2).");
			}
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "orbit_test::test_6()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	symmetry<4, double> sym(dims);
	mask<4> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true;
	cycle_msk[2] = false; cycle_msk[3] = false;
	symel_cycleperm<4, double> cycle1(cycle_msk, dims);
	cycle_msk[0] = false; cycle_msk[1] = false;
	cycle_msk[2] = true; cycle_msk[3] = true;
	symel_cycleperm<4, double> cycle2(cycle_msk, dims);
	sym.add_element(cycle1);
	sym.add_element(cycle2);

	index<4> io;
	do {
		orbit<4, double> orb(sym, io);
		bool can = (io[0] <= io[1] && io[2] <= io[3]);
		size_t abscanidx = orb.get_abs_canonical_index();
		if((can && abscanidx != dims.abs_index(io)) ||
			(!can && abscanidx == dims.abs_index(io))) {

			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		const transf<4, double> &tr = orb.get_transf(io);
		if(can) {
			if(!tr.m_perm.is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (1).");
			}
		} else {
			permutation<4> p1, p2, pref;
			p1.permute(0, 1);
			p2.permute(2, 3);
			index<4> io2(io);
			if(io2[0] > io2[1]) {
				io2.permute(p1);
				pref.permute(p1);
			}
			if(io2[2] > io2[3]) {
				io2.permute(p2);
				pref.permute(p2);
			}
			pref.invert();
			if(abscanidx != dims.abs_index(io2)) {
				std::ostringstream ss;
				ss << "Unexpected canonical index for " << io
					<< ": " << abscanidx << " vs. " << io2
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(!tr.m_perm.equals(pref)) {
				std::ostringstream ss;
				ss << "Incorrect block permutation for " << io
					<< ": " << tr.m_perm << " vs. " << pref
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (2).");
			}
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_test::test_7() throw(libtest::test_exception) {

	static const char *testname = "orbit_test::test_7()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	symmetry<4, double> sym(dims);
	mask<4> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true; cycle_msk[2] = true;
	symel_cycleperm<4, double> cycle1(cycle_msk, dims);
	cycle_msk[2] = false;
	symel_cycleperm<4, double> cycle2(cycle_msk, dims);
	sym.add_element(cycle1);
	sym.add_element(cycle2);

	index<4> io;
	do {
		orbit<4, double> orb(sym, io);
		bool can = (io[0] <= io[1] && io[1] <= io[2]);
		size_t abscanidx = orb.get_abs_canonical_index();
		index<4> canidx;
		dims.abs_index(abscanidx, canidx);
		if((can && abscanidx != dims.abs_index(io)) ||
			(!can && abscanidx == dims.abs_index(io))) {

			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		const transf<4, double> &tr = orb.get_transf(io);
		if(can) {
			if(!tr.m_perm.is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (1).");
			}
		} else {
			permutation<4> p1, p2;
			p1.permute(0, 1);
			p2.permute(1, 2);
			index<4> io2(io);
			if(io2[0] > io2[1]) io2.permute(p1);
			if(io2[1] > io2[2]) io2.permute(p2);
			if(io2[0] > io2[1]) io2.permute(p1);
			index<4> io3(io2);
			io3.permute(tr.m_perm);
			if(abscanidx != dims.abs_index(io2)) {
				std::ostringstream ss;
				ss << "Unexpected canonical index for " << io
					<< ": " << canidx << " vs. " << io2
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(!io3.equals(io)) {
				std::ostringstream ss;
				ss << "Incorrect block permutation for " << io3
					<< "->" << io << ": " << tr.m_perm
					<< ".";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (2).");
			}
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_test::test_8() throw(libtest::test_exception) {

	static const char *testname = "orbit_test::test_8()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	symmetry<4, double> sym(dims);
	mask<4> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true;
	cycle_msk[2] = true; cycle_msk[3] = true;
	symel_cycleperm<4, double> cycle1(cycle_msk, dims);
	cycle_msk[2] = false; cycle_msk[3] = false;
	symel_cycleperm<4, double> cycle2(cycle_msk, dims);
	sym.add_element(cycle1);
	sym.add_element(cycle2);

	index<4> io;
	do {
		orbit<4, double> orb(sym, io);
		bool can = (io[0] <= io[1] && io[1] <= io[2] && io[2] <= io[3]);
		size_t abscanidx = orb.get_abs_canonical_index();
		index<4> canidx;
		dims.abs_index(abscanidx, canidx);
		if((can && abscanidx != dims.abs_index(io)) ||
			(!can && abscanidx == dims.abs_index(io))) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		const transf<4, double> &tr = orb.get_transf(io);
		if(can) {
			if(!tr.m_perm.is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (1).");
			}
		} else {
			permutation<4> p1, p2, p3;
			p1.permute(0, 1);
			p2.permute(1, 2);
			p3.permute(2, 3);
			index<4> io2(io);
			if(io2[0] > io2[1]) io2.permute(p1);
			if(io2[1] > io2[2]) io2.permute(p2);
			if(io2[2] > io2[3]) io2.permute(p3);
			if(io2[0] > io2[1]) io2.permute(p1);
			if(io2[1] > io2[2]) io2.permute(p2);
			if(io2[0] > io2[1]) io2.permute(p1);
			index<4> io3(io2);
			io3.permute(tr.m_perm);
			if(abscanidx != dims.abs_index(io2)) {
				std::ostringstream ss;
				ss << "Unexpected canonical index for " << io
					<< ": " << canidx << " vs. " << io2
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(!io3.equals(io)) {
				std::ostringstream ss;
				ss << "Incorrect block permutation for " << io3
					<< "->" << io << ": " << tr.m_perm
					<< ".";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (2).");
			}
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
