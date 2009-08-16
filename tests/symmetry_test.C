#include <sstream>
#include <libtensor.h>
#include "symmetry_test.h"

namespace libtensor {

void symmetry_test::perform() throw(libtest::test_exception) {

	test_orbits_1();
	test_orbits_2();
	test_orbits_3();
	test_orbits_4();
	test_orbits_5();
	test_orbits_6();
	test_orbits_7();
	test_orbits_8();
}

void symmetry_test::test_orbits_1() throw(libtest::test_exception) {

	static const char *testname = "symmetry_test::test_orbits_1()";
	try {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	dimensions<2> dims(index_range<2>(i1, i2));
	symmetry<2, double> sym(dims);

	if(sym.get_num_orbits() != 9) {
		fail_test(testname, __FILE__, __LINE__,
			"Invalid number of orbits.");
	}

	index<2> io;
	do {
		if(!sym.is_canonical(io)) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		index<2> io_can;
		transf<2, double> tr;
		sym.get_transf(io, io_can, tr);
		if(!io.equals(io_can)) {
			fail_test(testname, __FILE__, __LINE__,
				"Inconsistent orbit composition.");
		}
		if(!tr.m_perm.is_identity()) {
			fail_test(testname, __FILE__, __LINE__,
				"Incorrect block transformation (perm).");
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

void symmetry_test::test_orbits_2() throw(libtest::test_exception) {

	static const char *testname = "symmetry_test::test_orbits_2()";
	try {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	dimensions<2> dims(index_range<2>(i1, i2));
	symmetry<2, double> sym(dims);
	mask<2> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true;
	symel_cycleperm<2, double> cycle(cycle_msk, dims);
	sym.add_element(cycle);

	if(sym.get_num_orbits() != 6) {
		fail_test(testname, __FILE__, __LINE__,
			"Invalid number of orbits.");
	}

	index<2> io;
	do {
		bool can = io[0] <= io[1];
		if(sym.is_canonical(io) != can) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		index<2> io_can;
		transf<2, double> tr;
		sym.get_transf(io, io_can, tr);
		if(can) {
			if(!io.equals(io_can)) {
				fail_test(testname, __FILE__, __LINE__,
					"Inconsistent orbit composition (1).");
			}
			if(!tr.m_perm.is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (1).");
			}
		} else {
			permutation<2> p2; p2.permute(0, 1);
			index<2> io2(io);
			io2.permute(p2);
			if(!io2.equals(io_can)) {
				fail_test(testname, __FILE__, __LINE__,
					"Inconsistent orbit composition (2).");
			}
			if(!tr.m_perm.equals(p2)) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (2).");
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

void symmetry_test::test_orbits_3() throw(libtest::test_exception) {

	static const char *testname = "symmetry_test::test_orbits_3()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	symmetry<4, double> sym(dims);
	mask<4> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true;
	symel_cycleperm<4, double> cycle(cycle_msk, dims);
	sym.add_element(cycle);

	if(sym.get_num_orbits() != 54) {
		fail_test(testname, __FILE__, __LINE__,
			"Invalid number of orbits.");
	}

	index<4> io;
	do {
		bool can = io[0] <= io[1];
		if(sym.is_canonical(io) != can) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		index<4> io_can;
		transf<4, double> tr;
		sym.get_transf(io, io_can, tr);
		if(can) {
			if(!io.equals(io_can)) {
				fail_test(testname, __FILE__, __LINE__,
					"Inconsistent orbit composition (1).");
			}
			if(!tr.m_perm.is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (1).");
			}
		} else {
			permutation<4> p2; p2.permute(0, 1);
			index<4> io2(io);
			io2.permute(p2);
			if(!io2.equals(io_can)) {
				fail_test(testname, __FILE__, __LINE__,
					"Inconsistent orbit composition (2).");
			}
			if(!tr.m_perm.equals(p2)) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (2).");
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

void symmetry_test::test_orbits_4() throw(libtest::test_exception) {

	static const char *testname = "symmetry_test::test_orbits_4()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	symmetry<4, double> sym(dims);
	mask<4> cycle_msk;
	cycle_msk[1] = true; cycle_msk[2] = true;
	symel_cycleperm<4, double> cycle(cycle_msk, dims);
	sym.add_element(cycle);

	if(sym.get_num_orbits() != 54) {
		fail_test(testname, __FILE__, __LINE__,
			"Invalid number of orbits.");
	}

	index<4> io;
	do {
		bool can = io[1] <= io[2];
		if(sym.is_canonical(io) != can) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		index<4> io_can;
		transf<4, double> tr;
		sym.get_transf(io, io_can, tr);
		if(can) {
			if(!io.equals(io_can)) {
				fail_test(testname, __FILE__, __LINE__,
					"Inconsistent orbit composition (1).");
			}
			if(!tr.m_perm.is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (1).");
			}
		} else {
			permutation<4> p2; p2.permute(1, 2);
			index<4> io2(io);
			io2.permute(p2);
			if(!io2.equals(io_can)) {
				fail_test(testname, __FILE__, __LINE__,
					"Inconsistent orbit composition (2).");
			}
			if(!tr.m_perm.equals(p2)) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (2).");
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

void symmetry_test::test_orbits_5() throw(libtest::test_exception) {

	static const char *testname = "symmetry_test::test_orbits_5()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	dimensions<4> dims(index_range<4>(i1, i2));
	symmetry<4, double> sym(dims);
	mask<4> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true; cycle_msk[2] = true;
	symel_cycleperm<4, double> cycle(cycle_msk, dims);
	sym.add_element(cycle);

	if(sym.get_num_orbits() != 33) {
		fail_test(testname, __FILE__, __LINE__,
			"Invalid number of orbits.");
	}

	index<4> io;
	do {
		bool can = (io[0] == io[1] && io[0] <= io[2]) ||
			(io[0] < io[1] && io[0] < io[2]);
		if(sym.is_canonical(io) != can) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		index<4> io_can;
		transf<4, double> tr;
		sym.get_transf(io, io_can, tr);
		if(can) {
			if(!io.equals(io_can)) {
				fail_test(testname, __FILE__, __LINE__,
					"Inconsistent orbit composition (1).");
			}
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
			if(!io2.equals(io_can)) {
				std::ostringstream ss;
				ss << "Unexpected canonical index for " << io
					<< ": " << io_can << " vs. " << io2
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

void symmetry_test::test_orbits_6() throw(libtest::test_exception) {

	static const char *testname = "symmetry_test::test_orbits_6()";
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

	if(sym.get_num_orbits() != 36) {
		fail_test(testname, __FILE__, __LINE__,
			"Invalid number of orbits.");
	}

	index<4> io;
	do {
		bool can = (io[0] <= io[1] && io[2] <= io[3]);
		if(sym.is_canonical(io) != can) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		index<4> io_can;
		transf<4, double> tr;
		sym.get_transf(io, io_can, tr);
		if(can) {
			if(!io.equals(io_can)) {
				fail_test(testname, __FILE__, __LINE__,
					"Inconsistent orbit composition (1).");
			}
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
			if(!io2.equals(io_can)) {
				std::ostringstream ss;
				ss << "Unexpected canonical index for " << io
					<< ": " << io_can << " vs. " << io2
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

void symmetry_test::test_orbits_7() throw(libtest::test_exception) {

	static const char *testname = "symmetry_test::test_orbits_7()";
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

	if(sym.get_num_orbits() != 30) {
		fail_test(testname, __FILE__, __LINE__,
			"Invalid number of orbits.");
	}

	index<4> io;
	do {
		bool can = (io[0] <= io[1] && io[1] <= io[2]);
		if(sym.is_canonical(io) != can) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		index<4> io_can;
		transf<4, double> tr;
		sym.get_transf(io, io_can, tr);
		if(can) {
			if(!io.equals(io_can)) {
				fail_test(testname, __FILE__, __LINE__,
					"Inconsistent orbit composition (1).");
			}
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
			p2.permute(1, 2);
			index<4> io2(io);
			if(io2[1] > io2[2]) {
				io2.permute(p2);
				pref.permute(p2);
			}
			if(io2[0] > io2[1]) {
				io2.permute(p1);
				pref.permute(p1);
			}
			pref.invert();
			if(!io2.equals(io_can)) {
				std::ostringstream ss;
				ss << "Unexpected canonical index for " << io
					<< ": " << io_can << " vs. " << io2
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

void symmetry_test::test_orbits_8() throw(libtest::test_exception) {

	static const char *testname = "symmetry_test::test_orbits_8()";
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

	if(sym.get_num_orbits() != 15) {
		fail_test(testname, __FILE__, __LINE__,
			"Invalid number of orbits.");
	}

	index<4> io;
	do {
		bool can = (io[0] <= io[1] && io[1] <= io[2] && io[2] <= io[3]);
		if(sym.is_canonical(io) != can) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can = " << can << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		index<4> io_can;
		transf<4, double> tr;
		sym.get_transf(io, io_can, tr);
		if(can) {
			if(!io.equals(io_can)) {
				fail_test(testname, __FILE__, __LINE__,
					"Inconsistent orbit composition (1).");
			}
			if(!tr.m_perm.is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.m_coeff != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (1).");
			}
		} else {
			permutation<4> p1, p2, p3, pref;
			p1.permute(0, 1);
			p2.permute(1, 2);
			p3.permute(2, 3);
			index<4> io2(io);
			if(io2[2] > io2[3]) {
				io2.permute(p3);
				pref.permute(p3);
			}
			if(io2[1] > io2[2]) {
				io2.permute(p2);
				pref.permute(p2);
			}
			if(io2[0] > io2[1]) {
				io2.permute(p1);
				pref.permute(p1);
			}
			if(io2[2] > io2[3]) {
				io2.permute(p3);
				pref.permute(p3);
			}
			if(io2[1] > io2[2]) {
				io2.permute(p2);
				pref.permute(p2);
			}
			pref.invert();
			if(!io2.equals(io_can)) {
				std::ostringstream ss;
				ss << "Unexpected canonical index for " << io
					<< ": " << io_can << " vs. " << io2
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

} // namespace libtensor
