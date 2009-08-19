#include <sstream>
#include <libtensor.h>
#include "orbit_list_test.h"

namespace libtensor {

void orbit_list_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();
	test_4();
	test_5();
	test_6();
	test_7();
	test_8();
}

void orbit_list_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "orbit_list_test::test_1()";
	try {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	mask<2> msk;
	msk[0] = true; msk[1] = true;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	symmetry<2, double> sym(bis);

	orbit_list<2, double> orblst(sym);
	size_t norb_ref = 9;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	index<2> io;
	do {
		bool can = false, can_ref = true;
		orbit_list<2, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(*i)) {
				can = true;
				break;
			}
			i++;
		}
		if(can != can_ref) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can_ref = " << can_ref << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "orbit_list_test::test_2()";
	try {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	mask<2> msk;
	msk[0] = true; msk[1] = true;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	symmetry<2, double> sym(bis);
	mask<2> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true;
	symel_cycleperm<2, double> cycle(cycle_msk, dims);
	sym.add_element(cycle);

	orbit_list<2, double> orblst(sym);
	size_t norb_ref = 6;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	index<2> io;
	do {
		bool can_ref = io[0] <= io[1];
		bool can = false;
		orbit_list<2, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(*i)) {
				can = true;
				break;
			}
			i++;
		}
		if(can != can_ref) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can_ref = " << can_ref << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "orbit_list_test::test_3()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	mask<4> msk;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	symmetry<4, double> sym(bis);
	mask<4> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true;
	symel_cycleperm<4, double> cycle(cycle_msk, dims);
	sym.add_element(cycle);

	orbit_list<4, double> orblst(sym);
	size_t norb_ref = 54;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	index<4> io;
	do {
		bool can_ref = io[0] <= io[1];
		bool can = false;
		orbit_list<4, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(*i)) {
				can = true;
				break;
			}
			i++;
		}
		if(can != can_ref) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can_ref = " << can_ref << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_4() throw(libtest::test_exception) {

	static const char *testname = "orbit_list_test::test_4()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	mask<4> msk;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	symmetry<4, double> sym(bis);
	mask<4> cycle_msk;
	cycle_msk[1] = true; cycle_msk[2] = true;
	symel_cycleperm<4, double> cycle(cycle_msk, dims);
	sym.add_element(cycle);

	orbit_list<4, double> orblst(sym);
	size_t norb_ref = 54;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	index<4> io;
	do {
		bool can_ref = io[1] <= io[2];
		bool can = false;
		orbit_list<4, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(*i)) {
				can = true;
				break;
			}
			i++;
		}
		if(can != can_ref) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can_ref = " << can_ref << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_5() throw(libtest::test_exception) {

	static const char *testname = "orbit_list_test::test_5()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	mask<4> msk;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	symmetry<4, double> sym(bis);
	mask<4> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true; cycle_msk[2] = true;
	symel_cycleperm<4, double> cycle(cycle_msk, dims);
	sym.add_element(cycle);

	orbit_list<4, double> orblst(sym);
	size_t norb_ref = 33;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	index<4> io;
	do {
		bool can_ref = (io[0] == io[1] && io[0] <= io[2]) ||
			(io[0] < io[1] && io[0] < io[2]);
		bool can = false;
		orbit_list<4, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(*i)) {
				can = true;
				break;
			}
			i++;
		}
		if(can != can_ref) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can_ref = " << can_ref << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_6() throw(libtest::test_exception) {

	static const char *testname = "orbit_list_test::test_6()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	mask<4> msk;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	symmetry<4, double> sym(bis);
	mask<4> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true;
	cycle_msk[2] = false; cycle_msk[3] = false;
	symel_cycleperm<4, double> cycle1(cycle_msk, dims);
	cycle_msk[0] = false; cycle_msk[1] = false;
	cycle_msk[2] = true; cycle_msk[3] = true;
	symel_cycleperm<4, double> cycle2(cycle_msk, dims);
	sym.add_element(cycle1);
	sym.add_element(cycle2);

	orbit_list<4, double> orblst(sym);
	size_t norb_ref = 36;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	index<4> io;
	do {
		bool can_ref = (io[0] <= io[1] && io[2] <= io[3]);
		bool can = false;
		orbit_list<4, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(*i)) {
				can = true;
				break;
			}
			i++;
		}
		if(can != can_ref) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can_ref = " << can_ref << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_7() throw(libtest::test_exception) {

	static const char *testname = "orbit_list_test::test_7()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	mask<4> msk;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	symmetry<4, double> sym(bis);
	mask<4> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true; cycle_msk[2] = true;
	symel_cycleperm<4, double> cycle1(cycle_msk, dims);
	cycle_msk[2] = false;
	symel_cycleperm<4, double> cycle2(cycle_msk, dims);
	sym.add_element(cycle1);
	sym.add_element(cycle2);

	orbit_list<4, double> orblst(sym);
	size_t norb_ref = 30;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	index<4> io;
	do {
		bool can_ref = (io[0] <= io[1] && io[1] <= io[2]);
		bool can = false;
		orbit_list<4, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(*i)) {
				can = true;
				break;
			}
			i++;
		}
		if(can != can_ref) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can_ref = " << can_ref << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_8() throw(libtest::test_exception) {

	static const char *testname = "orbit_list_test::test_8()";
	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
	mask<4> msk;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	symmetry<4, double> sym(bis);
	mask<4> cycle_msk;
	cycle_msk[0] = true; cycle_msk[1] = true;
	cycle_msk[2] = true; cycle_msk[3] = true;
	symel_cycleperm<4, double> cycle1(cycle_msk, dims);
	cycle_msk[2] = false; cycle_msk[3] = false;
	symel_cycleperm<4, double> cycle2(cycle_msk, dims);
	sym.add_element(cycle1);
	sym.add_element(cycle2);

	orbit_list<4, double> orblst(sym);
	size_t norb_ref = 15;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	index<4> io;
	do {
		bool can_ref =
			(io[0] <= io[1] && io[1] <= io[2] && io[2] <= io[3]);
		bool can = false;
		orbit_list<4, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(*i)) {
				can = true;
				break;
			}
			i++;
		}
		if(can != can_ref) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< " (can_ref = " << can_ref << ").";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
