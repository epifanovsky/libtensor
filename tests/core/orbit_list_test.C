#include <sstream>
#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/symmetry/se_perm.h>
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
	test_9();
}

void orbit_list_test::test_1() throw(libtest::test_exception) {

	//
	//	dim [3,3], split [3,3]
	//	(1)(2), 9 orbits
	//

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

	abs_index<2> aio(dims);
	do {
	    const index<2> &io = aio.get_index();
		bool can = false, can_ref = true;
		orbit_list<2, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(orblst.get_index(i))) {
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
	} while(aio.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_2() throw(libtest::test_exception) {

	//
	//	dim [3,3], split [3,3]
	//	(1)(1), 6 orbits
	//

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
	permutation<2> perm; perm.permute(0, 1);
	se_perm<2, double> cycle(perm, 1.0);
	sym.insert(cycle);

	orbit_list<2, double> orblst(sym);
	size_t norb_ref = 6;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	abs_index<2> aio(dims);
	do {
	    const index<2> &io = aio.get_index();
		bool can_ref = io[0] <= io[1];
		bool can = false;
		orbit_list<2, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(orblst.get_index(i))) {
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
	} while(aio.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_3() throw(libtest::test_exception) {

	//
	//	dim [3,3,3,3], split [3,3,3,3]
	//	(1)(1)(2)(3), 54 orbits
	//

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
	permutation<4> perm; perm.permute(0, 1);
	se_perm<4, double> cycle(perm, 1.0);
	sym.insert(cycle);

	orbit_list<4, double> orblst(sym);
	size_t norb_ref = 54;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	abs_index<4> aio(dims);
	do {
	    const index<4> &io = aio.get_index();
		bool can_ref = io[0] <= io[1];
		bool can = false;
		orbit_list<4, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(orblst.get_index(i))) {
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
	} while(aio.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_4() throw(libtest::test_exception) {

	//
	//	dim [3,3,3,3], split [3,3,3,3]
	//	(1)(2)(2)(3), 54 orbits
	//

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
	permutation<4> perm; perm.permute(1, 2);
	se_perm<4, double> cycle(perm, 1.0);
	sym.insert(cycle);

	orbit_list<4, double> orblst(sym);
	size_t norb_ref = 54;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	abs_index<4> aio(dims);
	do {
	    const index<4> &io = aio.get_index();
		bool can_ref = io[1] <= io[2];
		bool can = false;
		orbit_list<4, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(orblst.get_index(i))) {
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
	} while(aio.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_5() throw(libtest::test_exception) {

	//
	//	dim [3,3,3,3], split [3,3,3,3]
	//	(1)(1)(1)(2), 33 orbits
	//

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
	permutation<4> perm; perm.permute(0, 1).permute(1, 2);
	se_perm<4, double> cycle(perm, 1.0);
	sym.insert(cycle);

	orbit_list<4, double> orblst(sym);
	size_t norb_ref = 33;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	abs_index<4> aio(dims);
	do {
	    const index<4> &io = aio.get_index();
		bool can_ref = (io[0] == io[1] && io[0] <= io[2]) ||
			(io[0] < io[1] && io[0] < io[2]);
		bool can = false;
		orbit_list<4, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(orblst.get_index(i))) {
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
	} while(aio.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_6() throw(libtest::test_exception) {

	//
	//	dim [3,3,3,3], split [3,3,3,3]
	//	(1)(1)(2)(2), 36 orbits
	//

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
	permutation<4> perm1, perm2;
	perm1.permute(0, 1);
	perm2.permute(2, 3);
	se_perm<4, double> cycle1(perm1, 1.0);
	se_perm<4, double> cycle2(perm2, 1.0);
	sym.insert(cycle1);
	sym.insert(cycle2);

	orbit_list<4, double> orblst(sym);
	size_t norb_ref = 36;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	abs_index<4> aio(dims);
	do {
	    const index<4> &io = aio.get_index();
		bool can_ref = (io[0] <= io[1] && io[2] <= io[3]);
		bool can = false;
		orbit_list<4, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(orblst.get_index(i))) {
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
	} while(aio.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_7() throw(libtest::test_exception) {

	//
	//	dim [3,3,3,3], split [3,3,3,3]
	//	(1)(1)(1)(2), (1) fully symmetric, 30 orbits
	//

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
	permutation<4> perm1, perm2;
	perm1.permute(0, 1).permute(1, 2);
	perm2.permute(0, 1);
	se_perm<4, double> cycle1(perm1, 1.0);
	se_perm<4, double> cycle2(perm2, 1.0);
	sym.insert(cycle1);
	sym.insert(cycle2);

	orbit_list<4, double> orblst(sym);
	size_t norb_ref = 30;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	abs_index<4> aio(dims);
	do {
	    const index<4> &io = aio.get_index();
		bool can_ref = (io[0] <= io[1] && io[1] <= io[2]);
		bool can = false;
		orbit_list<4, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(orblst.get_index(i))) {
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
	} while(aio.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_8() throw(libtest::test_exception) {

	//
	//	dim [3,3,3,3], split [3,3,3,3]
	//	(1)(1)(1)(1), (1) fully symmetric, 15 orbits
	//

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
	permutation<4> perm1, perm2;
	perm1.permute(0, 1).permute(1, 2).permute(2, 3);
	perm2.permute(0, 1);
	se_perm<4, double> cycle1(perm1, 1.0);
	se_perm<4, double> cycle2(perm2, 1.0);
	sym.insert(cycle1);
	sym.insert(cycle2);

	orbit_list<4, double> orblst(sym);
	size_t norb_ref = 15;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	abs_index<4> aio(dims);
	do {
	    const index<4> &io = aio.get_index();
		bool can_ref =
			(io[0] <= io[1] && io[1] <= io[2] && io[2] <= io[3]);
		bool can = false;
		orbit_list<4, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(orblst.get_index(i))) {
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
	} while(aio.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void orbit_list_test::test_9() throw(libtest::test_exception) {

	//
	//	dim [3,6,3,6], split [2,3,2,3]
	//	(1)(2)(1)(2), 18 orbits
	//

	static const char *testname = "orbit_list_test::test_9()";

	try {

	index<4> i1, i2;
	i2[0] = 2; i2[1] = 5; i2[2] = 2; i2[3] = 5;
	mask<4> msk1, msk2;
	msk1[0] = true; msk1[1] = false; msk1[2] = true; msk1[3] = false;
	msk2[0] = false; msk2[1] = true; msk2[2] = false; msk2[3] = true;
	dimensions<4> dims(index_range<4>(i1, i2));
	i2[0] = 1; i2[1] = 2; i2[2] = 1; i2[3] = 2;
	dimensions<4> bdims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	bis.split(msk1, 1);
	bis.split(msk2, 2);
	bis.split(msk2, 4);
	symmetry<4, double> sym(bis);
	permutation<4> perm1, perm2;
	perm1.permute(0, 2);
	perm2.permute(1, 3);
	se_perm<4, double> cycle1(perm1, 1.0);
	se_perm<4, double> cycle2(perm2, 1.0);
	sym.insert(cycle1);
	sym.insert(cycle2);

	orbit_list<4, double> orblst(sym);
	size_t norb_ref = 18;
	if(orblst.get_size() != norb_ref) {
		std::ostringstream ss;
		ss << "Invalid number of orbits: " << orblst.get_size()
			<< " vs. " << norb_ref << " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}

	abs_index<4> aio(bdims);
	do {
	    const index<4> &io = aio.get_index();
		bool can_ref = (io[0] <= io[2] && io[1] <= io[3]);
		bool can = false;
		orbit_list<4, double>::iterator i = orblst.begin();
		while(i != orblst.end()) {
			if(io.equals(orblst.get_index(i))) {
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
	} while(aio.inc());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

} // namespace libtensor
