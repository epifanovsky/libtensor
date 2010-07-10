#include <sstream>
#include <libtensor/core/orbit.h>
#include <libtensor/btod/transf_double.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/se_label.h>
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
	test_9();
	test_10();
}


void orbit_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "orbit_test::test_1()";
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

	index<2> io;
	do {
		orbit<2, double> orb(sym, io);
		if(!orb.is_allowed()) {
			std::ostringstream ss;
			ss << "Orbit not allowed: " << io << ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
		if(orb.get_abs_canonical_index() != dims.abs_index(io)) {
			std::ostringstream ss;
			ss << "Failure to detect a canonical index: " << io
				<< ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}

		const transf<2, double> &tr = orb.get_transf(io);
		permutation<2> pref;
		if(!tr.get_perm().equals(pref)) {
			std::ostringstream ss;
			ss << "Incorrect block permutation for " << io
				<< ": " << tr.get_perm() << " vs. " << pref
				<< " (ref).";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
		if(tr.get_coeff() != 1.0) {
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
	mask<2> msk;
	msk[0] = true; msk[1] = true;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	symmetry<2, double> sym(bis);
	permutation<2> perm; perm.permute(0, 1);
	se_perm<2, double> cycle(perm, true);
	sym.insert(cycle);

	index<2> io;
	do {
		orbit<2, double> orb(sym, io);
		if(!orb.is_allowed()) {
			std::ostringstream ss;
			ss << "Orbit not allowed: " << io << ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
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
			if(!tr.get_perm().is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.get_coeff() != 1.0) {
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
			if(!tr.get_perm().equals(pref)) {
				std::ostringstream ss;
				ss << "Incorrect block permutation for " << io
					<< ": " << tr.get_perm() << " vs. " << pref
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.get_coeff() != 1.0) {
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
	mask<4> msk;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	symmetry<4, double> sym(bis);
	permutation<4> perm; perm.permute(0, 1);
	se_perm<4, double> cycle(perm, true);
	sym.insert(cycle);

	index<4> io;
	do {
		orbit<4, double> orb(sym, io);
		if(!orb.is_allowed()) {
			std::ostringstream ss;
			ss << "Orbit not allowed: " << io << ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
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
			if(!tr.get_perm().is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.get_coeff() != 1.0) {
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
			if(!tr.get_perm().equals(pref)) {
				std::ostringstream ss;
				ss << "Incorrect block permutation for " << io
					<< ": " << tr.get_perm() << " vs. " << pref
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.get_coeff() != 1.0) {
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
	mask<4> msk;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	symmetry<4, double> sym(bis);
	permutation<4> perm;
	perm.permute(1, 2);
	se_perm<4, double> cycle(perm, true);
	sym.insert(cycle);

	index<4> io;
	do {
		orbit<4, double> orb(sym, io);
		if(!orb.is_allowed()) {
			std::ostringstream ss;
			ss << "Orbit not allowed: " << io << ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
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
			if(!tr.get_perm().is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.get_coeff() != 1.0) {
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
			if(!tr.get_perm().equals(pref)) {
				std::ostringstream ss;
				ss << "Incorrect block permutation for " << io
					<< ": " << tr.get_perm() << " vs. " << pref
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.get_coeff() != 1.0) {
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
	mask<4> msk;
	msk[0] = true; msk[1] = true; msk[2] = true; msk[3] = true;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	symmetry<4, double> sym(bis);
	permutation<4> perm;
	perm.permute(0, 1).permute(1, 2);
	se_perm<4, double> cycle(perm, true);
	sym.insert(cycle);

	index<4> io;
	do {
		orbit<4, double> orb(sym, io);
		if(!orb.is_allowed()) {
			std::ostringstream ss;
			ss << "Orbit not allowed: " << io << ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
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
			if(!tr.get_perm().is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.get_coeff() != 1.0) {
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
			if(!tr.get_perm().equals(p3)) {
				std::ostringstream ss;
				ss << "Incorrect block permutation for " << io
					<< ": " << tr.get_perm() << " vs. " << p3
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.get_coeff() != 1.0) {
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
	se_perm<4, double> cycle1(perm1, true);
	se_perm<4, double> cycle2(perm2, true);
	sym.insert(cycle1);
	sym.insert(cycle2);

	index<4> io;
	do {
		orbit<4, double> orb(sym, io);
		if(!orb.is_allowed()) {
			std::ostringstream ss;
			ss << "Orbit not allowed: " << io << ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
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
			if(!tr.get_perm().is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.get_coeff() != 1.0) {
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
			if(!tr.get_perm().equals(pref)) {
				std::ostringstream ss;
				ss << "Incorrect block permutation for " << io
					<< ": " << tr.get_perm() << " vs. " << pref
					<< " (ref).";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.get_coeff() != 1.0) {
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
	se_perm<4, double> cycle1(perm1, true);
	se_perm<4, double> cycle2(perm2, true);
	sym.insert(cycle1);
	sym.insert(cycle2);

	index<4> io;
	do {
		orbit<4, double> orb(sym, io);
		if(!orb.is_allowed()) {
			std::ostringstream ss;
			ss << "Orbit not allowed: " << io << ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
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
			if(!tr.get_perm().is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.get_coeff() != 1.0) {
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
			io3.permute(tr.get_perm());
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
					<< "->" << io << ": " << tr.get_perm()
					<< ".";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.get_coeff() != 1.0) {
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
	se_perm<4, double> cycle1(perm1, true);
	se_perm<4, double> cycle2(perm2, true);
	sym.insert(cycle1);
	sym.insert(cycle2);

	index<4> io;
	do {
		orbit<4, double> orb(sym, io);
		if(!orb.is_allowed()) {
			std::ostringstream ss;
			ss << "Orbit not allowed: " << io << ".";
			fail_test(testname, __FILE__, __LINE__,
				ss.str().c_str());
		}
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
			if(!tr.get_perm().is_identity()) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block permutation (1).");
			}
			if(tr.get_coeff() != 1.0) {
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
			io3.permute(tr.get_perm());
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
					<< "->" << io << ": " << tr.get_perm()
					<< ".";
				fail_test(testname, __FILE__, __LINE__,
					ss.str().c_str());
			}
			if(tr.get_coeff() != 1.0) {
				fail_test(testname, __FILE__, __LINE__,
					"Incorrect block scaling coeff (2).");
			}
		}
	} while(dims.inc_index(io));

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void orbit_test::test_9() throw(libtest::test_exception) {

	static const char *testname = "orbit_test::test_9()";

	typedef point_group_table::label_t label_t;

	try {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	mask<2> msk;
	msk[0] = true; msk[1] = true;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	dimensions<2> bidims = bis.get_block_index_dims();

	point_group_table cs(2);
	label_t ap = 0, app = 1;
	cs.set_product(ap, ap, 0, ap);
	cs.set_product(ap, app, 0, app);
	cs.set_product(app, ap, 0, app);
	cs.set_product(app, app, 0, ap);
	cs.check();
	product_table_container::get_instance().add(cs);

	mask<2> m; m[0] = true; m[1] = true;
	se_label<2, double> elem1(bis.get_block_index_dims(),
			point_group_table::k_id);
	elem1.assign(m, 0, ap);
	elem1.assign(m, 1, app);
	elem1.set_target(ap);

	symmetry<2, double> sym(bis);
	sym.insert(elem1);

	permutation<2> p0;

	index<2> i00;
	abs_index<2> ai00(i00, bidims);
	orbit<2, double> o00(sym, i00);
	if(!o00.is_allowed()) {
		std::ostringstream ss;
		ss << "Orbit not allowed: " << i00 << ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(o00.get_abs_canonical_index() != ai00.get_abs_index()) {
		std::ostringstream ss;
		ss << "Failure to detect a canonical index: " << i00
			<< ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	const transf<2, double> &tr00 = o00.get_transf(i00);
	if(!tr00.get_perm().equals(p0)) {
		std::ostringstream ss;
		ss << "Incorrect block permutation for " << i00
			<< ": " << tr00.get_perm() << " vs. " << p0
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(tr00.get_coeff() != 1.0) {
		fail_test(testname, __FILE__, __LINE__,
			"Incorrect block transformation (coeff).");
	}

	index<2> i01; i01[1] = 1;
	abs_index<2> ai01(i01, bidims);
	orbit<2, double> o01(sym, i01);
	if(o01.is_allowed()) {
		std::ostringstream ss;
		ss << "Orbit allowed: " << i01 << ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(o01.get_abs_canonical_index() != ai01.get_abs_index()) {
		std::ostringstream ss;
		ss << "Failure to detect a canonical index: " << i01
			<< ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	const transf<2, double> &tr01 = o01.get_transf(i01);
	if(!tr01.get_perm().equals(p0)) {
		std::ostringstream ss;
		ss << "Incorrect block permutation for " << i01
			<< ": " << tr01.get_perm() << " vs. " << p0
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(tr01.get_coeff() != 1.0) {
		fail_test(testname, __FILE__, __LINE__,
			"Incorrect block transformation (coeff).");
	}

	index<2> i10; i10[0] = 1;
	abs_index<2> ai10(i10, bidims);
	orbit<2, double> o10(sym, i10);
	if(o10.is_allowed()) {
		std::ostringstream ss;
		ss << "Orbit allowed: " << i10 << ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(o10.get_abs_canonical_index() != ai10.get_abs_index()) {
		std::ostringstream ss;
		ss << "Failure to detect a canonical index: " << i10
			<< ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	const transf<2, double> &tr10 = o10.get_transf(i10);
	if(!tr10.get_perm().equals(p0)) {
		std::ostringstream ss;
		ss << "Incorrect block permutation for " << i10
			<< ": " << tr10.get_perm() << " vs. " << p0
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(tr10.get_coeff() != 1.0) {
		fail_test(testname, __FILE__, __LINE__,
			"Incorrect block transformation (coeff).");
	}

	index<2> i11; i11[0] = 1; i11[1] = 1;
	abs_index<2> ai11(i11, bidims);
	orbit<2, double> o11(sym, i11);
	if(!o11.is_allowed()) {
		std::ostringstream ss;
		ss << "Orbit not allowed: " << i11 << ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(o11.get_abs_canonical_index() != ai11.get_abs_index()) {
		std::ostringstream ss;
		ss << "Failure to detect a canonical index: " << i11
			<< ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	const transf<2, double> &tr11 = o11.get_transf(i11);
	if(!tr11.get_perm().equals(p0)) {
		std::ostringstream ss;
		ss << "Incorrect block permutation for " << i11
			<< ": " << tr11.get_perm() << " vs. " << p0
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(tr11.get_coeff() != 1.0) {
		fail_test(testname, __FILE__, __LINE__,
			"Incorrect block transformation (coeff).");
	}

	product_table_container::get_instance().erase(point_group_table::k_id);

	} catch(exception &e) {
		product_table_container::get_instance().erase(point_group_table::k_id);
		fail_test(testname, __FILE__, __LINE__, e.what());
	} catch(libtest::test_exception) {
		product_table_container::get_instance().erase(point_group_table::k_id);
		throw;
	}


}


void orbit_test::test_10() throw(libtest::test_exception) {

	static const char *testname = "orbit_test::test_10()";

	typedef point_group_table::label_t label_t;

	try {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	mask<2> msk;
	msk[0] = true; msk[1] = true;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(msk, 1);
	bis.split(msk, 2);
	dimensions<2> bidims = bis.get_block_index_dims();

	point_group_table cs(2);
	label_t ap = 0, app = 1;
	cs.set_product(ap, ap, 0, ap);
	cs.set_product(ap, app, 0, app);
	cs.set_product(app, ap, 0, app);
	cs.set_product(app, app, 0, ap);
	cs.check();
//	product_table_container::get_instance().add(cs);

	mask<2> m; m[0] = true; m[1] = true;
	se_label<2, double> elem1(bis.get_block_index_dims(),
			point_group_table::k_id);
	elem1.assign(m, 0, ap);
	elem1.assign(m, 1, app);
	elem1.set_target(ap);

	se_perm<2, double> elem2(permutation<2>().permute(0, 1), true);

	symmetry<2, double> sym(bis);
	sym.insert(elem1);
	sym.insert(elem2);

	permutation<2> p0, p1;
	p1.permute(0, 1);

	index<2> i00;
	abs_index<2> ai00(i00, bidims);
	orbit<2, double> o00(sym, i00);
	if(!o00.is_allowed()) {
		std::ostringstream ss;
		ss << "Orbit not allowed: " << i00 << ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(o00.get_abs_canonical_index() != ai00.get_abs_index()) {
		std::ostringstream ss;
		ss << "Failure to detect a canonical index: " << i00
			<< ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	const transf<2, double> &tr00 = o00.get_transf(i00);
	if(!tr00.get_perm().equals(p0)) {
		std::ostringstream ss;
		ss << "Incorrect block permutation for " << i00
			<< ": " << tr00.get_perm() << " vs. " << p0
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(tr00.get_coeff() != 1.0) {
		fail_test(testname, __FILE__, __LINE__,
			"Incorrect block transformation (coeff).");
	}

	index<2> i01; i01[1] = 1;
	abs_index<2> ai01(i01, bidims);
	orbit<2, double> o01(sym, i01);
	if(o01.is_allowed()) {
		std::ostringstream ss;
		ss << "Orbit allowed: " << i01 << ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(o01.get_abs_canonical_index() != ai01.get_abs_index()) {
		std::ostringstream ss;
		ss << "Failure to detect a canonical index: " << i01
			<< ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	const transf<2, double> &tr01 = o01.get_transf(i01);
	if(!tr01.get_perm().equals(p0)) {
		std::ostringstream ss;
		ss << "Incorrect block permutation for " << i01
			<< ": " << tr01.get_perm() << " vs. " << p0
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(tr01.get_coeff() != 1.0) {
		fail_test(testname, __FILE__, __LINE__,
			"Incorrect block transformation (coeff).");
	}

	index<2> i10; i10[0] = 1;
	abs_index<2> ai10(i10, bidims);
	orbit<2, double> o10(sym, i10);
	if(o10.is_allowed()) {
		std::ostringstream ss;
		ss << "Orbit allowed: " << i10 << ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(o10.get_abs_canonical_index() != ai01.get_abs_index()) {
		std::ostringstream ss;
		ss << "Failure to detect a canonical index: " << i10
			<< ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	const transf<2, double> &tr10 = o10.get_transf(i10);
	if(!tr10.get_perm().equals(p1)) {
		std::ostringstream ss;
		ss << "Incorrect block permutation for " << i10
			<< ": " << tr10.get_perm() << " vs. " << p1
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(tr10.get_coeff() != 1.0) {
		fail_test(testname, __FILE__, __LINE__,
			"Incorrect block transformation (coeff).");
	}

	index<2> i11; i11[0] = 1; i11[1] = 1;
	abs_index<2> ai11(i11, bidims);
	orbit<2, double> o11(sym, i11);
	if(!o11.is_allowed()) {
		std::ostringstream ss;
		ss << "Orbit not allowed: " << i11 << ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(o11.get_abs_canonical_index() != ai11.get_abs_index()) {
		std::ostringstream ss;
		ss << "Failure to detect a canonical index: " << i11
			<< ".";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	const transf<2, double> &tr11 = o11.get_transf(i11);
	if(!tr11.get_perm().equals(p0)) {
		std::ostringstream ss;
		ss << "Incorrect block permutation for " << i11
			<< ": " << tr11.get_perm() << " vs. " << p0
			<< " (ref).";
		fail_test(testname, __FILE__, __LINE__,
			ss.str().c_str());
	}
	if(tr11.get_coeff() != 1.0) {
		fail_test(testname, __FILE__, __LINE__,
			"Incorrect block transformation (coeff).");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
