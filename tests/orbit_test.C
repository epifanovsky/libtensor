#include <sstream>
#include <libtensor/core/orbit.h>
#include <libtensor/btod/transf_double.h>
#include <libtensor/symmetry/se_perm.h>
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

} // namespace libtensor
