#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include "linalg_ij_pi_pj_x_test.h"

namespace libtensor {


void linalg_ij_pi_pj_x_test::perform() throw(libtest::test_exception) {

	test_ij_pi_pj_x(1, 1, 1, 1, 1, 1);
	test_ij_pi_pj_x(1, 2, 3, 2, 1, 2);
	test_ij_pi_pj_x(2, 1, 3, 1, 2, 1);
	test_ij_pi_pj_x(16, 16, 1, 16, 16, 16);
	test_ij_pi_pj_x(3, 17, 2, 17, 3, 17);
	test_ij_pi_pj_x(2, 2, 2, 2, 3, 4);
	test_ij_pi_pj_x(2, 2, 2, 4, 3, 2);
}


void linalg_ij_pi_pj_x_test::test_ij_pi_pj_x(size_t ni, size_t nj, size_t np,
	size_t sic, size_t spa, size_t spb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_ij_pi_pj_x_test::test_ij_pi_pj_x("
		<< ni << ", " << nj << ", " << np << ", " << sic << ", "
		<< spa << ", " << spb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * spa, szb = np * spb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pi_pj_x(ni, nj, np, a, spa, b, spb, c_ref, sic,
		d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -rnd).");
		}
	}

	delete [] a; a = 0;
	delete [] b; b = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] b; b = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}

} // namespace libtensor
