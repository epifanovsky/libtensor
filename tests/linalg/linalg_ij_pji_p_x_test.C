#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include "linalg_ij_pji_p_x_test.h"

namespace libtensor {


void linalg_ij_pji_p_x_test::perform() throw(libtest::test_exception) {

	//              ni  nj  np  sic sja spa spb
	test_ij_pji_p_x(1,  1,  1,  1,  1,  1,  1);
	test_ij_pji_p_x(1,  1,  2,  1,  1,  2,  3);
	test_ij_pji_p_x(1,  2,  1,  2,  1,  2,  2);
	test_ij_pji_p_x(2,  1,  1,  1,  2,  2,  1);
	test_ij_pji_p_x(2,  2,  2,  2,  2,  4,  1);
	test_ij_pji_p_x(5,  3,  7,  6,  5,  15, 1);
	test_ij_pji_p_x(16, 16, 16, 16, 16, 256,16);
	test_ij_pji_p_x(17, 9,  5,  50, 20, 200,10);
}


void linalg_ij_pji_p_x_test::test_ij_pji_p_x(size_t ni, size_t nj, size_t np,
	size_t sic, size_t sja, size_t spa, size_t spb)
	throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_ij_pji_p_x_test::test_i_ipq_qp_x("
		<< ni << ", " << nj << ", " << np << ", " << sic << ", "
		<< sja << ", " << spa << ", " << spb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = np * spa, szb = np, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ij_pji_p_x(ni, nj, np, a, sja, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pji_p_x(ni, nj, np, a, sja, spa, b, spb, c_ref,
		sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ij_pji_p_x(ni, nj, np, a, sja, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pji_p_x(ni, nj, np, a, sja, spa, b, spb, c_ref,
		sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ij_pji_p_x(ni, nj, np, a, sja, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pji_p_x(ni, nj, np, a, sja, spa, b, spb, c_ref,
		sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ij_pji_p_x(ni, nj, np, a, sja, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pji_p_x(ni, nj, np, a, sja, spa, b, spb, c_ref,
		sic, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ij_pji_p_x(ni, nj, np, a, sja, spa, b, spb, c, sic, d);
	linalg_base_generic::ij_pji_p_x(ni, nj, np, a, sja, spa, b, spb, c_ref,
		sic, d);

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
