#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include "linalg_ijk_ipq_kjqp_x_test.h"

namespace libtensor {


void linalg_ijk_ipq_kjqp_x_test::perform() throw(libtest::test_exception) {

	test_ijk_ipq_kjqp_x(1, 1, 1, 1, 1);
	test_ijk_ipq_kjqp_x(2, 1, 1, 1, 1);
	test_ijk_ipq_kjqp_x(1, 2, 1, 1, 1);
	test_ijk_ipq_kjqp_x(1, 1, 2, 1, 1);
	test_ijk_ipq_kjqp_x(1, 1, 1, 1, 1);
	test_ijk_ipq_kjqp_x(1, 1, 1, 2, 1);
	test_ijk_ipq_kjqp_x(1, 1, 1, 1, 2);
	test_ijk_ipq_kjqp_x(2, 3, 2, 2, 3);
	test_ijk_ipq_kjqp_x(3, 5, 1, 13, 11);
	test_ijk_ipq_kjqp_x(16, 16, 16, 16, 16);
	test_ijk_ipq_kjqp_x(17, 16, 17, 17, 16);
}


void linalg_ijk_ipq_kjqp_x_test::test_ijk_ipq_kjqp_x(size_t ni, size_t nj,
	size_t nk, size_t np, size_t nq)
	throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_ijk_ipq_kjqp_x_test::test_ijk_ipq_kjqp_x("
		<< ni << ", " << nj << ", " << nk << ", " << np << ", "
		<< nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;
	double d = 0.0;

	try {

	size_t sza = ni * np * nq, szb = nk * nj * nq * np, szc = ni * nj * nk;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	d = 0.0;
	linalg::ijk_ipq_kjqp_x(ni, nj, nk, np, nq, a, b, c, d);
	linalg_base_generic::ijk_ipq_kjqp_x(ni, nj, nk, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	d = 1.0;
	linalg::ijk_ipq_kjqp_x(ni, nj, nk, np, nq, a, b, c, d);
	linalg_base_generic::ijk_ipq_kjqp_x(ni, nj, nk, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	d = -1.0;
	linalg::ijk_ipq_kjqp_x(ni, nj, nk, np, nq, a, b, c, d);
	linalg_base_generic::ijk_ipq_kjqp_x(ni, nj, nk, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	d = drand48();
	linalg::ijk_ipq_kjqp_x(ni, nj, nk, np, nq, a, b, c, d);
	linalg_base_generic::ijk_ipq_kjqp_x(ni, nj, nk, np, nq,
		a, b, c_ref, d);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	d = -drand48();
	linalg::ijk_ipq_kjqp_x(ni, nj, nk, np, nq, a, b, c, d);
	linalg_base_generic::ijk_ipq_kjqp_x(ni, nj, nk, np, nq,
		a, b, c_ref, d);

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
