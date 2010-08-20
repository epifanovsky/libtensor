#include <cmath>
#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/linalg/linalg.h>
#include "linalg_test.h"

namespace libtensor {


void linalg_test::perform() throw(libtest::test_exception) {

	test_x_p_p(1, 1, 1);
	test_x_p_p(2, 1, 1);
	test_x_p_p(16, 1, 1);
	test_x_p_p(17, 1, 1);
	test_x_p_p(2, 2, 3);
	test_x_p_p(2, 3, 2);

	test_i_i_x(1, 1, 1);
	test_i_i_x(2, 1, 1);
	test_i_i_x(16, 1, 1);
	test_i_i_x(17, 1, 1);
	test_i_i_x(2, 2, 3);
	test_i_i_x(2, 3, 2);

	test_i_ip_p(1, 1, 1, 1, 1);
	test_i_ip_p(1, 2, 2, 1, 1);
	test_i_ip_p(2, 1, 1, 1, 1);
	test_i_ip_p(16, 16, 16, 1, 1);
	test_i_ip_p(17, 3, 3, 1, 1);
	test_i_ip_p(2, 2, 2, 3, 4);
	test_i_ip_p(2, 2, 4, 3, 2);

	test_i_pi_p(1, 1, 1, 1, 1);
	test_i_pi_p(1, 2, 1, 1, 1);
	test_i_pi_p(2, 1, 1, 2, 1);
	test_i_pi_p(16, 16, 1, 16, 1);
	test_i_pi_p(17, 3, 1, 17, 1);
	test_i_pi_p(2, 2, 2, 3, 4);
	test_i_pi_p(2, 2, 4, 3, 2);

	test_ij_i_j(1, 1, 1, 1, 1);
	test_ij_i_j(1, 2, 1, 2, 1);
	test_ij_i_j(2, 1, 1, 1, 1);
	test_ij_i_j(16, 16, 1, 16, 1);
	test_ij_i_j(3, 17, 1, 17, 1);
	test_ij_i_j(2, 2, 2, 3, 4);
	test_ij_i_j(2, 2, 4, 3, 2);

	test_ij_ip_jp(1, 1, 1, 1, 1, 1);
	test_ij_ip_jp(1, 2, 3, 3, 2, 3);
	test_ij_ip_jp(2, 1, 3, 3, 1, 3);
	test_ij_ip_jp(16, 16, 1, 1, 16, 1);
	test_ij_ip_jp(3, 17, 2, 2, 17, 2);
	test_ij_ip_jp(2, 2, 2, 2, 3, 4);
	test_ij_ip_jp(2, 2, 2, 4, 3, 2);

	test_ij_ip_pj(1, 1, 1, 1, 1, 1);
	test_ij_ip_pj(1, 2, 3, 3, 2, 2);
	test_ij_ip_pj(2, 1, 3, 3, 1, 1);
	test_ij_ip_pj(16, 16, 1, 1, 16, 16);
	test_ij_ip_pj(3, 17, 2, 2, 17, 17);
	test_ij_ip_pj(2, 2, 2, 2, 3, 4);
	test_ij_ip_pj(2, 2, 2, 4, 3, 2);

	test_ij_pi_jp(1, 1, 1, 1, 1, 1);
	test_ij_pi_jp(1, 2, 3, 2, 3, 1);
	test_ij_pi_jp(2, 1, 3, 1, 3, 2);
	test_ij_pi_jp(16, 16, 1, 16, 1, 16);
	test_ij_pi_jp(3, 17, 2, 17, 2, 3);
	test_ij_pi_jp(2, 2, 2, 2, 3, 4);
	test_ij_pi_jp(2, 2, 2, 4, 3, 2);

	test_ij_pi_pj(1, 1, 1, 1, 1, 1);
	test_ij_pi_pj(1, 2, 3, 2, 1, 2);
	test_ij_pi_pj(2, 1, 3, 1, 2, 1);
	test_ij_pi_pj(16, 16, 1, 16, 16, 16);
	test_ij_pi_pj(3, 17, 2, 17, 3, 17);
	test_ij_pi_pj(2, 2, 2, 2, 3, 4);
	test_ij_pi_pj(2, 2, 2, 4, 3, 2);

	test_x_pq_qp(1, 1, 1, 1);
	test_x_pq_qp(1, 2, 2, 1);
	test_x_pq_qp(2, 1, 1, 2);
	test_x_pq_qp(2, 2, 2, 2);
	test_x_pq_qp(16, 2, 2, 16);
	test_x_pq_qp(17, 3, 3, 17);
	test_x_pq_qp(2, 2, 3, 4);
	test_x_pq_qp(2, 2, 4, 3);

	//            ni  np  nq  sia sic spa sqb
	test_i_ipq_qp(1,  1,  1,  1,  1,  1,  1);
	test_i_ipq_qp(1,  1,  2,  2,  1,  2,  1);
	test_i_ipq_qp(1,  2,  1,  2,  1,  1,  2);
	test_i_ipq_qp(2,  1,  1,  1,  2,  1,  1);
	test_i_ipq_qp(2,  2,  2,  4,  2,  2,  2);
	test_i_ipq_qp(5,  3,  7,  21, 5,  7,  3);
	test_i_ipq_qp(16, 16, 16, 256,16, 16, 16);
	test_i_ipq_qp(17, 9,  5,  50, 20, 5,  10);

	//              ni  nj  np  nq  sia sic sjb spa sqb
	test_ij_ipq_jqp(1,  1,  1,  1,  1,  1,  1,  1,  1);
	test_ij_ipq_jqp(1,  1,  2,  1,  2,  1,  2,  1,  2);
	test_ij_ipq_jqp(1,  2,  1,  2,  2,  2,  2,  2,  1);
	test_ij_ipq_jqp(2,  1,  1,  1,  1,  1,  1,  1,  1);
	test_ij_ipq_jqp(2,  2,  2,  2,  4,  2,  4,  2,  2);
	test_ij_ipq_jqp(5,  3,  7,  4,  28, 3,  28, 4,  7);
	test_ij_ipq_jqp(16, 16, 16, 16, 256,16, 256,16, 16);
	test_ij_ipq_jqp(17, 9,  5,  2,  60, 10, 30, 10, 10);

	test_ijkl_iplq_kpjq(1, 1, 1, 1, 1, 1);
	test_ijkl_iplq_kpjq(2, 1, 1, 1, 1, 1);
	test_ijkl_iplq_kpjq(1, 2, 1, 1, 1, 1);
	test_ijkl_iplq_kpjq(1, 1, 2, 1, 1, 1);
	test_ijkl_iplq_kpjq(1, 1, 1, 2, 1, 1);
	test_ijkl_iplq_kpjq(1, 1, 1, 1, 2, 1);
	test_ijkl_iplq_kpjq(1, 1, 1, 1, 1, 2);
	test_ijkl_iplq_kpjq(2, 3, 2, 3, 2, 3);
	test_ijkl_iplq_kpjq(3, 5, 1, 7, 13, 11);
	test_ijkl_iplq_kpjq(16, 16, 16, 16, 16, 16);
	test_ijkl_iplq_kpjq(17, 16, 17, 16, 17, 16);

	test_ijkl_iplq_pkjq(1, 1, 1, 1, 1, 1);
	test_ijkl_iplq_pkjq(2, 1, 1, 1, 1, 1);
	test_ijkl_iplq_pkjq(1, 2, 1, 1, 1, 1);
	test_ijkl_iplq_pkjq(1, 1, 2, 1, 1, 1);
	test_ijkl_iplq_pkjq(1, 1, 1, 2, 1, 1);
	test_ijkl_iplq_pkjq(1, 1, 1, 1, 2, 1);
	test_ijkl_iplq_pkjq(1, 1, 1, 1, 1, 2);
	test_ijkl_iplq_pkjq(2, 3, 2, 3, 2, 3);
	test_ijkl_iplq_pkjq(3, 5, 1, 7, 13, 11);
	test_ijkl_iplq_pkjq(16, 16, 16, 16, 16, 16);
	test_ijkl_iplq_pkjq(17, 16, 17, 16, 17, 16);

	test_ijkl_iplq_pkqj(1, 1, 1, 1, 1, 1);
	test_ijkl_iplq_pkqj(2, 1, 1, 1, 1, 1);
	test_ijkl_iplq_pkqj(1, 2, 1, 1, 1, 1);
	test_ijkl_iplq_pkqj(1, 1, 2, 1, 1, 1);
	test_ijkl_iplq_pkqj(1, 1, 1, 2, 1, 1);
	test_ijkl_iplq_pkqj(1, 1, 1, 1, 2, 1);
	test_ijkl_iplq_pkqj(1, 1, 1, 1, 1, 2);
	test_ijkl_iplq_pkqj(2, 3, 2, 3, 2, 3);
	test_ijkl_iplq_pkqj(3, 5, 1, 7, 13, 11);
	test_ijkl_iplq_pkqj(16, 16, 16, 16, 16, 16);
	test_ijkl_iplq_pkqj(17, 16, 17, 16, 17, 16);

	test_ijkl_pilq_kpjq(1, 1, 1, 1, 1, 1);
	test_ijkl_pilq_kpjq(2, 1, 1, 1, 1, 1);
	test_ijkl_pilq_kpjq(1, 2, 1, 1, 1, 1);
	test_ijkl_pilq_kpjq(1, 1, 2, 1, 1, 1);
	test_ijkl_pilq_kpjq(1, 1, 1, 2, 1, 1);
	test_ijkl_pilq_kpjq(1, 1, 1, 1, 2, 1);
	test_ijkl_pilq_kpjq(1, 1, 1, 1, 1, 2);
	test_ijkl_pilq_kpjq(2, 3, 2, 3, 2, 3);
	test_ijkl_pilq_kpjq(3, 5, 1, 7, 13, 11);
	test_ijkl_pilq_kpjq(16, 16, 16, 16, 16, 16);
	test_ijkl_pilq_kpjq(17, 16, 17, 16, 17, 16);

	test_ijkl_pilq_pkjq(1, 1, 1, 1, 1, 1);
	test_ijkl_pilq_pkjq(2, 1, 1, 1, 1, 1);
	test_ijkl_pilq_pkjq(1, 2, 1, 1, 1, 1);
	test_ijkl_pilq_pkjq(1, 1, 2, 1, 1, 1);
	test_ijkl_pilq_pkjq(1, 1, 1, 2, 1, 1);
	test_ijkl_pilq_pkjq(1, 1, 1, 1, 2, 1);
	test_ijkl_pilq_pkjq(1, 1, 1, 1, 1, 2);
	test_ijkl_pilq_pkjq(2, 3, 2, 3, 2, 3);
	test_ijkl_pilq_pkjq(3, 5, 1, 7, 13, 11);
	test_ijkl_pilq_pkjq(16, 16, 16, 16, 16, 16);
	test_ijkl_pilq_pkjq(17, 16, 17, 16, 17, 16);

}


void linalg_test::test_x_p_p(size_t np, size_t spa, size_t spb)
	throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_x_p_p(" << np << ", " << spa << ", " << spb
		<< ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0;

	try {

	size_t sza = np * spa, szb = np * spb;

	a = new double[sza];
	b = new double[szb];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();

	double c = linalg::x_p_p(a, b, np, spa, spb);
	double c_ref = linalg_impl_generic::x_p_p(a, b, np, spa, spb);

	if(!cmp(c - c_ref, c_ref)) {
		fail_test(tnss.c_str(), __FILE__, __LINE__,
			"Incorrect result.");
	}

	delete [] a; a = 0;
	delete [] b; b = 0;

	} catch(exception &e) {
		delete [] a;
		delete [] b;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a;
		delete [] b;
		throw;
	}
}


void linalg_test::test_i_i_x(size_t ni, size_t sia, size_t sic)
	throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_i_i_x(" << ni << ", " << sia << ", " << sic
		<< ")";
	std::string tnss = ss.str();

	double *a = 0, *c = 0, *c_ref = 0;
	double b = 0.0;

	try {

	size_t sza = ni * sia, szc = ni * sic;

	a = new double[sza];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();
	b = drand48();

	linalg::i_i_x(a, b, c, ni, sia, sic);
	linalg_impl_generic::i_i_x(a, b, c_ref, ni, sia, sic);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result.");
		}
	}

	delete [] a; a = 0;
	delete [] c; c = 0;
	delete [] c_ref; c_ref = 0;

	} catch(exception &e) {
		delete [] a; a = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a; a = 0;
		delete [] c; c = 0;
		delete [] c_ref; c_ref = 0;
		throw;
	}
}


void linalg_test::test_i_ip_p(size_t ni, size_t np, size_t sia, size_t sic,
	size_t spb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_i_ip_p(" << ni << ", " << np << ", " << sia
		<< ", " << sic << ", " << spb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = ni * sia, szb = np * spb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::i_ip_p(a, b, c, 0.0, ni, np, sia, sic, spb);
	linalg_impl_generic::i_ip_p(a, b, c_ref, 0.0, ni, np, sia, sic, spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::i_ip_p(a, b, c, 1.0, ni, np, sia, sic, spb);
	linalg_impl_generic::i_ip_p(a, b, c_ref, 1.0, ni, np, sia, sic, spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::i_ip_p(a, b, c, -1.0, ni, np, sia, sic, spb);
	linalg_impl_generic::i_ip_p(a, b, c_ref, -1.0, ni, np, sia, sic, spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::i_ip_p(a, b, c, d, ni, np, sia, sic, spb);
	linalg_impl_generic::i_ip_p(a, b, c_ref, d, ni, np, sia, sic, spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::i_ip_p(a, b, c, -d, ni, np, sia, sic, spb);
	linalg_impl_generic::i_ip_p(a, b, c_ref, -d, ni, np, sia, sic, spb);

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


void linalg_test::test_i_pi_p(size_t ni, size_t np, size_t sic, size_t spa,
	size_t spb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_i_pi_p(" << ni << ", " << np << ", " << sic
		<< ", " << spa << ", " << spb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = np * spa, szb = np * spb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::i_pi_p(a, b, c, 0.0, ni, np, sic, spa, spb);
	linalg_impl_generic::i_pi_p(a, b, c_ref, 0.0, ni, np, sic, spa, spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::i_pi_p(a, b, c, 1.0, ni, np, sic, spa, spb);
	linalg_impl_generic::i_pi_p(a, b, c_ref, 1.0, ni, np, sic, spa, spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::i_pi_p(a, b, c, -1.0, ni, np, sic, spa, spb);
	linalg_impl_generic::i_pi_p(a, b, c_ref, -1.0, ni, np, sic, spa, spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::i_pi_p(a, b, c, d, ni, np, sic, spa, spb);
	linalg_impl_generic::i_pi_p(a, b, c_ref, d, ni, np, sic, spa, spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::i_pi_p(a, b, c, -d, ni, np, sic, spa, spb);
	linalg_impl_generic::i_pi_p(a, b, c_ref, -d, ni, np, sic, spa, spb);

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


void linalg_test::test_ij_i_j(size_t ni, size_t nj, size_t sia, size_t sic,
	size_t sjb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ij_i_j(" << ni << ", " << nj << ", " << sia
		<< ", " << sic << ", " << sjb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = ni * sia, szb = nj * sjb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::ij_i_j(a, b, c, 0.0, ni, nj, sia, sic, sjb);
	linalg_impl_generic::ij_i_j(a, b, c_ref, 0.0, ni, nj, sia, sic, sjb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::ij_i_j(a, b, c, 1.0, ni, nj, sia, sic, sjb);
	linalg_impl_generic::ij_i_j(a, b, c_ref, 1.0, ni, nj, sia, sic, sjb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::ij_i_j(a, b, c, -1.0, ni, nj, sia, sic, sjb);
	linalg_impl_generic::ij_i_j(a, b, c_ref, -1.0, ni, nj, sia, sic, sjb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::ij_i_j(a, b, c, d, ni, nj, sia, sic, sjb);
	linalg_impl_generic::ij_i_j(a, b, c_ref, d, ni, nj, sia, sic, sjb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::ij_i_j(a, b, c, -d, ni, nj, sia, sic, sjb);
	linalg_impl_generic::ij_i_j(a, b, c_ref, -d, ni, nj, sia, sic, sjb);

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


void linalg_test::test_ij_ip_jp(size_t ni, size_t nj, size_t np, size_t sia,
	size_t sic, size_t sjb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ij_ip_jp(" << ni << ", " << nj << ", " << np
		<< ", " << sia << ", " << sic << ", " << sjb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = ni * sia, szb = nj * sjb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::ij_ip_jp(a, b, c, 0.0, ni, nj, np, sia, sic, sjb);
	linalg_impl_generic::ij_ip_jp(a, b, c_ref, 0.0, ni, nj, np, sia, sic,
		sjb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::ij_ip_jp(a, b, c, 1.0, ni, nj, np, sia, sic, sjb);
	linalg_impl_generic::ij_ip_jp(a, b, c_ref, 1.0, ni, nj, np, sia, sic,
		sjb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::ij_ip_jp(a, b, c, -1.0, ni, nj, np, sia, sic, sjb);
	linalg_impl_generic::ij_ip_jp(a, b, c_ref, -1.0, ni, nj, np, sia, sic,
		sjb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::ij_ip_jp(a, b, c, d, ni, nj, np, sia, sic, sjb);
	linalg_impl_generic::ij_ip_jp(a, b, c_ref, d, ni, nj, np, sia, sic,
		sjb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::ij_ip_jp(a, b, c, -d, ni, nj, np, sia, sic, sjb);
	linalg_impl_generic::ij_ip_jp(a, b, c_ref, -d, ni, nj, np, sia, sic,
		sjb);

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


void linalg_test::test_ij_ip_pj(size_t ni, size_t nj, size_t np, size_t sia,
	size_t sic, size_t spb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ij_ip_pj(" << ni << ", " << nj << ", " << np
		<< ", " << sia << ", " << sic << ", " << spb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = ni * sia, szb = np * spb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::ij_ip_pj(a, b, c, 0.0, ni, nj, np, sia, sic, spb);
	linalg_impl_generic::ij_ip_pj(a, b, c_ref, 0.0, ni, nj, np, sia, sic,
		spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::ij_ip_pj(a, b, c, 1.0, ni, nj, np, sia, sic, spb);
	linalg_impl_generic::ij_ip_pj(a, b, c_ref, 1.0, ni, nj, np, sia, sic,
		spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::ij_ip_pj(a, b, c, -1.0, ni, nj, np, sia, sic, spb);
	linalg_impl_generic::ij_ip_pj(a, b, c_ref, -1.0, ni, nj, np, sia, sic,
		spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::ij_ip_pj(a, b, c, d, ni, nj, np, sia, sic, spb);
	linalg_impl_generic::ij_ip_pj(a, b, c_ref, d, ni, nj, np, sia, sic,
		spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::ij_ip_pj(a, b, c, -d, ni, nj, np, sia, sic, spb);
	linalg_impl_generic::ij_ip_pj(a, b, c_ref, -d, ni, nj, np, sia, sic,
		spb);

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


void linalg_test::test_ij_pi_jp(size_t ni, size_t nj, size_t np, size_t sic,
	size_t sjb, size_t spa) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ij_pi_jp(" << ni << ", " << nj << ", " << np
		<< ", " << sic << ", " << sjb << ", " << spa << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = np * spa, szb = nj * sjb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::ij_pi_jp(a, b, c, 0.0, ni, nj, np, sic, sjb, spa);
	linalg_impl_generic::ij_pi_jp(a, b, c_ref, 0.0, ni, nj, np, sic, sjb,
		spa);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::ij_pi_jp(a, b, c, 1.0, ni, nj, np, sic, sjb, spa);
	linalg_impl_generic::ij_pi_jp(a, b, c_ref, 1.0, ni, nj, np, sic, sjb,
		spa);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::ij_pi_jp(a, b, c, -1.0, ni, nj, np, sic, sjb, spa);
	linalg_impl_generic::ij_pi_jp(a, b, c_ref, -1.0, ni, nj, np, sic, sjb,
		spa);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::ij_pi_jp(a, b, c, d, ni, nj, np, sic, sjb, spa);
	linalg_impl_generic::ij_pi_jp(a, b, c_ref, d, ni, nj, np, sic, sjb,
		spa);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::ij_pi_jp(a, b, c, -d, ni, nj, np, sic, sjb, spa);
	linalg_impl_generic::ij_pi_jp(a, b, c_ref, -d, ni, nj, np, sic, sjb,
		spa);

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


void linalg_test::test_ij_pi_pj(size_t ni, size_t nj, size_t np, size_t sic,
	size_t spa, size_t spb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ij_pi_pj(" << ni << ", " << nj << ", " << np
		<< ", " << sic << ", " << spa << ", " << spb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = np * spa, szb = np * spb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::ij_pi_pj(a, b, c, 0.0, ni, nj, np, sic, spa, spb);
	linalg_impl_generic::ij_pi_pj(a, b, c_ref, 0.0, ni, nj, np, sic, spa,
		spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::ij_pi_pj(a, b, c, 1.0, ni, nj, np, sic, spa, spb);
	linalg_impl_generic::ij_pi_pj(a, b, c_ref, 1.0, ni, nj, np, sic, spa,
		spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::ij_pi_pj(a, b, c, -1.0, ni, nj, np, sic, spa, spb);
	linalg_impl_generic::ij_pi_pj(a, b, c_ref, -1.0, ni, nj, np, sic, spa,
		spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::ij_pi_pj(a, b, c, d, ni, nj, np, sic, spa, spb);
	linalg_impl_generic::ij_pi_pj(a, b, c_ref, d, ni, nj, np, sic, spa,
		spb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::ij_pi_pj(a, b, c, -d, ni, nj, np, sic, spa, spb);
	linalg_impl_generic::ij_pi_pj(a, b, c_ref, -d, ni, nj, np, sic, spa,
		spb);

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

void linalg_test::test_x_pq_qp(size_t np, size_t nq, size_t spa, size_t sqb)
	throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_x_pq_qp(" << np << ", " << nq << ", " << spa
		<< ", " << sqb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0;

	try {

	size_t sza = np * spa, szb = nq * sqb;

	a = new double[sza];
	b = new double[szb];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();

	double c = linalg::x_pq_qp(a, b, np, nq, spa, sqb);
	double c_ref = linalg_impl_generic::x_pq_qp(a, b, np, nq, spa, sqb);

	if(!cmp(c - c_ref, c_ref)) {
		fail_test(tnss.c_str(), __FILE__, __LINE__,
			"Incorrect result.");
	}

	delete [] a; a = 0;
	delete [] b; b = 0;

	} catch(exception &e) {
		delete [] a;
		delete [] b;
		fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
	} catch(...) {
		delete [] a;
		delete [] b;
		throw;
	}
}


void linalg_test::test_i_ipq_qp(size_t ni, size_t np, size_t nq, size_t sia,
	size_t sic, size_t spa, size_t sqb) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_i_ipq_qp(" << ni << ", " << np << ", " << nq
		<< ", " << sia << ", " << sic << ", " << spa << ", "
		<< sqb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = ni * sia, szb = nq * sqb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::i_ipq_qp(a, b, c, 0.0, ni, np, nq, sia, sic, spa, sqb);
	linalg_impl_generic::i_ipq_qp(a, b, c_ref, 0.0, ni, np, nq, sia, sic,
		spa, sqb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::i_ipq_qp(a, b, c, 1.0, ni, np, nq, sia, sic, spa, sqb);
	linalg_impl_generic::i_ipq_qp(a, b, c_ref, 1.0, ni, np, nq, sia, sic,
		spa, sqb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::i_ipq_qp(a, b, c, -1.0, ni, np, nq, sia, sic, spa, sqb);
	linalg_impl_generic::i_ipq_qp(a, b, c_ref, -1.0, ni, np, nq, sia, sic,
		spa, sqb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::i_ipq_qp(a, b, c, d, ni, np, nq, sia, sic, spa, sqb);
	linalg_impl_generic::i_ipq_qp(a, b, c_ref, d, ni, np, nq, sia, sic, spa,
		sqb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::i_ipq_qp(a, b, c, -d, ni, np, nq, sia, sic, spa, sqb);
	linalg_impl_generic::i_ipq_qp(a, b, c_ref, -d, ni, np, nq, sia, sic,
		spa, sqb);

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


void linalg_test::test_ij_ipq_jqp(size_t ni, size_t nj, size_t np, size_t nq,
	size_t sia, size_t sic, size_t sjb, size_t spa, size_t sqb)
	throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ij_ipq_jqp(" << ni << ", " << nj << ", "
		<< np << ", " << nq << ", " << sia << ", " << sic << ", "
		<< sjb << ", " << spa << ", " << sqb << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = ni * sia, szb = nj * sjb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::ij_ipq_jqp(a, b, c, 0.0, ni, nj, np, nq, sia, sic, sjb, spa,
		sqb);
	linalg_impl_generic::ij_ipq_jqp(a, b, c_ref, 0.0, ni, nj, np, nq, sia,
		sic, sjb, spa, sqb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::ij_ipq_jqp(a, b, c, 1.0, ni, nj, np, nq, sia, sic, sjb, spa,
		sqb);
	linalg_impl_generic::ij_ipq_jqp(a, b, c_ref, 1.0, ni, nj, np, nq, sia,
		sic, sjb, spa, sqb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::ij_ipq_jqp(a, b, c, -1.0, ni, nj, np, nq, sia, sic, sjb, spa,
		sqb);
	linalg_impl_generic::ij_ipq_jqp(a, b, c_ref, -1.0, ni, nj, np, nq, sia,
		sic, sjb, spa, sqb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::ij_ipq_jqp(a, b, c, d, ni, nj, np, nq, sia, sic, sjb, spa, sqb);
	linalg_impl_generic::ij_ipq_jqp(a, b, c_ref, d, ni, nj, np, nq, sia,
		sic, sjb, spa, sqb);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::ij_ipq_jqp(a, b, c, -d, ni, nj, np, nq, sia, sic, sjb, spa,
		sqb);
	linalg_impl_generic::ij_ipq_jqp(a, b, c_ref, -d, ni, nj, np, nq, sia,
		sic, sjb, spa, sqb);

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


void linalg_test::test_ijkl_iplq_kpjq(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_iplq_kpjq(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = ni * np * nl * nq, szb = np * nk * nj * nq,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::ijkl_iplq_kpjq(a, b, c, 0.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_kpjq(a, b, c_ref, 0.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::ijkl_iplq_kpjq(a, b, c, 1.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_kpjq(a, b, c_ref, 1.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::ijkl_iplq_kpjq(a, b, c, -1.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_kpjq(a, b, c_ref, -1.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::ijkl_iplq_kpjq(a, b, c, d, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_kpjq(a, b, c_ref, d, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::ijkl_iplq_kpjq(a, b, c, -d, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_kpjq(a, b, c_ref, -d, ni, nj, nk, nl,
		np, nq);

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


void linalg_test::test_ijkl_iplq_pkjq(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_iplq_pkjq(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = ni * np * nl * nq, szb = np * nk * nj * nq,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::ijkl_iplq_pkjq(a, b, c, 0.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_pkjq(a, b, c_ref, 0.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::ijkl_iplq_pkjq(a, b, c, 1.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_pkjq(a, b, c_ref, 1.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::ijkl_iplq_pkjq(a, b, c, -1.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_pkjq(a, b, c_ref, -1.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::ijkl_iplq_pkjq(a, b, c, d, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_pkjq(a, b, c_ref, d, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::ijkl_iplq_pkjq(a, b, c, -d, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_pkjq(a, b, c_ref, -d, ni, nj, nk, nl,
		np, nq);

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


void linalg_test::test_ijkl_iplq_pkqj(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_iplq_pkqj(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = ni * np * nl * nq, szb = np * nk * nq * nj,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::ijkl_iplq_pkqj(a, b, c, 0.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_pkqj(a, b, c_ref, 0.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::ijkl_iplq_pkqj(a, b, c, 1.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_pkqj(a, b, c_ref, 1.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::ijkl_iplq_pkqj(a, b, c, -1.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_pkqj(a, b, c_ref, -1.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::ijkl_iplq_pkqj(a, b, c, d, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_pkqj(a, b, c_ref, d, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::ijkl_iplq_pkqj(a, b, c, -d, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_iplq_pkqj(a, b, c_ref, -d, ni, nj, nk, nl,
		np, nq);

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


void linalg_test::test_ijkl_pilq_kpjq(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_pilq_kpjq(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = np * ni * nl * nq, szb = nk * np * nj * nq,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::ijkl_pilq_kpjq(a, b, c, 0.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_pilq_kpjq(a, b, c_ref, 0.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::ijkl_pilq_kpjq(a, b, c, 1.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_pilq_kpjq(a, b, c_ref, 1.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::ijkl_pilq_kpjq(a, b, c, -1.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_pilq_kpjq(a, b, c_ref, -1.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::ijkl_pilq_kpjq(a, b, c, d, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_pilq_kpjq(a, b, c_ref, d, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::ijkl_pilq_kpjq(a, b, c, -d, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_pilq_kpjq(a, b, c_ref, -d, ni, nj, nk, nl,
		np, nq);

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


void linalg_test::test_ijkl_pilq_pkjq(size_t ni, size_t nj, size_t nk,
	size_t nl, size_t np, size_t nq) throw(libtest::test_exception) {

	std::ostringstream ss;
	ss << "linalg_test::test_ijkl_pilq_pkjq(" << ni << ", " << nj << ", "
		<< nk << ", " << nl << ", " << np << ", " << nq << ")";
	std::string tnss = ss.str();

	double *a = 0, *b = 0, *c = 0, *c_ref = 0;

	try {

	size_t sza = np * ni * nl * nq, szb = np * nk * nj * nq,
		szc = ni * nj * nk * nl;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

	for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

	linalg::ijkl_pilq_pkjq(a, b, c, 0.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_pilq_pkjq(a, b, c_ref, 0.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 0.0).");
		}
	}

	linalg::ijkl_pilq_pkjq(a, b, c, 1.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_pilq_pkjq(a, b, c_ref, 1.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = 1.0).");
		}
	}

	linalg::ijkl_pilq_pkjq(a, b, c, -1.0, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_pilq_pkjq(a, b, c_ref, -1.0, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = -1.0).");
		}
	}

	double d = drand48();
	linalg::ijkl_pilq_pkjq(a, b, c, d, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_pilq_pkjq(a, b, c_ref, d, ni, nj, nk, nl,
		np, nq);

	for(size_t i = 0; i < szc; i++) {
		if(!cmp(c[i] - c_ref[i], c_ref[i])) {
			fail_test(tnss.c_str(), __FILE__, __LINE__,
				"Incorrect result (d = rnd).");
		}
	}

	linalg::ijkl_pilq_pkjq(a, b, c, -d, ni, nj, nk, nl, np, nq);
	linalg_impl_generic::ijkl_pilq_pkjq(a, b, c_ref, -d, ni, nj, nk, nl,
		np, nq);

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


bool linalg_test::cmp(double diff, double ref) {

	static const double k_thresh = 1e-12;

	if(fabs(ref) > 1.0) return fabs(diff) < fabs(ref) * k_thresh;
	else return fabs(diff) < k_thresh;
}


} // namespace libtensor
