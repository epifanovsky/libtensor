#include <cmath>
#include <cstdlib>
#include <ctime>
#include <libtensor.h>
#include "contraction2_processor_test.h"

namespace libtensor {

void contraction2_processor_test::perform() throw(libtest::test_exception) {
	srand48(time(NULL));

	test_0_p_p_p(10);

	test_i_p_ip_p(1, 10);
	test_i_p_ip_p(10, 10);
	test_i_p_ip_p(10, 20);

	test_i_p_p_ip(10, 10);
	test_i_p_p_ip(10, 20);

	test_ij_p_ip_jp(10, 10, 10);
	test_ij_p_ip_jp(10, 10, 20);
	test_ij_p_ip_jp(10, 20, 10);
	test_ij_p_ip_jp(20, 10, 10);
	test_ij_p_ip_jp(10, 20, 30);

	test_ji_p_ip_jp(10, 10, 10);
	test_ji_p_ip_jp(10, 10, 20);
	test_ji_p_ip_jp(10, 20, 10);
	test_ji_p_ip_jp(20, 10, 10);
	test_ji_p_ip_jp(10, 20, 30);

	test_kij_p_ip_jkp(10, 10, 10, 10);
	test_kij_p_ip_jkp(10, 10, 10, 20);
	test_kij_p_ip_jkp(10, 10, 20, 10);
	test_kij_p_ip_jkp(10, 20, 10, 10);
	test_kij_p_ip_jkp(20, 10, 10, 10);
	test_kij_p_ip_jkp(10, 10, 20, 20);
	test_kij_p_ip_jkp(10, 20, 10, 20);
	test_kij_p_ip_jkp(10, 20, 20, 10);
	test_kij_p_ip_jkp(20, 10, 10, 20);
	test_kij_p_ip_jkp(20, 10, 20, 10);
	test_kij_p_ip_jkp(20, 20, 10, 10);
	test_kij_p_ip_jkp(10, 20, 30, 40);

	test_kji_p_ikp_jp(10, 10, 10, 10);
	test_kji_p_ikp_jp(10, 10, 10, 20);
	test_kji_p_ikp_jp(10, 10, 20, 10);
	test_kji_p_ikp_jp(10, 20, 10, 10);
	test_kji_p_ikp_jp(20, 10, 10, 10);
	test_kji_p_ikp_jp(10, 10, 20, 20);
	test_kji_p_ikp_jp(10, 20, 10, 20);
	test_kji_p_ikp_jp(10, 20, 20, 10);
	test_kji_p_ikp_jp(20, 10, 10, 20);
	test_kji_p_ikp_jp(20, 10, 20, 10);
	test_kji_p_ikp_jp(20, 20, 10, 10);
	test_kji_p_ikp_jp(10, 20, 30, 40);
}

void contraction2_processor_test::test_0_p_p_p(size_t np)
	throw(libtest::test_exception) {

	// c = \sum_p a_p b_p
	// (simple dot product)

	contraction2_list<1> list;
	list.append(np, 1, 1, 0);

	double *ptra = new double[np];
	double *ptrb = new double[np];
	double *ptrc = new double;
	double c = 0.0;
	double c_ref = 0.0;

	for(size_t i=0; i<np; i++) {
		ptra[i] = drand48();
		ptrb[i] = drand48();
	}
	for(size_t i=0; i<np; i++) c_ref += ptra[i]*ptrb[i];

	contraction2_processor<1>(list, ptrc, ptra, ptrb).contract();
	c = *ptrc;

	delete ptrc;
	delete [] ptra; delete [] ptrb;

	if(!compare(c_ref, c)) {
		char msg[128];
		snprintf(msg, 128, "Difference found: "
			"%.5lg (act) vs. %.5lg (ref)", c, c_ref);
		fail_test("contraction2_processor_test::test_0_p_p_p()",
			__FILE__, __LINE__, msg);
	}

}

void contraction2_processor_test::test_i_p_ip_p(size_t ni, size_t np)
	throw(libtest::test_exception) {

	// c_i = \sum_p a_{ip} b_p
	// (matrix-vector multiplication)

	contraction2_list<2> list;
	list.append(ni, np, 0, 1);
	list.append(np, 1, 1, 0);

	double *ptra = new double[ni*np];
	double *ptrb = new double[np];
	double *ptrc = new double[ni];
	double *ptrc_ref = new double[ni];

	for(size_t i=0; i<ni*np; i++) ptra[i] = drand48();
	for(size_t i=0; i<np; i++) ptrb[i] = drand48();
	for(size_t i=0; i<ni; i++) ptrc_ref[i] = 0.0;

	for(size_t i=0; i<ni; i++) {
	for(size_t p=0; p<np; p++) {
		ptrc_ref[i] += ptra[i*np+p]*ptrb[p];
	}
	}

	contraction2_processor<2>(list, ptrc, ptra, ptrb).contract();

	bool fail = false;
	size_t fail_pos;
	double fail_c, fail_cref;

	for(size_t i=0; i<ni && !fail; i++) {
		if(!compare(ptrc_ref[i], ptrc[i])) {
			fail = true;
			fail_pos = i;
			fail_c = ptrc[i];
			fail_cref = ptrc_ref[i];
		}
	}

	delete [] ptra; delete [] ptrb; delete [] ptrc; delete [] ptrc_ref;

	if(fail) {
		char msg[128];
		snprintf(msg, 128, "Difference found at position %lu: "
			"%.5lg (act) vs. %.5lg (ref)",
			fail_pos, fail_c, fail_cref);
		fail_test("contraction2_processor_test::test_i_p_ip_p",
			__FILE__, __LINE__, msg);
	}
}

void contraction2_processor_test::test_i_p_p_ip(size_t ni, size_t np)
	throw(libtest::test_exception) {

	// c_i = \sum_p a_p b_{ip}
	// (matrix-vector multiplication)

	contraction2_list<2> list;
	list.append(ni, 0, np, 1);
	list.append(np, 1, 1, 0);

	double *ptra = new double[np];
	double *ptrb = new double[ni*np];
	double *ptrc = new double[ni];
	double *ptrc_ref = new double[ni];

	for(size_t i=0; i<np; i++) ptra[i] = drand48();
	for(size_t i=0; i<ni*np; i++) ptrb[i] = drand48();
	for(size_t i=0; i<ni; i++) ptrc_ref[i] = 0.0;

	for(size_t i=0; i<ni; i++) {
	for(size_t p=0; p<np; p++) {
		ptrc_ref[i] += ptra[p]*ptrb[i*np+p];
	}
	}

	contraction2_processor<2>(list, ptrc, ptra, ptrb).contract();

	bool fail = false;
	size_t fail_pos;
	double fail_c, fail_cref;

	for(size_t i=0; i<ni && !fail; i++) {
		if(!compare(ptrc_ref[i], ptrc[i])) {
			fail = true;
			fail_pos = i;
			fail_c = ptrc[i];
			fail_cref = ptrc_ref[i];
		}
	}

	delete [] ptra; delete [] ptrb; delete [] ptrc; delete [] ptrc_ref;

	if(fail) {
		char msg[128];
		snprintf(msg, 128, "Difference found at position %lu: "
			"%.5lg (act) vs. %.5lg (ref)",
			fail_pos, fail_c, fail_cref);
		fail_test("contraction2_processor_test::test_i_p_p_ip",
			__FILE__, __LINE__, msg);
	}
}

void contraction2_processor_test::test_ij_p_ip_jp(size_t ni, size_t nj,
	size_t np) throw(libtest::test_exception) {

	// c_{ij} = \sum_p a_{ip} b_{jp}
	// (matrix-matrix multiplication)

	contraction2_list<3> list;
	list.append(ni, np, 0, nj);
	list.append(nj, 0, np, 1);
	list.append(np, 1, 1, 0);

	double *ptra = new double[ni*np];
	double *ptrb = new double[nj*np];
	double *ptrc = new double[ni*nj];
	double *ptrc_ref = new double[ni*nj];

	for(size_t i=0; i<ni*np; i++) ptra[i] = drand48();
	for(size_t i=0; i<nj*np; i++) ptrb[i] = drand48();
	for(size_t i=0; i<ni*nj; i++) ptrc_ref[i] = 0.0;

	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
	for(size_t p=0; p<np; p++) {
		ptrc_ref[i*nj+j] += ptra[i*np+p]*ptrb[j*np+p];
	}
	}
	}

	contraction2_processor<3>(list, ptrc, ptra, ptrb).contract();

	bool fail = false;
	size_t fail_pos;
	double fail_c, fail_cref;

	for(size_t i=0; i<ni*nj && !fail; i++) {
		if(!compare(ptrc_ref[i], ptrc[i])) {
			fail = true;
			fail_pos = i;
			fail_c = ptrc[i];
			fail_cref = ptrc_ref[i];
		}
	}

	delete [] ptra; delete [] ptrb; delete [] ptrc; delete [] ptrc_ref;

	if(fail) {
		char msg[128];
		snprintf(msg, 128, "Difference found at position %lu: "
			"%.5lg (act) vs. %.5lg (ref)",
			fail_pos, fail_c, fail_cref);
		fail_test("contraction2_processor_test::test_ij_p_ip_jp",
			__FILE__, __LINE__, msg);
	}
}

void contraction2_processor_test::test_ji_p_ip_jp(size_t ni, size_t nj,
	size_t np) throw(libtest::test_exception) {

	// c_{ji} = \sum_p a_{ip} b_{jp}
	// (matrix-matrix multiplication)

	contraction2_list<3> list;
	list.append(ni, np, 0, 1);
	list.append(nj, 0, np, ni);
	list.append(np, 1, 1, 0);

	double *ptra = new double[ni*np];
	double *ptrb = new double[nj*np];
	double *ptrc = new double[ni*nj];
	double *ptrc_ref = new double[ni*nj];

	for(size_t i=0; i<ni*np; i++) ptra[i] = drand48();
	for(size_t i=0; i<nj*np; i++) ptrb[i] = drand48();
	for(size_t i=0; i<ni*nj; i++) ptrc_ref[i] = 0.0;

	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
	for(size_t p=0; p<np; p++) {
		ptrc_ref[j*ni+i] += ptra[i*np+p]*ptrb[j*np+p];
	}
	}
	}

	contraction2_processor<3>(list, ptrc, ptra, ptrb).contract();

	bool fail = false;
	size_t fail_pos;
	double fail_c, fail_cref;

	for(size_t i=0; i<ni*nj && !fail; i++) {
		if(!compare(ptrc_ref[i], ptrc[i])) {
			fail = true;
			fail_pos = i;
			fail_c = ptrc[i];
			fail_cref = ptrc_ref[i];
		}
	}

	delete [] ptra; delete [] ptrb; delete [] ptrc; delete [] ptrc_ref;

	if(fail) {
		char msg[128];
		snprintf(msg, 128, "Difference found at position %lu: "
			"%.5lg (act) vs. %.5lg (ref)",
			fail_pos, fail_c, fail_cref);
		fail_test("contraction2_processor_test::test_ij_p_ip_jp",
			__FILE__, __LINE__, msg);
	}
}

void contraction2_processor_test::test_kij_p_ip_jkp(size_t ni, size_t nj,
	size_t nk, size_t np) throw(libtest::test_exception) {

	// c_{kij} = \sum_p a_{ip} b_{jkp}

	contraction2_list<4> list;
	list.append(nk, 0, np, ni*nj);
	list.append(ni, np, 0, nj);
	list.append(nj, 0, nk*np, 1);
	list.append(np, 1, 1, 0);

	double *ptra = new double[ni*np];
	double *ptrb = new double[nj*nk*np];
	double *ptrc = new double[nk*ni*nj];
	double *ptrc_ref = new double[nk*ni*nj];

	for(size_t i=0; i<ni*np; i++) ptra[i] = drand48();
	for(size_t i=0; i<nj*nk*np; i++) ptrb[i] = drand48();
	for(size_t i=0; i<nk*ni*nj; i++) ptrc_ref[i] = 0.0;

	for(size_t k=0; k<nk; k++) {
	for(size_t i=0; i<ni; i++) {
	for(size_t j=0; j<nj; j++) {
	for(size_t p=0; p<np; p++) {
		ptrc_ref[k*ni*nj+i*nj+j] += ptra[i*np+p]*ptrb[j*nk*np+k*np+p];
	}
	}
	}
	}

	contraction2_processor<4>(list, ptrc, ptra, ptrb).contract();

	bool fail = false;
	size_t fail_pos;
	double fail_c, fail_cref;

	for(size_t i=0; i<nk*ni*nj && !fail; i++) {
		if(!compare(ptrc_ref[i], ptrc[i])) {
			fail = true;
			fail_pos = i;
			fail_c = ptrc[i];
			fail_cref = ptrc_ref[i];
		}
	}

	delete [] ptra; delete [] ptrb; delete [] ptrc; delete [] ptrc_ref;

	if(fail) {
		char msg[128];
		snprintf(msg, 128, "Difference found at position %lu: "
			"%.5lg (act) vs. %.5lg (ref)",
			fail_pos, fail_c, fail_cref);
		fail_test("contraction2_processor_test::test_kij_p_ip_jkp",
			__FILE__, __LINE__, msg);
	}
}

void contraction2_processor_test::test_kji_p_ikp_jp(size_t ni, size_t nj,
	size_t nk, size_t np) throw(libtest::test_exception) {

	// c_{kji} = \sum_p a_{ikp} b_{jp}

	contraction2_list<4> list;
	list.append(nk, np, 0, nj*ni);
	list.append(nj, 0, np, ni);
	list.append(ni, nk*np, 0, 1);
	list.append(np, 1, 1, 0);

	size_t sza = ni*nk*np;
	size_t szb = nj*np;
	size_t szc = nk*nj*ni;
	double *ptra = new double[sza];
	double *ptrb = new double[szb];
	double *ptrc = new double[szc];
	double *ptrc_ref = new double[szc];

	for(size_t i=0; i<sza; i++) ptra[i] = drand48();
	for(size_t i=0; i<szb; i++) ptrb[i] = drand48();
	for(size_t i=0; i<szc; i++) ptrc_ref[i] = 0.0;

	for(size_t k=0; k<nk; k++) {
	for(size_t j=0; j<nj; j++) {
	for(size_t i=0; i<ni; i++) {
	for(size_t p=0; p<np; p++) {
		ptrc_ref[k*nj*ni+j*ni+i] += ptra[i*nk*np+k*np+p]*ptrb[j*np+p];
	}
	}
	}
	}

	contraction2_processor<4>(list, ptrc, ptra, ptrb).contract();

	bool fail = false;
	size_t fail_pos;
	double fail_c, fail_cref;

	for(size_t i=0; i<szc && !fail; i++) {
		if(!compare(ptrc_ref[i], ptrc[i])) {
			fail = true;
			fail_pos = i;
			fail_c = ptrc[i];
			fail_cref = ptrc_ref[i];
		}
	}

	delete [] ptra; delete [] ptrb; delete [] ptrc; delete [] ptrc_ref;

	if(fail) {
		char msg[128];
		snprintf(msg, 128, "Difference found at position %lu: "
			"%.5lg (act) vs. %.5lg (ref)",
			fail_pos, fail_c, fail_cref);
		fail_test("contraction2_processor_test::test_kji_p_ikp_jp",
			__FILE__, __LINE__, msg);
	}
}

} // namespace libtensor
