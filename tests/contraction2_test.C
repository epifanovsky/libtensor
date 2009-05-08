#include <libtensor.h>
#include "contraction2_test.h"

namespace libtensor {

void contraction2_test::perform() throw(libtest::test_exception) {
	test_1();
}

void contraction2_test::test_1() throw(libtest::test_exception) {

	// Testing c[ijkl] = sum[pq] a[ijpq] b[klpq]
	//
	// Number of fused nodes = 3:
	// c[ab] = sum[z] a[az] b[bz]
	// a={ij} b={kl} z={pq}

	permutation<4> perm;
	contraction2<2,2,2> c(perm);

	if(c.is_complete()) {
		fail_test("contraction2_test::test_1()", __FILE__, __LINE__,
			"Empty contraction declares complete");
	}

	c.contract(2, 2);
	if(c.is_complete()) {
		fail_test("contraction2_test::test_1()", __FILE__, __LINE__,
			"Incomplete contraction declares complete");
	}

	c.contract(3, 3);
	if(!c.is_complete()) {
		fail_test("contraction2_test::test_1()", __FILE__, __LINE__,
			"Complete contraction declares incomplete");
	}

	size_t nodes = c.get_num_nodes();
	if(nodes != 3) {
		char s[128];
		snprintf(s, 128, "Incorrect number of fused nodes: "
			"%lu (returned), %lu (expected)", nodes, 3);
		fail_test("contraction2_test::test_1()", __FILE__, __LINE__, s);
	}

	// Dimensions i=3 j=4 k=3 l=4 p=5 q=6
	size_t ni = 3, nj = 4, nk = 3, nl = 4, np = 5, nq = 6;
	index<4> idxa1_1, idxa1_2, idxb1_1, idxb1_2;
	idxa1_2[0]=ni-1; idxa1_2[1]=nj-1; idxa1_2[2]=np-1; idxa1_2[3]=nq-1;
	idxb1_2[0]=nk-1; idxb1_2[1]=nl-1; idxb1_2[2]=np-1; idxb1_2[3]=nq-1;
	index_range<4> ira1(idxa1_1, idxa1_2), irb1(idxb1_1, idxb1_2);
	dimensions<4> dima1(ira1), dimb1(irb1);

	size_t weight0 = c.get_weight(0, dima1, dimb1);
	size_t weight1 = c.get_weight(1, dima1, dimb1);
	size_t weight2 = c.get_weight(2, dima1, dimb1);
	if(weight0 != ni*nj) {
		char s[128];
		snprintf(s, 128, "Incorrect weight of node 0: "
			"%lu (returned), %lu (expected)", weight0, ni*nj);
		fail_test("contraction2_test::test_1()", __FILE__, __LINE__, s);
	}
	if(weight1 != nk*nl) {
		char s[128];
		snprintf(s, 128, "Incorrect weight of node 1: "
			"%lu (returned), %lu (expected)", weight1, nk*nl);
		fail_test("contraction2_test::test_1()", __FILE__, __LINE__, s);
	}
	if(weight2 != np*nq) {
		char s[128];
		snprintf(s, 128, "Incorrect weight of node 2: "
			"%lu (returned), %lu (expected)", weight2, np*nq);
		fail_test("contraction2_test::test_1()", __FILE__, __LINE__, s);
	}
}

} // namespace libtensor
