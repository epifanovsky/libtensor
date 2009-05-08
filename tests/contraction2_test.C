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

	// Dimensions i=3 j=4 k=3 l=4 p=5 q=6
	size_t ni = 3, nj = 4, nk = 3, nl = 4, np = 5, nq = 6;
	index<4> idxa1_1, idxa1_2, idxb1_1, idxb1_2;
	idxa1_2[0]=ni-1; idxa1_2[1]=nj-1; idxa1_2[2]=np-1; idxa1_2[3]=nq-1;
	idxb1_2[0]=nk-1; idxb1_2[1]=nl-1; idxb1_2[2]=np-1; idxb1_2[3]=nq-1;
	index_range<4> ira1(idxa1_1, idxa1_2), irb1(idxb1_1, idxb1_2);
	dimensions<4> dima1(ira1), dimb1(irb1);

	contraction2_list<6> list;
	c.populate(list, dima1, dimb1);

	size_t nodes = list.get_length();
	if(nodes != 3) {
		char s[128];
		snprintf(s, 128, "Incorrect number of fused nodes: "
			"%lu (returned), %lu (expected)", nodes, 3);
		fail_test("contraction2_test::test_1()", __FILE__, __LINE__, s);
	}

	size_t node[3];
	node[0] = list.get_first();
	node[1] = list.get_next(node[0]);
	node[2] = list.get_next(node[1]);

	size_t weight[3], inca[3], incb[3], incc[3];
	for(size_t i=0; i<3; i++) {
		weight[i] = list.get_node(node[i]).m_weight;
		inca[i] = list.get_node(node[i]).m_inca;
		incb[i] = list.get_node(node[i]).m_incb;
		incc[i] = list.get_node(node[i]).m_incc;
	}
	size_t weight_ref[3] = { ni*nj, nk*nl, np*nq };
	size_t inca_ref[3] = { np*nq, 0, 1 };
	size_t incb_ref[3] = { 0, np*nq, 1 };
	size_t incc_ref[3] = { nk*nl, 1, 0 };

	for(size_t i=0; i<3; i++) {
		if(weight[i] != weight_ref[i]) {
			char s[128];
			snprintf(s, 128, "Incorrect weight of node %lu: "
				"%lu (returned), %lu (expected)",
				i, weight[i], weight_ref[i]);
			fail_test("contraction2_test::test_1()",
				__FILE__, __LINE__, s);
		}
		if(inca[i] != inca_ref[i]) {
			char s[128];
			snprintf(s, 128, "Incorrect inc(a) of node %lu: "
				"%lu (returned), %lu (expected)",
				i, inca[i], inca_ref[i]);
			fail_test("contraction2_test::test_1()",
				__FILE__, __LINE__, s);
		}
		if(incb[i] != incb_ref[i]) {
			char s[128];
			snprintf(s, 128, "Incorrect inc(b) of node %lu: "
				"%lu (returned), %lu (expected)",
				i, incb[i], incb_ref[i]);
			fail_test("contraction2_test::test_1()",
				__FILE__, __LINE__, s);
		}
		if(incc[i] != incc_ref[i]) {
			char s[128];
			snprintf(s, 128, "Incorrect inc(c) of node %lu: "
				"%lu (returned), %lu (expected)",
				i, incc[i], incc_ref[i]);
			fail_test("contraction2_test::test_1()",
				__FILE__, __LINE__, s);
		}
	}
}

} // namespace libtensor
