#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/expr/node_add.h>
#include <libtensor/expr/node_assign.h>
#include <libtensor/expr/node_contract.h>
#include <libtensor/expr/node_diag.h>
#include <libtensor/expr/node_div.h>
#include <libtensor/expr/node_ident.h>
#include <libtensor/expr/node_symm.h>
#include <libtensor/expr/node_transform.h>
#include <libtensor/expr/expr_tree.h>
#include "expr_tree_test.h"

namespace libtensor {


void expr_tree_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
    test_6();
    test_7();
}

using namespace expr;


namespace {

class test_node : public node {
public:
    static const char k_op_type[];

public:
    test_node(size_t n) : node(k_op_type, n) { }

    virtual ~test_node() { }

    virtual test_node *clone() const {
        return new test_node(*this);
    }
};

const char test_node::k_op_type[] = "test";

} // unnamed namespace


/** \brief Build a simple tree
 **/
void expr_tree_test::test_1() {

    static const char testname[] = "expr_tree_test::test_1()";

    expr_tree e(test_node(5));
    expr_tree::node_id_t id = e.get_root();
    e.add(id, test_node(2));
    id = e.add(id, test_node(3));
    expr_tree::node_id_t id2 = e.add(id, test_node(4));
    e.add(id2, test_node(1));
    e.add(id, test_node(6));

    id = e.get_root();
    const node &n1 = e.get_vertex(id);
    if (n1.get_n() != 5) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n1).");
    }
    const expr_tree::edge_list_t &l1_in  = e.get_edges_in(id);
    if (l1_in.size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n1).");
    }
    const expr_tree::edge_list_t &l1_out = e.get_edges_out(id);
    if (l1_out.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n1).");
    }

    const node &n2a = e.get_vertex(l1_out[0]);
    if (n2a.get_n() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n2a).");
    }
    const expr_tree::edge_list_t &l2a_in  = e.get_edges_in(l1_out[0]);
    if (l2a_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n2a).");
    }
    const expr_tree::edge_list_t &l2a_out = e.get_edges_out(l1_out[0]);
    if (l2a_out.size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n2a).");
    }

    const node &n2b = e.get_vertex(l1_out[1]);
    if (n2b.get_n() != 3) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n2b).");
    }
    const expr_tree::edge_list_t &l2b_in  = e.get_edges_in(l1_out[1]);
    if (l2b_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n2b).");
    }
    const expr_tree::edge_list_t &l2b_out = e.get_edges_out(l1_out[1]);
    if (l2b_out.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n2b).");
    }

    const node &n3a = e.get_vertex(l2b_out[0]);
    if (n3a.get_n() != 4) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n3a).");
    }
    const expr_tree::edge_list_t &l3a_in  = e.get_edges_in(l2b_out[0]);
    if (l3a_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n3a).");
    }
    const expr_tree::edge_list_t &l3a_out = e.get_edges_out(l2b_out[0]);
    if (l3a_out.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n3a).");
    }

    const node &n3b = e.get_vertex(l2b_out[1]);
    if (n3b.get_n() != 6) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n3b).");
    }
    const expr_tree::edge_list_t &l3b_in  = e.get_edges_in(l2b_out[1]);
    if (l3b_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n3b).");
    }
    const expr_tree::edge_list_t &l3b_out = e.get_edges_out(l2b_out[1]);
    if (l3b_out.size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n3b).");
    }

    const node &n4 = e.get_vertex(l3a_out[0]);
    if (n4.get_n() != 1) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n4).");
    }
    const expr_tree::edge_list_t &l4_in  = e.get_edges_in(l3a_out[0]);
    if (l4_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n4).");
    }
    const expr_tree::edge_list_t &l4_out = e.get_edges_out(l3a_out[0]);
    if (l4_out.size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n4).");
    }

}


/** \brief Build tree from subtree(s)
 **/
void expr_tree_test::test_2() {

    static const char testname[] = "expr_tree_test::test_2()";

    expr_tree sub(test_node(2));
    expr_tree::node_id_t id = sub.get_root();
    sub.add(id, test_node(1));
    sub.add(id, test_node(3));

    expr_tree e(test_node(4));
    id = e.get_root();
    e.add(id, test_node(6));
    e.add(id, sub);

    id = e.get_root();
    const node &n1 = e.get_vertex(id);
    if (n1.get_n() != 4) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n1).");
    }
    const expr_tree::edge_list_t &l1_in  = e.get_edges_in(id);
    if (l1_in.size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n1).");
    }
    const expr_tree::edge_list_t &l1_out = e.get_edges_out(id);
    if (l1_out.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n1).");
    }

    id = l1_out[0];
    const node &n2a = e.get_vertex(id);
    if (n2a.get_n() != 6) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n2a).");
    }
    const expr_tree::edge_list_t &l2a_in  = e.get_edges_in(id);
    if (l2a_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n2a).");
    }
    const expr_tree::edge_list_t &l2a_out = e.get_edges_out(id);
    if (l2a_out.size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n2a).");
    }

    id = l1_out[1];
    const node &n2b = e.get_vertex(id);
    if (n2b.get_n() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n2b).");
    }
    const expr_tree::edge_list_t &l2b_in  = e.get_edges_in(id);
    if (l2b_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n2b).");
    }
    const expr_tree::edge_list_t &l2b_out = e.get_edges_out(id);
    if (l2b_out.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n2b).");
    }

    id = l2b_out[0];
    const node &n3a = e.get_vertex(id);
    if (n3a.get_n() != 1) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n3a).");
    }
    const expr_tree::edge_list_t &l3a_in  = e.get_edges_in(id);
    if (l3a_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n3a).");
    }
    const expr_tree::edge_list_t &l3a_out = e.get_edges_out(id);
    if (l3a_out.size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n3a).");
    }

    id = l2b_out[1];
    const node &n3b = e.get_vertex(id);
    if (n3b.get_n() != 3) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n3b).");
    }
    const expr_tree::edge_list_t &l3b_in  = e.get_edges_in(id);
    if (l3b_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n3b).");
    }
    const expr_tree::edge_list_t &l3b_out = e.get_edges_out(id);
    if (l3b_out.size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n3b).");
    }
}


/** \brief Insert node into tree
 **/
void expr_tree_test::test_3() {

    static const char testname[] = "expr_tree_test::test_3()";

    expr_tree e(test_node(1));
    expr_tree::node_id_t id1 = e.get_root();
    expr_tree::node_id_t id2 = e.add(id1, test_node(2));
    expr_tree::node_id_t id3 = e.add(id1, test_node(3));
    expr_tree::node_id_t id4 = e.add(id3, test_node(4));
    e.graph::add(id2, id4);
    e.add(id3, test_node(5));
    e.add(id4, test_node(6));

    e.insert(id4, test_node(7));

    expr_tree::node_id_t id = e.get_root();
    const node &n1 = e.get_vertex(id);
    if (n1.get_n() != 1) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n1).");
    }
    const expr_tree::edge_list_t &l1 = e.get_edges_out(id);
    if (l1.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n1).");
    }
    const node &n2 = e.get_vertex(l1[0]);
    if (n2.get_n() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n2).");
    }
    const node &n3 = e.get_vertex(l1[1]);
    if (n3.get_n() != 3) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n3).");
    }

    const expr_tree::edge_list_t &l2 = e.get_edges_out(l1[0]);
    if (l2.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n2).");
    }
    const node &n7a = e.get_vertex(l2[0]);
    if (n7a.get_n() != 7) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n7a).");
    }

    const expr_tree::edge_list_t &l3 = e.get_edges_out(l1[1]);
    if (l3.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n3).");
    }
    const node &n7b = e.get_vertex(l3[0]);
    if (n7b.get_n() != 7) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n7b).");
    }
    const node &n5 = e.get_vertex(l3[1]);
    if (n5.get_n() != 5) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n5).");
    }

    const expr_tree::edge_list_t &l7  = e.get_edges_out(l2[0]);
    if (l7.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n7).");
    }
    const node &n4 = e.get_vertex(l7[0]);
    if (n4.get_n() != 4) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n4).");
    }
}


/** \brief Erase subtree
 **/
void expr_tree_test::test_4() {

    static const char testname[] = "expr_tree_test::test_4()";

    expr_tree e(test_node(1));
    expr_tree::node_id_t id1 = e.get_root();
    expr_tree::node_id_t id2 = e.add(id1, test_node(2));
    expr_tree::node_id_t id3 = e.add(id1, test_node(3));
    expr_tree::node_id_t id4 = e.add(id2, test_node(4));
    expr_tree::node_id_t id5 = e.add(id2, test_node(5));
    e.graph::add(id3, id5);
    e.add(id3, test_node(6));
    e.add(id4, test_node(7));
    e.add(id4, test_node(8));

    e.erase_subtree(id4);
    if (e.get_n_vertexes() != 5) {
        fail_test(testname, __FILE__, __LINE__, "Wrong # vertexes (1).");
    }

    e.erase_subtree(id3);
    if (e.get_n_vertexes() != 3) {
        fail_test(testname, __FILE__, __LINE__, "Wrong # vertexes (2).");
    }

    expr_tree::node_id_t id = e.get_root();
    const node &n1 = e.get_vertex(id);
    if (n1.get_n() != 1) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n1).");
    }
    const expr_tree::edge_list_t &l1 = e.get_edges_out(id);
    if (l1.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n1).");
    }
    const node &n2 = e.get_vertex(l1[0]);
    if (n2.get_n() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n2).");
    }
    const expr_tree::edge_list_t &l2 = e.get_edges_out(l1[0]);
    if (l2.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n2).");
    }
    const node &n5 = e.get_vertex(l2[0]);
    if (n5.get_n() != 5) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n5).");
    }
}


/** \brief Move subtree to another part
**/
void expr_tree_test::test_5() {

    static const char testname[] = "expr_tree_test::test_5()";

    expr_tree e(test_node(1));
    expr_tree::node_id_t id1 = e.get_root();
    expr_tree::node_id_t id2 = e.add(id1, test_node(2));
    expr_tree::node_id_t id3 = e.add(id1, test_node(3));
    expr_tree::node_id_t id4 = e.add(id2, test_node(4));
    expr_tree::node_id_t id5 = e.add(id2, test_node(5));
    e.graph::add(id3, id5);
    e.add(id3, test_node(6));
    e.add(id4, test_node(7));
    e.add(id4, test_node(8));

    e.move(id4, id3);
    if (e.get_n_vertexes() != 8) {
        fail_test(testname, __FILE__, __LINE__, "Wrong # vertexes.");
    }

    expr_tree::node_id_t id = e.get_root();
    const node &n1 = e.get_vertex(id);
    if (n1.get_n() != 1) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n1).");
    }
    const expr_tree::edge_list_t &l1 = e.get_edges_out(id);
    if (l1.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n1).");
    }
    const node &n2 = e.get_vertex(l1[0]);
    if (n2.get_n() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n2).");
    }
    const expr_tree::edge_list_t &l2 = e.get_edges_out(l1[0]);
    if (l2.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n2).");
    }
    const node &n5a = e.get_vertex(l2[0]);
    if (n5a.get_n() != 5) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n5).");
    }
    const node &n3 = e.get_vertex(l1[1]);
    if (n3.get_n() != 3) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n3).");
    }
    const expr_tree::edge_list_t &l3 = e.get_edges_out(l1[1]);
    if (l3.size() != 3) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n3).");
    }
    const node &n5b = e.get_vertex(l3[0]);
    if (n5b.get_n() != 5) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n5).");
    }
    const node &n6 = e.get_vertex(l3[1]);
    if (n6.get_n() != 6) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n6).");
    }
    const node &n4 = e.get_vertex(l3[2]);
    if (n4.get_n() != 4) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n4).");
    }
    const expr_tree::edge_list_t &l4 = e.get_edges_out(l3[2]);
    if (l4.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n4).");
    }
}


/** \brief Replace one subtree with another
 **/
void expr_tree_test::test_6() {

    static const char testname[] = "expr_tree_test::test_6()";

    expr_tree e(test_node(1));
    expr_tree::node_id_t id1 = e.get_root();
    expr_tree::node_id_t id2 = e.add(id1, test_node(2));
    expr_tree::node_id_t id3 = e.add(id1, test_node(3));
    expr_tree::node_id_t id4 = e.add(id2, test_node(4));
    expr_tree::node_id_t id5 = e.add(id2, test_node(5));
    e.graph::add(id3, id5);
    e.add(id3, test_node(6));
    e.add(id4, test_node(7));
    e.add(id4, test_node(8));

    e.replace(id3, id4);
    if (e.get_n_vertexes() != 6) {
        fail_test(testname, __FILE__, __LINE__, "Wrong # vertexes.");
    }

    expr_tree::node_id_t id = e.get_root();
    const node &n1 = e.get_vertex(id);
    if (n1.get_n() != 1) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n1).");
    }
    const expr_tree::edge_list_t &l1 = e.get_edges_out(id);
    if (l1.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n1).");
    }
    const node &n2 = e.get_vertex(l1[0]);
    if (n2.get_n() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n2).");
    }
    const expr_tree::edge_list_t &l2 = e.get_edges_out(l1[0]);
    if (l2.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n2).");
    }
    const node &n4a = e.get_vertex(l2[0]);
    if (n4a.get_n() != 4) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n4).");
    }
    const node &n5 = e.get_vertex(l2[1]);
    if (n5.get_n() != 5) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n5).");
    }
    const node &n4b = e.get_vertex(l1[1]);
    if (n4b.get_n() != 4) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n4).");
    }
    const expr_tree::edge_list_t &l4 = e.get_edges_out(l1[1]);
    if (l4.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n4).");
    }
}


/** \brief Test the copy constructor
 **/
void expr_tree_test::test_7() {

    static const char testname[] = "expr_tree_test::test_7()";

    expr_tree e0(test_node(5));
    expr_tree::node_id_t id = e0.get_root();
    e0.add(id, test_node(2));
    id = e0.add(id, test_node(3));
    expr_tree::node_id_t id2 = e0.add(id, test_node(4));
    e0.add(id2, test_node(1));
    e0.add(id, test_node(6));

    expr_tree e(e0);

    id = e.get_root();
    const node &n1 = e.get_vertex(id);
    if (n1.get_n() != 5) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n1).");
    }
    const expr_tree::edge_list_t &l1_in  = e.get_edges_in(id);
    if (l1_in.size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n1).");
    }
    const expr_tree::edge_list_t &l1_out = e.get_edges_out(id);
    if (l1_out.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n1).");
    }

    const node &n2a = e.get_vertex(l1_out[0]);
    if (n2a.get_n() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n2a).");
    }
    const expr_tree::edge_list_t &l2a_in  = e.get_edges_in(l1_out[0]);
    if (l2a_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n2a).");
    }
    const expr_tree::edge_list_t &l2a_out = e.get_edges_out(l1_out[0]);
    if (l2a_out.size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n2a).");
    }

    const node &n2b = e.get_vertex(l1_out[1]);
    if (n2b.get_n() != 3) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n2b).");
    }
    const expr_tree::edge_list_t &l2b_in  = e.get_edges_in(l1_out[1]);
    if (l2b_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n2b).");
    }
    const expr_tree::edge_list_t &l2b_out = e.get_edges_out(l1_out[1]);
    if (l2b_out.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n2b).");
    }

    const node &n3a = e.get_vertex(l2b_out[0]);
    if (n3a.get_n() != 4) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n3a).");
    }
    const expr_tree::edge_list_t &l3a_in  = e.get_edges_in(l2b_out[0]);
    if (l3a_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n3a).");
    }
    const expr_tree::edge_list_t &l3a_out = e.get_edges_out(l2b_out[0]);
    if (l3a_out.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n3a).");
    }

    const node &n3b = e.get_vertex(l2b_out[1]);
    if (n3b.get_n() != 6) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n3b).");
    }
    const expr_tree::edge_list_t &l3b_in  = e.get_edges_in(l2b_out[1]);
    if (l3b_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n3b).");
    }
    const expr_tree::edge_list_t &l3b_out = e.get_edges_out(l2b_out[1]);
    if (l3b_out.size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n3b).");
    }

    const node &n4 = e.get_vertex(l3a_out[0]);
    if (n4.get_n() != 1) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n4).");
    }
    const expr_tree::edge_list_t &l4_in  = e.get_edges_in(l3a_out[0]);
    if (l4_in.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# parent nodes (n4).");
    }
    const expr_tree::edge_list_t &l4_out = e.get_edges_out(l3a_out[0]);
    if (l4_out.size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "# child nodes (n4).");
    }

}


} // namespace libtensor
