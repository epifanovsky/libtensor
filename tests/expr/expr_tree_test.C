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
}


namespace {

class tensor_i {
public:
    bool equals(const tensor_i &other) const {
        return this == &other;
    }
};

class tensor : public tensor_i {

};

} // unnamed namespace

using namespace expr;

void expr_tree_test::test_1() throw(libtest::test_exception) {

    //
    // Build an expr_tree which adds two tensors
    //

    static const char *testname = "expr_tree_test::test_1()";

    tensor t;
    tensor_i &ti = t;
    any_tensor<2, double> tt1(ti), tt2(ti), tt3(ti);

    std::multimap<size_t, size_t> map;
    map.insert(std::pair<size_t, size_t>(0, 1));
    map.insert(std::pair<size_t, size_t>(1, 0));

    expr_tree e(node_assign(2));
    expr_tree::node_id_t id = e.get_root();
    e.add(id, node_ident<2, double>(tt1));
    e.add(id, node_add(2, map));

    id = e.get_edges_out(id).back();
    e.add(id, node_ident<2, double>(tt2));
    e.add(id, node_ident<2, double>(tt3));

    id = e.get_root();
    const node &n1 = e.get_vertex(id);
    if (n1.get_op().compare(node_assign::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Op type (n1).");
    }
    if (n1.get_n() != 2) {
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
    if (n2a.get_op().compare(node_ident_base::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Op type (n2a).");
    }
    if (n2a.get_n() != 2) {
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
    if (n2b.get_op().compare(node_add::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Op type (n2b).");
    }
    if (n2b.get_n() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n2b).");
    }
    if (n2b.recast_as<node_add>().get_map().size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Addition map (n2b).");
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
    if (n3a.get_op().compare(node_ident_base::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Op type (n3a).");
    }
    if (n3a.get_n() != 2) {
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
    if (n3b.get_op().compare(node_ident_base::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Op type (n3b).");
    }
    if (n3b.get_n() != 2) {
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


void expr_tree_test::test_2() throw(libtest::test_exception) {

    //
    // Build tree from subtree(s)
    //

    static const char *testname = "expr_tree_test::test_2()";

    tensor t;
    tensor_i &ti = t;
    any_tensor<2, double> tt1(ti), tt2(ti), tt3(ti);

    std::multimap<size_t, size_t> map;
    map.insert(std::pair<size_t, size_t>(0, 1));
    map.insert(std::pair<size_t, size_t>(1, 0));

    expr_tree sub(node_add(2, map));
    expr_tree::node_id_t id = sub.get_root();
    sub.add(id, node_ident<2, double>(tt2));
    sub.add(id, node_ident<2, double>(tt3));

    expr_tree e(node_assign(2));
    id = e.get_root();
    e.add(id, node_ident<2, double>(tt1));
    e.add(id, sub);

    id = e.get_root();
    const node &n1 = e.get_vertex(id);
    if (n1.get_op().compare(node_assign::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Op type (n1).");
    }
    if (n1.get_n() != 2) {
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
    if (n2a.get_op().compare(node_ident_base::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Op type (n2a).");
    }
    if (n2a.get_n() != 2) {
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
    if (n2b.get_op().compare(node_add::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Op type (n2b).");
    }
    if (n2b.get_n() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Dim (n2b).");
    }
    if (n2b.recast_as<node_add>().get_map().size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Addition map (n2b).");
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
    if (n3a.get_op().compare(node_ident_base::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Op type (n3a).");
    }
    if (n3a.get_n() != 2) {
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
    if (n3b.get_op().compare(node_ident_base::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Op type (n3b).");
    }
    if (n3b.get_n() != 2) {
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


void expr_tree_test::test_3() throw(libtest::test_exception) {

    //
    // Insert node into tree
    //

    static const char *testname = "expr_tree_test::test_3()";
}


void expr_tree_test::test_4() throw(libtest::test_exception) {

    //
    // Erase subtree
    //

}


void expr_tree_test::test_5() throw(libtest::test_exception) {

    //
    // Move subtree to another part
    //

    static const char *testname = "expr_tree_test::test_5()";

}


void expr_tree_test::test_6() throw(libtest::test_exception) {

    //
    // Replace one subtree with another
    //

    static const char *testname = "expr_tree_test::test_6()";

}


} // namespace libtensor
