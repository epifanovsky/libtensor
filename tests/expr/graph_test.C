#include <algorithm>
#include <libtensor/exception.h>
#include <libtensor/expr/graph.h>
#include "graph_test.h"

namespace libtensor {


void graph_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
}


using namespace expr;

namespace {

class test_node : public node {
public:
    static const char k_op_type[];

public:
    test_node(size_t n) : node(k_op_type, n) { }

    virtual test_node *clone() const {
        return new test_node(*this);
    }
};

const char test_node::k_op_type[] = "test";

}

/** \brief Tests adding and deleting vertexes and edges
 **/
void graph_test::test_1() {

    static const char testname[] = "graph_test::test_1()";

    try {

    graph g;

    test_node n1(1), n2(2), n3(3), n4(4);
    std::vector<graph::node_id_t> ids(4, 0);
    ids[0] = g.add(n1);
    ids[1] = g.add(n2);
    ids[2] = g.add(n3);
    ids[3] = g.add(n4);

    g.add(ids[0], ids[1]);
    g.add(ids[0], ids[2]);
    g.add(ids[1], ids[0]);
    g.add(ids[1], ids[2]);
    g.add(ids[1], ids[3]);
    g.add(ids[2], ids[3]);
    g.add(ids[3], ids[0]);

    if(g.get_n_vertexes() != 4) {
        fail_test(testname, __FILE__, __LINE__, "Wrong # nodes.");
    }

    for(graph::iterator i = g.begin(); i != g.end(); ++i) {

        const node &item = g.get_vertex(i);
        const graph::edge_list_t &in = g.get_edges_in(i);
        const graph::edge_list_t &out = g.get_edges_out(i);

        if (item.get_op().compare(test_node::k_op_type) != 0) {
            fail_test(testname, __FILE__, __LINE__, "Wrong node type.");
        }
        if (item.get_n() == 1) {
            if (g.get_id(i) != ids[0]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong id (1).");
            }
            if (in.size() != 2) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (1).");
            }
            if (in[0] != ids[1]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (1<-2).");
            }
            if (in[1] != ids[3]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (1<-4).");
            }
            if (out.size() != 2) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (1).");
            }
            if (out[0] != ids[1]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (1->2).");
            }
            if (out[1] != ids[2]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (1->3).");
            }
        }
        else if (item.get_n() == 2) {
            if (g.get_id(i) != ids[1]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong id (2).");
            }
            if (in.size() != 1) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (2).");
            }
            if (in[0] != ids[0]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (2<-1).");
            }
            if (out.size() != 3) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (2).");
            }
            if (out[0] != ids[0]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (2->1).");
            }
            if (out[1] != ids[2]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (2->3).");
            }
            if (out[2] != ids[3]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (2->4).");
            }
        }
        else if (item.get_n() == 3) {
            if (g.get_id(i) != ids[2]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong id (3).");
            }
            if (in.size() != 2) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (3).");
            }
            if (in[0] != ids[0]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (3<-1).");
            }
            if (in[1] != ids[1]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (3<-2).");
            }
            if (out.size() != 1) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (1).");
            }
            if (out[0] != ids[3]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (3->4).");
            }
        }
        else if (item.get_n() == 4) {
            if (g.get_id(i) != ids[3]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong id (4).");
            }
            if (in.size() != 2) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (4).");
            }
            if (in[0] != ids[1]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (4<-2).");
            }
            if (in[1] != ids[2]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (4<-3).");
            }
            if (out.size() != 1) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (4).");
            }
            if (out[0] != ids[0]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (4->1).");
            }
        }
        else {
            fail_test(testname, __FILE__, __LINE__, "Unknown id.");
        }
    }

    g.erase(ids[3]);
    g.erase(ids[1], ids[0]);
    if(g.get_n_vertexes() != 3) {
        fail_test(testname, __FILE__, __LINE__, "Wrong # nodes.");
    }

    for(graph::iterator i = g.begin(); i != g.end(); ++i) {

        const node &item = g.get_vertex(i);
        const graph::edge_list_t &in = g.get_edges_in(i);
        const graph::edge_list_t &out = g.get_edges_out(i);

        if (item.get_op().compare(test_node::k_op_type) != 0) {
            fail_test(testname, __FILE__, __LINE__, "Wrong node type.");
        }
        if (item.get_n() == 1) {
            if (g.get_id(i) != ids[0]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong id (1).");
            }
            if (in.size() != 0) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (1).");
            }
            if (out.size() != 2) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (1).");
            }
            if (out[0] != ids[1]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (1->2).");
            }
            if (out[1] != ids[2]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (1->3).");
            }
        }
        else if (item.get_n() == 2) {
            if (g.get_id(i) != ids[1]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong id (2).");
            }
            if (in.size() != 1) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (2).");
            }
            if (in[0] != ids[0]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (2<-1).");
            }
            if (out.size() != 1) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (2).");
            }
            if (out[0] != ids[2]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (2->3).");
            }
        }
        else if (item.get_n() == 3) {
            if (g.get_id(i) != ids[2]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong id (3).");
            }
            if (in.size() != 2) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (3).");
            }
            if (in[0] != ids[0]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (3<-1).");
            }
            if (in[1] != ids[1]) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (3<-2).");
            }
            if (out.size() != 0) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (1).");
            }
        }
        else {
            fail_test(testname, __FILE__, __LINE__, "Unknown id.");
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \brief Tests the connectivity information
 **/
void graph_test::test_2() {

    static const char testname[] = "graph_test::test_2()";

    try {

    graph g;

    test_node n1(1), n2(2), n3(3), n4(4);
    std::vector<graph::node_id_t> ids(4, 0);
    ids[0] = g.add(n1);
    ids[1] = g.add(n2);
    ids[2] = g.add(n3);
    ids[3] = g.add(n4);

    g.add(ids[0], ids[1]);
    g.add(ids[0], ids[2]);
    g.add(ids[1], ids[2]);
    g.add(ids[2], ids[3]);
    g.add(ids[3], ids[1]);

    if (! is_connected(ids[0], ids[1])) {
        fail_test(testname, __FILE__, __LINE__, "No connection (1->2).");
    }
    if (! is_connected(ids[0], ids[2])) {
        fail_test(testname, __FILE__, __LINE__, "No connection (1->3).");
    }
    if (! is_connected(ids[0], ids[3])) {
        fail_test(testname, __FILE__, __LINE__, "No connection (1->4).");
    }
    if (is_connected(ids[1], ids[0])) {
        fail_test(testname, __FILE__, __LINE__, "Connection (2->1).");
    }
    if (! is_connected(ids[1], ids[2])) {
        fail_test(testname, __FILE__, __LINE__, "No connection (2->3).");
    }
    if (! is_connected(ids[1], ids[3])) {
        fail_test(testname, __FILE__, __LINE__, "No connection (2->4).");
    }
    if (is_connected(ids[2], ids[0])) {
        fail_test(testname, __FILE__, __LINE__, "Connection (3->1).");
    }
    if (! is_connected(ids[2], ids[1])) {
        fail_test(testname, __FILE__, __LINE__, "No connection (3->2).");
    }
    if (! is_connected(ids[2], ids[3])) {
        fail_test(testname, __FILE__, __LINE__, "No connection (3->4).");
    }
    if (is_connected(ids[3], ids[0])) {
        fail_test(testname, __FILE__, __LINE__, "Connection (4->1).");
    }
    if (! is_connected(ids[3], ids[1])) {
        fail_test(testname, __FILE__, __LINE__, "No connection (4->2).");
    }
    if (! is_connected(ids[3], ids[2])) {
        fail_test(testname, __FILE__, __LINE__, "No connection (4->3).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
