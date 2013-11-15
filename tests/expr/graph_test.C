#include <algorithm>
#include <libtensor/exception.h>
#include <libtensor/expr/graph.h>
#include <libtensor/expr/node_div.h>
#include "graph_test.h"

namespace libtensor {


void graph_test::perform() throw(libtest::test_exception) {

    test_1();
}


using namespace expr;


void graph_test::test_1() {

    static const char testname[] = "graph_test::test_1()";

    try {

    graph g;

    node_div d1(1), d2(2), d3(3);
    graph::node_id_t id1 = g.add(d1);
    graph::node_id_t id2 = g.add(d2);
    graph::node_id_t id3 = g.add(d3);

    g.add(id1, id2);
    g.add(id1, id3);
    g.add(id2, id3);
    g.add(id2, id1);

    if(g.get_n_vertexes() != 3) {
        fail_test(testname, __FILE__, __LINE__, "Wrong # nodes.");
    }

    for(graph::iterator i = g.begin(); i != g.end(); ++i) {

        const node &item = g.get_vertex(i);
        const graph::edge_list_t &in = g.get_edges_in(i);
        const graph::edge_list_t &out = g.get_edges_out(i);

        if (item.get_op().compare(node_div::k_op_type) != 0) {
            fail_test(testname, __FILE__, __LINE__, "Wrong node type.");
        }
        if (item.get_n() == 1) {
            if (g.get_id(i) != id1) {
                fail_test(testname, __FILE__, __LINE__, "Wrong id (1).");
            }
            if (in.size() != 1) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (in).");
            }
            if (in[0] != id2) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (2->1).");
            }
            if (out.size() != 2) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (out).");
            }
            if (out[0] != id2) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (1->2).");
            }
            if (out[1] != id3) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (1->3).");
            }
        }
        else if (item.get_n() == 2) {
            if (g.get_id(i) != id2) {
                fail_test(testname, __FILE__, __LINE__, "Wrong id (2).");
            }
            if (in.size() != 1) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (in).");
            }
            if (in[0] != id1) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (1->2).");
            }
            if (out.size() != 2) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (out).");
            }
            if (out[0] != id3) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (2->3).");
            }
            if (out[1] != id1) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (2->1).");
            }
        }
        else if (item.get_n() == 3) {
            if (g.get_id(i) != id3) {
                fail_test(testname, __FILE__, __LINE__, "Wrong id (3).");
            }
            if (in.size() != 2) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (in).");
            }
            if (in[0] != id1) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (1->3).");
            }
            if (in[1] != id2) {
                fail_test(testname, __FILE__, __LINE__, "Wrong edge (2->3).");
            }
            if (out.size() != 0) {
                fail_test(testname, __FILE__, __LINE__, "Wrong # edges (out).");
            }
        }
        else {
            fail_test(testname, __FILE__, __LINE__, "Unknown id.");
        }
    }

    g.erase(id2);
    if(g.get_n_vertexes() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Wrong # nodes.");
    }

    const graph::edge_list_t &in1 = g.get_edges_in(id1);
    if (in1.size() != 0) {
        fail_test(testname, __FILE__, __LINE__, "Wrong # edges (in).");
    }
    const graph::edge_list_t &out1 = g.get_edges_out(id1);
    if (out1.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong # edges (out).");
    }
    if (out1[0] != id3) {
        fail_test(testname, __FILE__, __LINE__, "Wrong edge (1->3).");
    }

    const graph::edge_list_t &in3 = g.get_edges_in(id3);
    if (in3.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong # edges (out).");
    }
    if (in3[0] != id1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong edge (1->3).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
