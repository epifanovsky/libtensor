#include <libtensor/exception.h>
#include <libtensor/symmetry/adjacency_list.h>
#include "adjacency_list_test.h"


namespace libtensor {


void adjacency_list_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
}


/** \test Tests adding and removing edges
 **/
void adjacency_list_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "adjacency_list_test::test_1()";

    try {

        adjacency_list alst;
        alst.add(0, 1);
        alst.add(1, 0, 2);
        alst.add(1, 4, 2);

        if (! alst.exist(0, 1)) {
            fail_test(testname, __FILE__, __LINE__, "Edge 0-1");
        }
        if (! alst.exist(1, 4)) {
            fail_test(testname, __FILE__, __LINE__, "Edge 1-4");
        }
        if (alst.exist(0, 4)) {
            fail_test(testname, __FILE__, __LINE__, "Edge 0-4");
        }


        if (alst.weight(0, 1) != 3) {
            fail_test(testname, __FILE__, __LINE__, "Weight 0-1");
        }
        if (alst.weight(1, 4) != 2) {
            fail_test(testname, __FILE__, __LINE__, "Weight 1-4");
        }
        if (alst.weight(0, 4) != 0) {
            fail_test(testname, __FILE__, __LINE__, "Weight 0-4");
        }

        alst.erase(0, 1);
        if (alst.exist(0, 1)) {
            fail_test(testname, __FILE__, __LINE__, "Edge 0-1");
        }
        if (alst.weight(0, 1) != 0) {
            fail_test(testname, __FILE__, __LINE__, "Weight 0-1");
        }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Test retrieving neighbours
 **/
void adjacency_list_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "adjacency_list_test::test_2()";

    try {

        adjacency_list alst;
        alst.add(0, 1);
        alst.add(0, 2, 2);
        alst.add(0, 3);
        alst.add(0, 6);
        alst.add(1, 3, 3);
        alst.add(1, 4, 2);
        alst.add(2, 3);
        alst.add(2, 6);
        alst.add(3, 4);
        alst.add(3, 5);
        alst.add(3, 6);
        alst.add(5, 6);

        std::vector<size_t> prev0, prev3, prev6;
        alst.get_prev_neighbours(0, prev0);
        if (prev0.size() != 0) {
            fail_test(testname, __FILE__, __LINE__, "# prev(0)");
        }

        alst.get_prev_neighbours(3, prev3);
        if (prev3.size() != 3) {
            fail_test(testname, __FILE__, __LINE__, "# prev(3)");
        }
        if (prev3[0] != 0 || prev3[1] != 1 || prev3[2] != 2) {
            fail_test(testname, __FILE__, __LINE__, "prev(3)");
        }

        alst.get_prev_neighbours(6, prev6);
        if (prev6.size() != 4) {
            fail_test(testname, __FILE__, __LINE__, "# prev(6)");
        }
        if (prev6[0] != 0 || prev6[1] != 2 || prev6[2] != 3 || prev6[3] != 5) {
            fail_test(testname, __FILE__, __LINE__, "prev(6)");
        }

        std::vector<size_t> next2, next4, next5;
        alst.get_next_neighbours(2, next2);
        if (next2.size() != 2) {
            fail_test(testname, __FILE__, __LINE__, "# next(2)");
        }
        if (next2[0] != 3 || next2[1] != 6) {
            fail_test(testname, __FILE__, __LINE__, "next(2)");
        }

        alst.get_next_neighbours(4, next4);
        if (next4.size() != 0) {
            fail_test(testname, __FILE__, __LINE__, "# next(4)");
        }

        alst.get_next_neighbours(5, next5);
        if (next5.size() != 1) {
            fail_test(testname, __FILE__, __LINE__, "# next(5)");
        }
        if (next5[0] != 6) {
            fail_test(testname, __FILE__, __LINE__, "next(5)");
        }


        std::vector<size_t> nb2, nb4, nb7;
        alst.get_neighbours(2, nb2);
        if (nb2.size() != 3) {
            fail_test(testname, __FILE__, __LINE__, "# neighbours(2)");
        }
        if (nb2[0] != 0 || nb2[1] != 3 || nb2[2] != 6) {
            fail_test(testname, __FILE__, __LINE__, "neighbours(2)");
        }

        alst.get_neighbours(4, nb4);
        if (nb4.size() != 2) {
            fail_test(testname, __FILE__, __LINE__, "# neighbours(4)");
        }
        if (nb4[0] != 1 || nb4[1] != 3) {
            fail_test(testname, __FILE__, __LINE__, "neighbours(4)");
        }

        alst.get_neighbours(7, nb7);
        if (nb7.size() != 0) {
            fail_test(testname, __FILE__, __LINE__, "# neighbours(7)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Test retrieving connected nodes
 **/
void adjacency_list_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "adjacency_list_test::test_3()";

    try {

        adjacency_list alst;
        alst.add(0, 1);
        alst.add(0, 2);
        alst.add(0, 4);
        alst.add(0, 8);
        alst.add(1, 2);
        alst.add(2, 4);
        alst.add(3, 5);
        alst.add(3, 6);
        alst.add(3, 7);
        alst.add(5, 9);
        alst.add(6, 9);

        std::vector<size_t> conn0, conn3, conn7, conn8, conn10;
        alst.get_connected(0, conn0);
        if (conn0.size() != 5) {
            fail_test(testname, __FILE__, __LINE__, "# connected(0)");
        }
        alst.get_connected(8, conn8);
        if (conn8.size() != 5) {
            fail_test(testname, __FILE__, __LINE__, "# connected(8)");
        }
        for (size_t i = 0; i < 5; i++) {
            if (conn0[i] != conn8[i]) {
                fail_test(testname, __FILE__, __LINE__,
                        "connected(1) != connected(8)");
            }
        }

        alst.get_connected(3, conn3);
        if (conn3.size() != 5) {
            fail_test(testname, __FILE__, __LINE__, "# connected(3)");
        }
        alst.get_connected(7, conn7);
        if (conn7.size() != 5) {
            fail_test(testname, __FILE__, __LINE__, "# connected(7)");
        }
        for (size_t i = 0; i < 5; i++) {
            if (conn3[i] != conn7[i]) {
                fail_test(testname, __FILE__, __LINE__,
                        "connected(3) != connected(7)");
            }
        }

        alst.get_connected(10, conn10);
        if (conn10.size() != 0) {
            fail_test(testname, __FILE__, __LINE__, "# connected(10)");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
