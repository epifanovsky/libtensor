#include <libtensor/core/sequence.h>
#include "sequence_test.h"

namespace libtensor {


void sequence_test::perform() throw (libtest::test_exception) {

    test_ctor_1();
    test_ctor_2();
    test_ctor_3();
    test_ctor_4();
    test_ctor_5();
    test_ctor_6();
    test_ctor_7();
    test_ctor_8();
    test_ctor_9();

    test_at_1();
    test_at_2();

    test_exc_1();
    test_exc_2();
}


namespace sequence_test_ns {

struct s1type {
    int i;
    s1type() : i(4) { }
};

struct s2type {
    int i;
    s2type(int i_ = -1) : i(i_) { }
};

} // namespace sequence_test_ns
using namespace sequence_test_ns;


/** \test Default constructor for POD types
 **/
void sequence_test::test_ctor_1() throw (libtest::test_exception) {

    static const char *testname = "sequence_test::test_ctor_1()";

    try {

        sequence<2, int> seq;

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Initializing constructor for POD types
 **/
void sequence_test::test_ctor_2() throw (libtest::test_exception) {

    static const char *testname = "sequence_test::test_ctor_2()";

    try {

        sequence<4, double> seq(1.0);

        for(size_t i = 0; i < 4; i++) {
            if(seq[i] != 1.0) {
                fail_test(testname, __FILE__, __LINE__,
                    "Unexpected value of element.");
            }
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Default constructor for POD types (zero-length sequence)
 **/
void sequence_test::test_ctor_3() throw (libtest::test_exception) {

    static const char *testname = "sequence_test::test_ctor_3()";

    try {

        sequence<0, int> seq;

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Initializing constructor for POD types (zero-length sequence)
 **/
void sequence_test::test_ctor_4() throw (libtest::test_exception) {

    static const char *testname = "sequence_test::test_ctor_4()";

    try {

        sequence<0, double> seq(1.0);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Copy constructor for POD types
 **/
void sequence_test::test_ctor_5() throw (libtest::test_exception) {

    static const char *testname = "sequence_test::test_ctor_5()";

    try {

        sequence<2, unsigned> seq1(2);
        sequence<2, unsigned> seq2(seq1);

        for(size_t i = 0; i < 2; i++) {
            if(seq2[i] != 2) {
                fail_test(testname, __FILE__, __LINE__,
                    "Unexpected value of element.");
            }
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Copy constructor for POD types (zero-length sequence)
 **/
void sequence_test::test_ctor_6() throw (libtest::test_exception) {

    static const char *testname = "sequence_test::test_ctor_6()";

    try {

        sequence<4, double> seq1(1.0);
        sequence<4, double> seq2(seq1);

        for(size_t i = 0; i < 4; i++) {
            if(seq2[i] != 1.0) {
                fail_test(testname, __FILE__, __LINE__,
                    "Unexpected value of element.");
            }
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Default constructor for non-POD types
 **/
void sequence_test::test_ctor_7() throw (libtest::test_exception) {

    static const char *testname = "sequence_test::test_ctor_7()";

    try {

        sequence<2, s1type> seq;

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Initializing constructor for non-POD types
 **/
void sequence_test::test_ctor_8() throw (libtest::test_exception) {

    static const char *testname = "sequence_test::test_ctor_8()";

    try {

        sequence<4, s2type> seq(s2type(2));

        for(size_t i = 0; i < 4; i++) {
            if(seq[i].i != 2) {
                fail_test(testname, __FILE__, __LINE__,
                    "Unexpected value of element.");
            }
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Copy constructor for non-POD types
 **/
void sequence_test::test_ctor_9() throw (libtest::test_exception) {

    static const char *testname = "sequence_test::test_ctor_9()";

    try {

        sequence<2, s2type> seq1(s2type(2));

        for(size_t i = 0; i < 2; i++) seq1[i].i = i + 1;

        sequence<2, s2type> seq2(seq1);

        for(size_t i = 0; i < 2; i++) {
            if(seq2[i].i != i + 1) {
                fail_test(testname, __FILE__, __LINE__,
                    "Unexpected value of element.");
            }
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests sequence<N, T>::operator[] with POD types
 **/
void sequence_test::test_at_1() throw (libtest::test_exception) {

    static const char *testname = "sequence_test::test_at_1()";

    try {

        sequence<2, int> seq;

        seq[0] = 5; seq[1] = 3;
        if(5 != seq[0]) {
            fail_test(testname, __FILE__, __LINE__,
                "(1) Unexpected value of element 0.");
        }
        if(3 != seq[1]) {
            fail_test(testname, __FILE__, __LINE__,
                "(1) Unexpected value of element 1.");
        }

        seq[0] = -2; seq[1] = 7;
        if(-2 != seq[0]) {
            fail_test(testname, __FILE__, __LINE__,
                "(2) Unexpected value of element 0.");
        }
        if(7 != seq[1]) {
            fail_test(testname, __FILE__, __LINE__,
                "(2) Unexpected value of element 1.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Tests sequence<N, T>::at() with POD types
 **/
void sequence_test::test_at_2() throw (libtest::test_exception) {

    static const char *testname = "sequence_test::test_at_2()";

    try {

        sequence<2, int> seq;

        seq.at(0) = 5; seq.at(1) = 3;
        if(5 != seq.at(0)) {
            fail_test(testname, __FILE__, __LINE__,
                "(1) Unexpected value of element 0.");
        }
        if(3 != seq.at(1)) {
            fail_test(testname, __FILE__, __LINE__,
                "(1) Unexpected value of element 1.");
        }

        seq.at(0) = -2; seq.at(1) = 7;
        if(-2 != seq.at(0)) {
            fail_test(testname, __FILE__, __LINE__,
                "(2) Unexpected value of element 0.");
        }
        if(7 != seq.at(1)) {
            fail_test(testname, __FILE__, __LINE__,
                "(2) Unexpected value of element 1.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Verifies that out_of_bounds is raised properly by
        sequence<N, T>::operator[]
 **/
void sequence_test::test_exc_1() throw (libtest::test_exception) {

    static const char *testname = "sequence_test::test_exc_1()";

    bool exc;

    try {

        sequence<2, int> seq;

        try {
            seq[0] = 2;
        } catch(out_of_bounds&) {
            fail_test(testname, __FILE__, __LINE__,
                "Unexpected out_of_bounds.");
        }

        try {
            seq[1] = 4;
        } catch(out_of_bounds&) {
            fail_test(testname, __FILE__, __LINE__,
                "Unexpected out_of_bounds.");
        }

#ifdef LIBTENSOR_DEBUG
        exc = false;
        try {
            seq[2] = 6;
        } catch(out_of_bounds&) {
            exc = true;
        }
        if(!exc) {
            fail_test(testname, __FILE__, __LINE__,
                "Expected out_of_bounds, but not raised.");
        }

        exc = false;
        try {
            seq[7] = 7;
        } catch(out_of_bounds&) {
            exc = true;
        }
        if(!exc) {
            fail_test(testname, __FILE__, __LINE__,
                "Expected out_of_bounds, but not raised.");
        }
#endif // LIBTENSOR_DEBUG

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Verifies that out_of_bounds is raised properly by
        sequence<N, T>::at()
 **/
void sequence_test::test_exc_2() throw (libtest::test_exception) {

    static const char *testname = "sequence_test::test_exc_2()";

    bool exc;

    try {

        sequence<2, int> seq;

        try {
            seq.at(0) = 2;
        } catch(out_of_bounds&) {
            fail_test(testname, __FILE__, __LINE__,
                "Unexpected out_of_bounds.");
        }

        try {
            seq.at(1) = 4;
        } catch(out_of_bounds&) {
            fail_test(testname, __FILE__, __LINE__,
                "Unexpected out_of_bounds.");
        }

        exc = false;
        try {
            seq.at(2) = 6;
        } catch(out_of_bounds&) {
            exc = true;
        }
        if(!exc) {
            fail_test(testname, __FILE__, __LINE__,
                "Expected out_of_bounds, but not raised.");
        }

        exc = false;
        try {
            seq.at(7) = 7;
        } catch(out_of_bounds&) {
            exc = true;
        }
        if(!exc) {
            fail_test(testname, __FILE__, __LINE__,
                "Expected out_of_bounds, but not raised.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
