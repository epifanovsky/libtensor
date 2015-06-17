#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor_ctrl.h>
#include "ctf_dense_tensor_test.h"
#include "ctf_symmetry_test_equals.h"

namespace libtensor {


void ctf_dense_tensor_test::perform() throw(libtest::test_exception) {

    ctf::init();

    test_1();
    test_2();
    test_3();

    ctf::exit();
}


void ctf_dense_tensor_test::test_1() {

    static const char testname[] = "ctf_dense_tensor_test::test_1()";

    try {

    index<1> i1, i2;
    i2[0] = 2;
    dimensions<1> dims(index_range<1>(i1, i2));
    ctf_dense_tensor<1, double> t1(dims);

    if(t1.is_immutable()) {
        fail_test(testname, __FILE__, __LINE__,
            "New tensor must be mutable (t1)");
    }

    if(t1.get_dims()[0] != 3) {
        fail_test(testname, __FILE__, __LINE__,
            "Incorrect tensor dimension 0 (t1)");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_dense_tensor_test::test_2() {

    static const char testname[] = "ctf_dense_tensor_test::test_2()";

    try {

    index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    dimensions<2> dims(index_range<2>(i1, i2));
    ctf_dense_tensor<2, double> t1(dims);

    if(t1.get_dims()[0] != 3) {
        fail_test(testname, __FILE__, __LINE__,
            "Incorrect tensor dimension 0 (t1)");
    }
    if(t1.get_dims()[1] != 3) {
        fail_test(testname, __FILE__, __LINE__,
            "Incorrect tensor dimension 1 (t1)");
    }

    ctf_dense_tensor_ctrl<2, double> c1(t1);
    const ctf_symmetry<2, double> &sym1 = c1.req_symmetry();
    sequence<2, unsigned> grp1(0), indic1(0);
    grp1[0] = 0; grp1[1] = 1;
    ctf_symmetry<2, double> sym1_ref(grp1, indic1);
    if(!ctf_symmetry_test_equals(sym1, sym1_ref)) {
        fail_test(testname, __FILE__, __LINE__, "Incorrect symmetry of t1 (1)");
    }

    sequence<2, unsigned> grp2(0), indic2(0);
    grp2[0] = 0; grp2[1] = 0;
    ctf_symmetry<2, double> sym2_ref(grp2, indic2);
    c1.reset_symmetry(sym2_ref);
    const ctf_symmetry<2, double> &sym2 = c1.req_symmetry();
    if(!ctf_symmetry_test_equals(sym2, sym2_ref)) {
        fail_test(testname, __FILE__, __LINE__, "Incorrect symmetry of t1 (2)");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_dense_tensor_test::test_3() {

    static const char testname[] = "ctf_dense_tensor_test::test_3()";

    try {

    index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    dimensions<2> dims(index_range<2>(i1, i2));
    sequence<2, unsigned> grp1(0), indic1(0);
    grp1[0] = 0; grp1[1] = 1;
    ctf_symmetry<2, double> sym1_ref(grp1, indic1);

    ctf_dense_tensor<2, double> t1(dims, sym1_ref);

    if(t1.get_dims()[0] != 3) {
        fail_test(testname, __FILE__, __LINE__,
            "Incorrect tensor dimension 0 (t1)");
    }
    if(t1.get_dims()[1] != 3) {
        fail_test(testname, __FILE__, __LINE__,
            "Incorrect tensor dimension 1 (t1)");
    }

    ctf_dense_tensor_ctrl<2, double> c1(t1);
    const ctf_symmetry<2, double> &sym1 = c1.req_symmetry();
    if(!ctf_symmetry_test_equals(sym1, sym1_ref)) {
        fail_test(testname, __FILE__, __LINE__, "Incorrect symmetry of t1 (1)");
    }

    sequence<2, unsigned> grp2(0), indic2(0);
    grp2[0] = 0; grp2[1] = 0;
    ctf_symmetry<2, double> sym2_ref(grp2, indic2);
    c1.reset_symmetry(sym2_ref);
    const ctf_symmetry<2, double> &sym2 = c1.req_symmetry();
    if(!ctf_symmetry_test_equals(sym2, sym2_ref)) {
        fail_test(testname, __FILE__, __LINE__, "Incorrect symmetry of t1 (2)");
    }

    ctf_dense_tensor<2, double> t2(dims, sym2_ref);
    ctf_dense_tensor_ctrl<2, double> c2(t2);
    if(!ctf_symmetry_test_equals(c2.req_symmetry(), sym2_ref)) {
        fail_test(testname, __FILE__, __LINE__, "Incorrect symmetry of t2");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

