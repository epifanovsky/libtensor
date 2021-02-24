#include <sstream>
#include <libtensor/core/combined_orbits.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/se_label.h>
#include "../test_utils.h"

using namespace libtensor;


int test_1() {

    static const char testname[] = "combined_orbits_test::test_1()";

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(m11, 1);
    bis.split(m11, 2);
    symmetry<2, double> sym1(bis), sym2(bis), sym3(bis);

    combined_orbits<2, double> co(sym1, sym2, sym3, 0);
    if(co.get_size() != 1) {
        return fail_test(testname, __FILE__, __LINE__, "so.get_size() != 1");
    }
    if(!co.contains(0)) {
        return fail_test(testname, __FILE__, __LINE__, "!co.contains(0)");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_2() {

    static const char testname[] = "combined_orbits_test::test_2()";

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(m11, 1);
    bis.split(m11, 2);
    symmetry<2, double> sym1(bis), sym2(bis), sym3(bis);

    se_perm<2, double> se1(permutation<2>().permute(0, 1),
        scalar_transf<double>(1.0));
    se_perm<2, double> se2(permutation<2>().permute(0, 1),
        scalar_transf<double>(-1.0));
    sym1.insert(se1);
    sym2.insert(se2);

    combined_orbits<2, double> co(sym1, sym2, sym3, 2);
    if(co.get_size() != 2) {
        return fail_test(testname, __FILE__, __LINE__, "co.get_size() != 2");
    }
    // [0,2] = 2
    if(!co.contains(2)) {
        return fail_test(testname, __FILE__, __LINE__, "!co.contains(2)");
    }
    // [2,0] = 6
    if(!co.contains(6)) {
        return fail_test(testname, __FILE__, __LINE__, "!co.contains(6)");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_3() {

    static const char testname[] = "combined_orbits_test::test_3()";

    try {

    libtensor::index<4> i1, i2;
    i2[0] = 2; i2[1] = 2; i2[2] = 2; i2[3] = 2;
    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    bis.split(m1111, 1);
    bis.split(m1111, 2);
    symmetry<4, double> sym1(bis), sym2(bis), sym3(bis);

    se_perm<4, double> se1(permutation<4>().permute(0, 1),
        scalar_transf<double>(1.0));
    se_perm<4, double> se2(permutation<4>().permute(1, 2),
        scalar_transf<double>(1.0));
    se_perm<4, double> se3(permutation<4>().permute(2, 3),
        scalar_transf<double>(1.0));
    sym1.insert(se1);
    sym1.insert(se2);
    sym2.insert(se2);
    sym2.insert(se3);
    sym3.insert(se2);

    //  [0,1,2,0] = 15
    combined_orbits<4, double> co(sym1, sym2, sym3, 15);
    if(co.get_size() != 7) {
        return fail_test(testname, __FILE__, __LINE__, "co.get_size() != 7");
    }
    if(!co.contains(15)) {
        return fail_test(testname, __FILE__, __LINE__, "!co.contains(15)");
    }
    //  [1,0,2,0] = 33
    if(!co.contains(33)) {
        return fail_test(testname, __FILE__, __LINE__, "!co.contains(33)");
    }
    //  [2,0,1,0] = 57
    if(!co.contains(57)) {
        return fail_test(testname, __FILE__, __LINE__, "!co.contains(57)");
    }
    //  [1,0,0,2] = 29
    if(!co.contains(29)) {
        return fail_test(testname, __FILE__, __LINE__, "!co.contains(29)");
    }
    //  [2,0,0,1] = 55
    if(!co.contains(55)) {
        return fail_test(testname, __FILE__, __LINE__, "!co.contains(55)");
    }
    //  [0,0,1,2] = 5
    if(!co.contains(5)) {
        return fail_test(testname, __FILE__, __LINE__, "!co.contains(5)");
    }
    //  [0,0,2,1] = 7
    if(!co.contains(7)) {
        return fail_test(testname, __FILE__, __LINE__, "!co.contains(7)");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    return

    test_1() |
    test_2() |
    test_3() |

    0;
}

