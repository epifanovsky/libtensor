#include <sstream>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/subgroup_orbits.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/se_label.h>
#include "../test_utils.h"

using namespace libtensor;


int test_1() {

    static const char testname[] = "subgroup_orbits_test::test_1()";

    try {

    index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(m11, 1);
    bis.split(m11, 2);
    symmetry<2, double> sym1(bis), sym2(bis);

    subgroup_orbits<2, double> so(sym1, sym2, 0);
    if(so.get_size() != 1) {
        return fail_test(testname, __FILE__, __LINE__, "so.get_size() != 1");
    }
    if(!so.contains(0)) {
        return fail_test(testname, __FILE__, __LINE__, "!so.contains(0)");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_2() {

    static const char testname[] = "subgroup_orbits_test::test_2()";

    try {

    index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(m11, 1);
    bis.split(m11, 2);
    symmetry<2, double> sym1(bis), sym2(bis);

    se_perm<2, double> se1(permutation<2>().permute(0, 1),
        scalar_transf<double>(1.0));
    sym1.insert(se1);

    subgroup_orbits<2, double> so(sym1, sym2, 2);
    if(so.get_size() != 2) {
        return fail_test(testname, __FILE__, __LINE__, "so.get_size() != 2");
    }
    // [0,2] = 2
    if(!so.contains(2)) {
        return fail_test(testname, __FILE__, __LINE__, "!so.contains(2)");
    }
    // [2,0] = 6
    if(!so.contains(6)) {
        return fail_test(testname, __FILE__, __LINE__, "!so.contains(6)");
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

    0;
}

