#include <sstream>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/short_orbit.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/se_label.h>
#include "../test_utils.h"

using namespace libtensor;


int test_1() {

    static const char testname[] = "short_orbit_test::test_1()";

    try {

    index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    mask<2> msk;
    msk[0] = true; msk[1] = true;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(msk, 1);
    bis.split(msk, 2);
    symmetry<2, double> sym(bis);

    abs_index<2> aio(dims);
    do {
        const index<2> &io = aio.get_index();
        short_orbit<2, double> orb(sym, io);
        if(orb.get_acindex() != aio.get_abs_index()) {
            std::ostringstream ss;
            ss << "Failure to detect a canonical index: " << io << ".";
            return fail_test(testname, __FILE__, __LINE__,
                ss.str().c_str());
        }
    } while(aio.inc());

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_2() {

    static const char testname[] = "orbit_test::test_2()";

    try {

    index<2> i1, i2;
    i2[0] = 2; i2[1] = 2;
    mask<2> msk;
    msk[0] = true; msk[1] = true;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(msk, 1);
    bis.split(msk, 2);
    symmetry<2, double> sym(bis);
    permutation<2> perm; perm.permute(0, 1);
    scalar_transf<double> tr0;
    se_perm<2, double> cycle(perm, tr0);
    sym.insert(cycle);

    abs_index<2> aio(dims);
    do {
        const index<2> &io = aio.get_index();
        short_orbit<2, double> orb(sym, io);
        bool can = io[0] <= io[1];
        size_t abscanidx = orb.get_acindex();
        if((can && abscanidx != aio.get_abs_index()) ||
            (!can && abscanidx == aio.get_abs_index())) {

            std::ostringstream ss;
            ss << "Failure to detect a canonical index: " << io
                << " (can = " << can << ").";
            return fail_test(testname, __FILE__, __LINE__,
                ss.str().c_str());
        }
    } while(aio.inc());

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

