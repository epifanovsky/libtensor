#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_random.h>
#include "../compare_ref.h"
#include "ctf_btod_random_test.h"

namespace libtensor {


void ctf_btod_random_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_btod_random_test::test_1() {

    static const char testname[] = "ctf_btod_random_test::test_1()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2));
    block_index_space<2> bisa(dimsa);
    bisa.split(m11, 30);

    ctf_block_tensor<2, double> dbta(bisa);

    ctf_btod_random<2>().perform(dbta);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

