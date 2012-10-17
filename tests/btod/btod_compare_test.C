#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/btod_compare.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/core/allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include "btod_compare_test.h"

namespace libtensor {


void btod_compare_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 65536, 65536);
    try {

    test_1();
    test_2a();
    test_2b();
    test_3a();
    test_3b();
    test_4a();
    test_4b();
    test_5a();
    test_5b();
    test_6();
    test_exc();
    test_operation();

    } catch (...) {
        allocator<double>::vmm().shutdown();
        throw;
    }
    allocator<double>::vmm().shutdown();
}


/** \test Comparison of a block %tensor with itself
 **/
void btod_compare_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "btod_compare_test::test_1()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1, m2;
    m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
    bis.split(m1, 5);
    bis.split(m2, 2);

    block_tensor<4, double, allocator_t> bt1(bis);

    btod_compare<4> cmp(bt1, bt1);
    if(!cmp.compare()) {
        fail_test(testname, __FILE__, __LINE__, "!cmp.compare()");
    }
    if(cmp.get_diff().kind != btod_compare<4>::diff::DIFF_NODIFF) {
        fail_test(testname, __FILE__, __LINE__,
            "kind != diff::DIFF_NODIFF");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Comparison of two block %tensors with different number of orbits
 **/
void btod_compare_test::test_2a() throw(libtest::test_exception) {

    static const char *testname = "btod_compare_test::test_2a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1, m2;
    m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
    bis.split(m1, 5);
    bis.split(m2, 2);

    block_tensor<4, double, allocator_t> bt1(bis), bt2(bis);

    {
        scalar_transf<double> tr0;
        block_tensor_ctrl<4, double> ctrl1(bt1);
        ctrl1.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr0));
    }

    btod_compare<4> cmp(bt1, bt2);
    if(cmp.compare()) {
        fail_test(testname, __FILE__, __LINE__, "cmp.compare()");
    }
    if(cmp.get_diff().kind != btod_compare<4>::diff::DIFF_ORBLSTSZ) {
        fail_test(testname, __FILE__, __LINE__,
            "kind != diff::DIFF_ORBLSTSZ");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Comparison of two block %tensors with different number of orbits
 **/
void btod_compare_test::test_2b() throw(libtest::test_exception) {

    static const char *testname = "btod_compare_test::test_2b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1, m2;
    m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
    bis.split(m1, 5);
    bis.split(m2, 2);

    block_tensor<4, double, allocator_t> bt1(bis), bt2(bis);

    {
        scalar_transf<double> tr0;
        block_tensor_ctrl<4, double> ctrl2(bt2);
        ctrl2.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr0));
    }

    btod_compare<4> cmp(bt1, bt2);
    if(cmp.compare()) {
        fail_test(testname, __FILE__, __LINE__, "cmp.compare()");
    }
    if(cmp.get_diff().kind != btod_compare<4>::diff::DIFF_ORBLSTSZ) {
        fail_test(testname, __FILE__, __LINE__,
            "kind != diff::DIFF_ORBLSTSZ");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Comparison of two block %tensors with the same number of orbits,
        but different set of canonical indexes
 **/
void btod_compare_test::test_3a() throw(libtest::test_exception) {

    static const char *testname = "btod_compare_test::test_3a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1, m2;
    m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
    bis.split(m1, 5);
    bis.split(m2, 2);

    block_tensor<4, double, allocator_t> bt1(bis), bt2(bis);

    {
        scalar_transf<double> tr0;
        block_tensor_ctrl<4, double> ctrl1(bt1), ctrl2(bt2);
        ctrl1.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr0));
        ctrl2.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr0));
    }

    btod_compare<4> cmp(bt1, bt2);
    if(cmp.compare()) {
        fail_test(testname, __FILE__, __LINE__, "cmp.compare()");
    }
    if(cmp.get_diff().kind != btod_compare<4>::diff::DIFF_ORBIT) {
        fail_test(testname, __FILE__, __LINE__,
            "kind != diff::DIFF_ORBIT");
    }
    index<4> bidx_ref;
    bidx_ref[0] = 0; bidx_ref[1] = 0; bidx_ref[2] = 1; bidx_ref[3] = 0;
    if(!cmp.get_diff().bidx.equals(bidx_ref)) {
        fail_test(testname, __FILE__, __LINE__, "bidx != bidx_ref");
    }
    if(!cmp.get_diff().can1) {
        fail_test(testname, __FILE__, __LINE__, "can1 != true");
    }
    if(cmp.get_diff().can2) {
        fail_test(testname, __FILE__, __LINE__, "can2 != false");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Comparison of two block %tensors with the same number of orbits,
        but different set of canonical indexes
 **/
void btod_compare_test::test_3b() throw(libtest::test_exception) {

    static const char *testname = "btod_compare_test::test_3b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 5; i2[3] = 5;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1, m2;
    m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
    bis.split(m1, 5);
    bis.split(m2, 2);

    block_tensor<4, double, allocator_t> bt1(bis), bt2(bis);

    {
        scalar_transf<double> tr0;
        block_tensor_ctrl<4, double> ctrl1(bt1), ctrl2(bt2);
        ctrl1.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr0));
        ctrl2.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr0));
    }

    btod_compare<4> cmp(bt2, bt1);
    if(cmp.compare()) {
        fail_test(testname, __FILE__, __LINE__, "cmp.compare()");
    }
    if(cmp.get_diff().kind != btod_compare<4>::diff::DIFF_ORBIT) {
        fail_test(testname, __FILE__, __LINE__,
            "kind != diff::DIFF_ORBIT");
    }
    index<4> bidx_ref;
    bidx_ref[0] = 1; bidx_ref[1] = 0; bidx_ref[2] = 0; bidx_ref[3] = 0;
    if(!cmp.get_diff().bidx.equals(bidx_ref)) {
        fail_test(testname, __FILE__, __LINE__, "bidx != bidx_ref");
    }
    if(!cmp.get_diff().can1) {
        fail_test(testname, __FILE__, __LINE__, "can1 != true");
    }
    if(cmp.get_diff().can2) {
        fail_test(testname, __FILE__, __LINE__, "can2 != false");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Comparison of two block %tensors with the same number of orbits,
        but different set of canonical indexes
 **/
void btod_compare_test::test_4a() throw(libtest::test_exception) {

    static const char *testname = "btod_compare_test::test_4a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1;
    m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
    bis.split(m1, 5);
    bis.split(m1, 7);
    dimensions<4> bidims = bis.get_block_index_dims();

    block_tensor<4, double, allocator_t> bt1(bis), bt2(bis);
    symmetry<4, double> sym1(bis), sym2(bis);

    scalar_transf<double> tr0;
    sym1.insert(se_perm<4, double>(permutation<4>().permute(0, 1), tr0));
    sym1.insert(se_perm<4, double>(permutation<4>().permute(2, 3), tr0));
    sym2.insert(se_perm<4, double>(permutation<4>().permute(0, 2), tr0));
    sym2.insert(se_perm<4, double>(permutation<4>().permute(1, 3), tr0));

    {
        block_tensor_ctrl<4, double> ctrl1(bt1), ctrl2(bt2);
        so_copy<4, double>(sym1).perform(ctrl1.req_symmetry());
        so_copy<4, double>(sym2).perform(ctrl2.req_symmetry());
    }

    btod_compare<4> cmp(bt1, bt2);
    if(cmp.compare()) {
        fail_test(testname, __FILE__, __LINE__, "cmp.compare()");
    }
    if(cmp.get_diff().kind != btod_compare<4>::diff::DIFF_ORBIT) {
        fail_test(testname, __FILE__, __LINE__,
            "kind != diff::DIFF_ORBIT");
    }
    abs_index<4> ai(cmp.get_diff().bidx, bidims);
    orbit<4, double> o1(sym1, ai.get_index()), o2(sym2, ai.get_index());
    if((o1.get_abs_canonical_index() == ai.get_abs_index()) !=
        cmp.get_diff().can1) {
        fail_test(testname, __FILE__, __LINE__, "bad can1");
    }
    if((o2.get_abs_canonical_index() == ai.get_abs_index()) !=
        cmp.get_diff().can2) {
        fail_test(testname, __FILE__, __LINE__, "bad can2");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Comparison of two block %tensors with the same number of orbits,
        but different set of canonical indexes
 **/
void btod_compare_test::test_4b() throw(libtest::test_exception) {

    static const char *testname = "btod_compare_test::test_4b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1;
    m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
    bis.split(m1, 5);
    bis.split(m1, 7);
    dimensions<4> bidims = bis.get_block_index_dims();

    block_tensor<4, double, allocator_t> bt1(bis), bt2(bis);
    symmetry<4, double> sym1(bis), sym2(bis);

    scalar_transf<double> tr0;
    sym1.insert(se_perm<4, double>(permutation<4>().permute(0, 1), tr0));
    sym1.insert(se_perm<4, double>(permutation<4>().permute(2, 3), tr0));
    sym2.insert(se_perm<4, double>(permutation<4>().permute(0, 2), tr0));
    sym2.insert(se_perm<4, double>(permutation<4>().permute(1, 3), tr0));

    {
        block_tensor_ctrl<4, double> ctrl1(bt1), ctrl2(bt2);
        so_copy<4, double>(sym1).perform(ctrl1.req_symmetry());
        so_copy<4, double>(sym2).perform(ctrl2.req_symmetry());
    }

    btod_compare<4> cmp(bt2, bt1);
    if(cmp.compare()) {
        fail_test(testname, __FILE__, __LINE__, "cmp.compare()");
    }
    if(cmp.get_diff().kind != btod_compare<4>::diff::DIFF_ORBIT) {
        fail_test(testname, __FILE__, __LINE__,
            "kind != diff::DIFF_ORBIT");
    }
    abs_index<4> ai(cmp.get_diff().bidx, bidims);
    orbit<4, double> o1(sym1, ai.get_index()), o2(sym2, ai.get_index());
    if((o1.get_abs_canonical_index() == ai.get_abs_index()) !=
        cmp.get_diff().can2) {
        fail_test(testname, __FILE__, __LINE__, "bad can2");
    }
    if((o2.get_abs_canonical_index() == ai.get_abs_index()) !=
        cmp.get_diff().can1) {
        fail_test(testname, __FILE__, __LINE__, "bad can1");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Comparison of two block %tensors with the same number of orbits,
        same canonical indexes, but different block transformations
 **/
void btod_compare_test::test_5a() throw(libtest::test_exception) {

    static const char *testname = "btod_compare_test::test_5a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1;
    m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
    bis.split(m1, 5);
    bis.split(m1, 7);
    dimensions<4> bidims = bis.get_block_index_dims();

    block_tensor<4, double, allocator_t> bt1(bis), bt2(bis);
    symmetry<4, double> sym1(bis), sym2(bis);

    scalar_transf<double> tr0, tr1(-1.);
    sym1.insert(se_perm<4, double>(permutation<4>().permute(0, 1), tr0));
    sym2.insert(se_perm<4, double>(permutation<4>().permute(0, 1), tr1));

    {
        block_tensor_ctrl<4, double> ctrl1(bt1), ctrl2(bt2);
        so_copy<4, double>(sym1).perform(ctrl1.req_symmetry());
        so_copy<4, double>(sym2).perform(ctrl2.req_symmetry());
    }

    btod_compare<4> cmp(bt1, bt2);
    if(cmp.compare()) {
        fail_test(testname, __FILE__, __LINE__, "cmp.compare()");
    }
    if(cmp.get_diff().kind != btod_compare<4>::diff::DIFF_TRANSF) {
        fail_test(testname, __FILE__, __LINE__,
            "kind != diff::DIFF_TRANSF");
    }
    abs_index<4> ai(cmp.get_diff().bidx, bidims);
    orbit<4, double> o1(sym1, ai.get_index()), o2(sym2, ai.get_index());
    if((o1.get_abs_canonical_index() == ai.get_abs_index()) !=
        cmp.get_diff().can1) {
        fail_test(testname, __FILE__, __LINE__, "bad can1");
    }
    if((o2.get_abs_canonical_index() == ai.get_abs_index()) !=
        cmp.get_diff().can2) {
        fail_test(testname, __FILE__, __LINE__, "bad can2");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Comparison of two block %tensors with the same number of orbits,
        same canonical indexes, but different block transformations
 **/
void btod_compare_test::test_5b() throw(libtest::test_exception) {

    static const char *testname = "btod_compare_test::test_5b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1;
    m1[0] = true; m1[1] = true; m1[2] = true; m1[3] = true;
    bis.split(m1, 5);
    bis.split(m1, 7);
    dimensions<4> bidims = bis.get_block_index_dims();

    block_tensor<4, double, allocator_t> bt1(bis), bt2(bis);
    symmetry<4, double> sym1(bis), sym2(bis);

    scalar_transf<double> tr0, tr1(-1.);
    sym1.insert(se_perm<4, double>(permutation<4>().permute(0, 1), tr0));
    sym2.insert(se_perm<4, double>(permutation<4>().permute(0, 1), tr1));

    {
        block_tensor_ctrl<4, double> ctrl1(bt1), ctrl2(bt2);
        so_copy<4, double>(sym1).perform(ctrl1.req_symmetry());
        so_copy<4, double>(sym2).perform(ctrl2.req_symmetry());
    }

    btod_compare<4> cmp(bt2, bt1);
    if(cmp.compare()) {
        fail_test(testname, __FILE__, __LINE__, "cmp.compare()");
    }
    if(cmp.get_diff().kind != btod_compare<4>::diff::DIFF_TRANSF) {
        fail_test(testname, __FILE__, __LINE__,
            "kind != diff::DIFF_TRANSF");
    }
    abs_index<4> ai(cmp.get_diff().bidx, bidims);
    orbit<4, double> o1(sym1, ai.get_index()), o2(sym2, ai.get_index());
    if((o1.get_abs_canonical_index() == ai.get_abs_index()) !=
        cmp.get_diff().can2) {
        fail_test(testname, __FILE__, __LINE__, "bad can2");
    }
    if((o2.get_abs_canonical_index() == ai.get_abs_index()) !=
        cmp.get_diff().can1) {
        fail_test(testname, __FILE__, __LINE__, "bad can1");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Comparison of two block %tensors with the same orbits and
        transformations, but symmetries set up differently
 **/
void btod_compare_test::test_6() throw(libtest::test_exception) {

    static const char *testname = "btod_compare_test::test_6()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 5;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1, m2;
    m1[0] = true; m1[1] = true; m1[2] = true; m2[3] = true;
    bis.split(m1, 5);
    bis.split(m2, 2);

    block_tensor<4, double, allocator_t> bt1(bis), bt2(bis);

    {
        scalar_transf<double> tr0;
        block_tensor_ctrl<4, double> ctrl1(bt1);
        block_tensor_ctrl<4, double> ctrl2(bt2);
        ctrl1.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr0));
        ctrl1.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1).permute(1, 2), tr0));
        ctrl2.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1).permute(1, 2), tr0));
        ctrl2.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr0));
    }

    btod_compare<4> cmp(bt1, bt2);
    if(!cmp.compare()) {
        fail_test(testname, __FILE__, __LINE__, "!cmp.compare()");
    }
    if(cmp.get_diff().kind != btod_compare<4>::diff::DIFF_NODIFF) {
        fail_test(testname, __FILE__, __LINE__,
            "kind != diff::DIFF_NODIFF");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_compare_test::test_exc() throw(libtest::test_exception) {
    typedef index<2> index_t;
    typedef index_range<2> index_range_t;
    typedef dimensions<2> dimensions_t;
    typedef mask<2> mask_t;
    typedef block_index_space<2> block_index_space_t;
    typedef std_allocator<double> allocator_t;
    typedef block_tensor<2, double, allocator_t> block_tensor_t;


    index_t i1, i2, i3;
    i2[0]=5; i2[1]=5;
    i3[0]=7; i3[1]=7;
    index_range_t ir1(i1,i2), ir2(i1,i3);
    dimensions_t dim1(ir1), dim2(ir2);
    block_index_space_t bis1(dim1);
    mask_t mask;
    mask[0]=true; mask[1]=true;
    bis1.split(mask,3);
    block_tensor_t bt1(bis1);


    bool ok = false;
    try {
        block_index_space_t bis2(dim2);
        bis2.split(mask,3);
        block_tensor_t bt2(bis2);
        btod_compare<2> btc(bt1, bt2, 0);
    } catch(exception &e) {
        ok = true;
    }

    if(!ok) {
        fail_test("tod_compare_test::test_exc()", __FILE__, __LINE__,
            "Expected an exception with heterogeneous arguments");
    }

    ok = false;
    try {
        block_index_space_t bis2(dim2);
        mask[1]=false;
        bis2.split(mask,4);
        mask[0]=false; mask[1]=true;
        bis2.split(mask,2);
        block_tensor_t bt2(bis2);

        btod_compare<2> btc(bt1, bt2, 0);
    } catch(exception &e) {
        ok = true;
    }

    if(!ok) {
        fail_test("tod_compare_test::test_exc()", __FILE__, __LINE__,
            "Expected an exception with heterogeneous arguments");
    }


}


void btod_compare_test::test_operation() throw(libtest::test_exception) {

    static const char *testname = "btod_compare_test::test_operation()";

    typedef index<2> index_t;
    typedef index_range<2> index_range_t;
    typedef dimensions<2> dimensions_t;
    typedef mask<2> mask_t;
    typedef block_index_space<2> block_index_space_t;
    typedef std_allocator<double> allocator_t;
    typedef block_tensor<2, double, allocator_t> block_tensor_t;
    typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;


    index_t i1, i2;
    i2[0]=5; i2[1]=5;
    index_range_t ir(i1,i2);
    dimensions_t dim(ir);
    block_index_space_t bis(dim);
    mask_t mask;
    mask[0]=true; mask[1]=true;
    bis.split(mask,3);
    block_tensor_t bt1(bis), bt2(bis);

    btod_random<2> randr;
    randr.perform(bt1);

    btod_copy<2> docopy(bt1);
    docopy.perform(bt2);

    index_t block_idx, inblock_idx;
    block_idx[0]=1; block_idx[1]=0;
    inblock_idx[0]=1; inblock_idx[1]=1;

    block_tensor_ctrl_t btctrl(bt2);
    dense_tensor_wr_i<2,double> &t2 = btctrl.req_block(block_idx);
    double diff1, diff2;
    {
        dense_tensor_wr_ctrl<2,double> tctrl(t2);
        double *ptr=tctrl.req_dataptr();
        diff1=ptr[4];
        ptr[4]-=1.0;
        diff2=ptr[4];
        tctrl.ret_dataptr(ptr);
    }
    btctrl.ret_block(block_idx);
    bt1.set_immutable();
    bt2.set_immutable();

    btod_compare<2> cmp(bt1, bt2, 1e-7);

    if(cmp.compare()) {
        fail_test(testname, __FILE__, __LINE__,
            "Operation failed to find the difference.");
    }

//  if( ! op1.get_diff().m_number_of_orbits ) {
//      fail_test(testname, __FILE__, __LINE__,
//          "btod_compare returned different number of orbits");
//  }
//  if( ! op1.get_diff().m_similar_orbit ) {
//      fail_test(testname, __FILE__, __LINE__,
//          "btod_compare returned different orbit");
//  }
//  if( ! op1.get_diff().m_canonical_block_index_1
//          .equals(op1.get_diff().m_canonical_block_index_2) ) {
//      fail_test(testname, __FILE__, __LINE__,
//          "btod_compare returned different canonical blocks");
//  }
//  if( ! inblock_idx.equals(op1.get_diff().m_inblock) ) {
//      fail_test(testname, __FILE__, __LINE__,
//          "btod_compare returned an incorrect index");
//  }
//  if( op1.get_diff().m_diff_elem_1 != diff1 ||
//      op1.get_diff().m_diff_elem_2 != diff2) {
//      fail_test(testname, __FILE__,
//          __LINE__, "btod_compare returned an incorrect "
//          "element value");
//  }

}


} // namespace libtensor
