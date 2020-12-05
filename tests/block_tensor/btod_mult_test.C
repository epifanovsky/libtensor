#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_mult.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/dense_tensor/tod_mult.h>
#include <iomanip>
#include <sstream>
#include "btod_mult_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_mult_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(4, 16, 16777216, 16777216);

    try {

    test_1(false, false); test_1(false, true);
    test_1(true, false);  test_1(true, true);
    test_2(false, false); test_2(false, true);
    test_2(true, false);  test_2(true, true);
    test_3(false, false); test_3(false, true);
    test_3(true, false);  test_3(true, true);
    test_4(false, false); test_4(false, true);
    test_4(true, false);  test_4(true, true);
    test_5(false, false); test_5(false, true);
    test_5(true, false);  test_5(true, true);
    test_6(false, false); test_6(false, true);
    test_6(true, false);  test_6(true, true);
    test_7(false, false, false, false, false);
    test_7(false, false, false, false, true);
    test_7(false, false, false, true, false);
    test_7(false, false, false, true, true);
    test_7(false, false, true, false, false);
    test_7(false, false, true, false, true);
    test_7(false, false, true, true, false);
    test_7(false, false, true, true, true);
    test_7(false, true, false, false, false);
    test_7(false, true, false, false, true);
    test_7(false, true, false, true, false);
    test_7(false, true, false, true, true);
    test_7(false, true, true, false, false);
    test_7(false, true, true, false, true);
    test_7(false, true, true, true, false);
    test_7(false, true, true, true, true);
    test_7(true, false, false, false, false);
    test_7(true, false, false, false, true);
    test_7(true, false, false, true, false);
    test_7(true, false, false, true, true);
    test_7(true, false, true, false, false);
    test_7(true, false, true, false, true);
    test_7(true, false, true, true, false);
    test_7(true, false, true, true, true);
    test_7(true, true, false, false, false);
    test_7(true, true, false, false, true);
    test_7(true, true, false, true, false);
    test_7(true, true, false, true, true);
    test_7(true, true, true, false, false);
    test_7(true, true, true, false, true);
    test_7(true, true, true, true, false);
    test_7(true, true, true, true, true);
    test_8a(false, false);
    test_8a(false, true);
    test_8a(true, false);
    test_8a(true, true);
    test_8b(false, false);
    test_8b(false, true);
    test_8b(true, false);
    test_8b(true, true);

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


/** \test Elementwise multiplication/division of two order-2 tensors with no symmetry
        and no zero blocks.
 **/
void btod_mult_test::test_1(
        bool recip, bool doadd) throw(libtest::test_exception) {

    std::ostringstream oss;
    oss << "btod_mult_test::test_1("
            << (recip ? "true" : "false") << ","
            << (doadd ? "true" : "false") << ")";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    dimensions<2> bidims(bis.get_block_index_dims());

    block_tensor<2, double, allocator_t> bta(bis), btb(bis), btc(bis);
    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tc(dims),
        tc_ref(dims);

    //  Fill in random data

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);
    btod_random<2>().perform(btc);
    bta.set_immutable();
    btb.set_immutable();

    //  Prepare the reference

    tod_btconv<2>(bta).perform(ta);
    tod_btconv<2>(btb).perform(tb);
    tod_btconv<2>(btc).perform(tc_ref);

    //  Invoke the operation
    if (doadd) {
        tod_mult<2>(ta, tb, recip, 0.5).perform(false, tc_ref);
        btod_mult<2>(bta, btb, recip).perform(btc, 0.5);
    }
    else {
        tod_mult<2>(ta, tb, recip).perform(true, tc_ref);
        btod_mult<2>(bta, btb, recip).perform(btc);
    }

    tod_btconv<2>(btc).perform(tc);

    //  Compare against the reference

    compare_ref<2>::compare(oss.str().c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(oss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Elementwise multiplication/division of two order-2 tensors
        with no symmetry and no zero blocks, second tensor permuted
 **/
void btod_mult_test::test_2(
        bool recip, bool doadd) throw(libtest::test_exception) {

    static const char *testname = "btod_mult_test::test_2";
    std::ostringstream oss;
    oss << testname << "("
            << (recip ? "true" : "false") << ","
            << (doadd ? "true" : "false") << ")";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    dimensions<2> bidims(bis.get_block_index_dims());

    block_tensor<2, double, allocator_t> bta(bis), btb(bis), btc(bis);
    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tc(dims),
        tc_ref(dims);

    //  Fill in random data

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);
    btod_random<2>().perform(btc);
    bta.set_immutable();
    btb.set_immutable();

    //  Prepare the reference

    tod_btconv<2>(bta).perform(ta);
    tod_btconv<2>(btb).perform(tb);
    tod_btconv<2>(btc).perform(tc_ref);

    permutation<2> pa, pb;
    pb.permute(0, 1);

    //  Invoke the operation
    if (doadd) {
        tod_mult<2>(ta, pa, tb, pb, recip, 0.5).perform(false, tc_ref);
        btod_mult<2>(bta, pa, btb, pb, recip).perform(btc, 0.5);
    }
    else {
        tod_mult<2>(ta, pa, tb, pb, recip).perform(true, tc_ref);
        btod_mult<2>(bta, pa, btb, pb, recip).perform(btc);
    }
    tod_btconv<2>(btc).perform(tc);

    //  Compare against the reference

    compare_ref<2>::compare(oss.str().c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(oss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Elementwise multiplication/division of two order-2 tensors with
         symmetry and zero blocks.
 **/
void btod_mult_test::test_3(
        bool recip, bool doadd) throw(libtest::test_exception) {

    static const char *testname = "btod_mult_test::test_3";
    std::ostringstream oss;
    oss << testname << "("
            << (recip ? "true" : "false") << ","
            << (doadd ? "true" : "false") << ")";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> msk;
    msk[0] =  true; msk[1] = true;
    bis.split(msk, 3);
    bis.split(msk, 7);
    dimensions<2> bidims(bis.get_block_index_dims());

    permutation<2> perm;
    perm.permute(0, 1);
    scalar_transf<double> tr0;
    se_perm<2, double> sp(perm, tr0);
    block_tensor<2, double, allocator_t> bta(bis), btb(bis), btc(bis);

    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tc(dims),
        tc_ref(dims);


    { // add symmetry and set zero blocks
    block_tensor_ctrl<2, double> cbta(bta), cbtb(btb), cbtc(btc);
    cbta.req_symmetry().insert(sp);
    cbtb.req_symmetry().insert(sp);
    cbtc.req_symmetry().insert(sp);
    }

    //  Fill in random data

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);
    btod_random<2>().perform(btc);

    { // set zero blocks
    block_tensor_ctrl<2, double> cbta(bta);
    libtensor::index<2> idxa;
    idxa[0] = 0; idxa[1] = 2;
    orbit<2, double> oa(cbta.req_const_symmetry(), idxa);
    abs_index<2> cidxa(oa.get_acindex(), bidims);
    cbta.req_zero_block(cidxa.get_index());
    }

    bta.set_immutable();
    btb.set_immutable();

    //  Prepare the reference

    tod_btconv<2>(bta).perform(ta);
    tod_btconv<2>(btb).perform(tb);
    tod_btconv<2>(btc).perform(tc_ref);

    //  Invoke the operation

    if (doadd) {
        tod_mult<2>(ta, tb, recip, -0.5).perform(false, tc_ref);
        btod_mult<2>(bta, btb, recip).perform(btc, -0.5);
    }
    else {
        tod_mult<2>(ta, tb, recip).perform(true, tc_ref);
        btod_mult<2>(bta, btb, recip).perform(btc);
    }
    tod_btconv<2>(btc).perform(tc);

    //  Compare against the reference

    compare_ref<2>::compare(oss.str().c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(oss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Elementwise multiplaction/division of two order-4 tensors
        with symmetry and zero blocks.
 **/
void btod_mult_test::test_4(
        bool recip, bool doadd) throw(libtest::test_exception) {

    static const char *testname = "btod_mult_test::test_4";
    std::ostringstream oss;
    oss << testname << "("
            << (recip ? "true" : "false") << ","
            << (doadd ? "true" : "false") << ")";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 7; i2[3] = 7;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> msk1, msk2;
    msk1[0] = true; msk1[1] = true;
    msk2[2] = true; msk2[3] = true;
    bis.split(msk1, 3); bis.split(msk1, 6);
    bis.split(msk2, 4);

    dimensions<4> bidims(bis.get_block_index_dims());

    permutation<4> p10, p32;
    p10.permute(0, 1);
    p32.permute(2, 3);
    scalar_transf<double> tr0, tr1(-1.);
    se_perm<4, double> spa(p10, tr1), spb(p32, tr0);

    block_tensor<4, double, allocator_t> bta(bis), btb(bis), btc(bis);
    dense_tensor<4, double, allocator_t> ta(dims), tb(dims), tc(dims),
        tc_ref(dims);

    {
    block_tensor_ctrl<4, double> cbta(bta), cbtb(btb);
    cbta.req_symmetry().insert(spa);
    cbtb.req_symmetry().insert(spb);
    }

    //  Fill in random data

    btod_random<4>().perform(bta);
    btod_random<4>().perform(btb);
    btod_random<4>().perform(btc);

    {
    block_tensor_ctrl<4, double> cbta(bta);
    libtensor::index<4> idxa;
    idxa[0] = 0; idxa[1] = 1; idxa[2] = 1; idxa[3] = 0;
    orbit<4, double> oa(cbta.req_const_symmetry(), idxa);
    abs_index<4> cidxa(oa.get_acindex(), bidims);
    cbta.req_zero_block(cidxa.get_index());
    }

    bta.set_immutable();
    btb.set_immutable();

    //  Prepare the reference

    tod_btconv<4>(bta).perform(ta);
    tod_btconv<4>(btb).perform(tb);
    tod_btconv<4>(btc).perform(tc_ref);


    //  Invoke the operation

    if (doadd) {
        tod_mult<4>(ta, tb, recip, 0.5).perform(false, tc_ref);
        btod_mult<4>(bta, btb, recip).perform(btc, 0.5);
    }
    else {
        tod_mult<4>(ta, tb, recip).perform(true, tc_ref);
        btod_mult<4>(bta, btb, recip).perform(btc);
    }

    tod_btconv<4>(btc).perform(tc);

    //  Compare against the reference

    compare_ref<4>::compare(oss.str().c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(oss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Elementwise multiplaction/division of two order-2 tensors
        with permutational symmetry and anti-symmetry.
        Test for the right result symmetry!
 **/
void btod_mult_test::test_5(bool symm1, bool symm2) throw(libtest::test_exception) {

    std::ostringstream testname;
    testname << "btod_mult_test::test_5("
            << (symm1 ? "true" : "false") << ", "
            << (symm2 ? "true" : "false") << ")";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> msk1;
    msk1[0] = true; msk1[1] = true;
    bis.split(msk1, 5);

    dimensions<2> bidims(bis.get_block_index_dims());

    permutation<2> p10;
    p10.permute(0, 1);
    scalar_transf<double> tr1(symm1 ? 1. : -1.), tr2(symm2 ? 1. : -1.);
    se_perm<2, double> spa(p10, tr1), spb(p10, tr2);

    block_tensor<2, double, allocator_t> bta(bis), btb(bis);

    {
    block_tensor_ctrl<2, double> cbta(bta), cbtb(btb);
    cbta.req_symmetry().insert(spa);
    cbtb.req_symmetry().insert(spb);
    }

    //  Fill in random data

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);

    bta.set_immutable();
    btb.set_immutable();

    //  Invoke the operation
    btod_mult<2> op(bta, btb);
    const symmetry<2, double> &sym = op.get_symmetry();

    bool found = false;
    for (symmetry<2, double>::iterator is = sym.begin();
            is != sym.end(); is++) {

        const symmetry_element_set<2, double> &set = sym.get_subset(is);
        if (set.get_id().compare(spa.get_type()) != 0)
            fail_test(testname.str().c_str(), __FILE__, __LINE__,
                    "Unknown symmetry element type.");

        if (set.is_empty())
            fail_test(testname.str().c_str(), __FILE__, __LINE__,
                    "Permutational symmetry missing.");

        libtensor::index<2> idx;
        tensor_transf<2, double> tr;
        for (symmetry_element_set<2, double>::const_iterator iss =
                set.begin(); iss != set.end(); iss++) {

            const symmetry_element_i<2, double> &elem = set.get_elem(iss);
            elem.apply(idx, tr);

        }

        if (! tr.get_perm().equals(p10))
            fail_test(testname.str().c_str(), __FILE__, __LINE__,
                    "Wrong permutational symmetry.");

        if (symm1 == symm2) {
            if (tr.get_scalar_tr().get_coeff() != 1.0) {
                fail_test(testname.str().c_str(), __FILE__, __LINE__,
                        "Wrong permutational symmetry.");
            }
        }
        else {
            if (tr.get_scalar_tr().get_coeff() != -1.0) {
                fail_test(testname.str().c_str(), __FILE__, __LINE__,
                        "Wrong permutational symmetry.");
            }
        }

        found = true;
    }

    if (! found)
        fail_test(testname.str().c_str(), __FILE__, __LINE__, "Symmetry missing.");

    } catch(exception &e) {
        fail_test(testname.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Elementwise multiplaction/division of two order-4 tensors
        with permutational symmetry and anti-symmetry.
        Test for the right result symmetry!
 **/
void btod_mult_test::test_6(bool symm1, bool symm2) throw(libtest::test_exception) {

    std::ostringstream testname;
    testname << "btod_mult_test::test_6("
            << (symm1 ? "true" : "false") << ", "
            << (symm2 ? "true" : "false") << ")";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 7; i2[3] = 7;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> msk1, msk2;
    msk1[0] = true; msk1[1] = true;
    msk2[2] = true; msk2[3] = true;
    bis.split(msk1, 5);
    bis.split(msk2, 4);
    dimensions<4> bidims(bis.get_block_index_dims());

    permutation<4> p10, p32;
    p10.permute(0, 1); p32.permute(2, 3);

    scalar_transf<double> tr1(symm1 ? 1. : -1.), tr2(symm2 ? 1. : -1.);
    se_perm<4, double> spa1(p10, tr1), spa2(p32, tr1);
    se_perm<4, double> spb(p10, tr2);

    block_tensor<4, double, allocator_t> bta(bis), btb(bis);

    {
    block_tensor_ctrl<4, double> cbta(bta), cbtb(btb);
    cbta.req_symmetry().insert(spa1);
    cbta.req_symmetry().insert(spa2);
    cbtb.req_symmetry().insert(spb);
    }

    //  Fill in random data

    btod_random<4>().perform(bta);
    btod_random<4>().perform(btb);

    bta.set_immutable();
    btb.set_immutable();

    //  Invoke the operation
    btod_mult<4> op(bta, btb);
    const symmetry<4, double> &sym = op.get_symmetry();

    bool found = false;
    for (symmetry<4, double>::iterator is = sym.begin();
            is != sym.end(); is++) {

        const symmetry_element_set<4, double> &set = sym.get_subset(is);
        if (set.get_id().compare(spb.get_type()) != 0)
            fail_test(testname.str().c_str(), __FILE__, __LINE__,
                    "Unknown symmetry element type.");

        if (set.is_empty())
            fail_test(testname.str().c_str(), __FILE__, __LINE__,
                    "Permutational symmetry missing.");

        libtensor::index<4> idx;
        tensor_transf<4, double> tr;
        for (symmetry_element_set<4, double>::const_iterator iss =
                set.begin(); iss != set.end(); iss++) {

            const symmetry_element_i<4, double> &elem = set.get_elem(iss);
            elem.apply(idx, tr);

        }

        if (! tr.get_perm().equals(p10))
            fail_test(testname.str().c_str(), __FILE__, __LINE__,
                    "Wrong permutational symmetry.");

        if (symm1 == symm2) {
            if (tr.get_scalar_tr().get_coeff() != 1.0) {
                fail_test(testname.str().c_str(), __FILE__, __LINE__,
                        "Wrong symm flag symmetry.");
            }
        }
        else {
            if (tr.get_scalar_tr().get_coeff() != -1.0) {
                fail_test(testname.str().c_str(), __FILE__, __LINE__,
                        "Wrong symm flag symmetry.");
            }
        }

        found = true;
    }

    if (! found)
        fail_test(testname.str().c_str(), __FILE__, __LINE__, "Symmetry missing.");

    } catch(exception &e) {
        fail_test(testname.str().c_str(), __FILE__, __LINE__, e.what());
    }
}

/** \test Elementwise multiplaction/division of two order-4 tensors
        with permutational symmetry and anti-symmetry and se_part / se_label.
 **/
void btod_mult_test::test_7(bool label, bool part,
        bool samesym, bool recip, bool doadd) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "btod_mult_test::test_7(" << label << ", " << part << ", "
            << samesym << ", " << recip << ", " << doadd << ")";

    std::string tns = tnss.str();

    if (label) {
        std::vector<std::string> irn(2);
        irn[0] = "g"; irn[1] = "u";
        point_group_table pg(tns, irn, irn[0]);
        pg.add_product(1, 1, 0);

        product_table_container::get_instance().add(pg);
    }

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 7; i2[3] = 7;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> msk1, msk2, msk;
    msk1[0] = true; msk1[1] = true;
    msk2[2] = true; msk2[3] = true;
    msk = msk1; msk |= msk2;
    bis.split(msk1, 2);
    bis.split(msk1, 5);
    bis.split(msk1, 7);
    bis.split(msk2, 2);
    bis.split(msk2, 4);
    bis.split(msk2, 6);

    block_tensor<4, double, allocator_t> bta(bis), btb(bis), btc(bis);
    symmetry<4, double> sym_ref(bis);

    // set up symmetries
    {
    block_tensor_ctrl<4, double> ca(bta), cb(btb), cc(btc);

    scalar_transf<double> tr0, tr1(-1.);
    se_perm<4, double> sp10(permutation<4>().permute(0, 1), tr0);
    se_perm<4, double> ap10(permutation<4>().permute(0, 1), tr1);
    se_perm<4, double> sp32(permutation<4>().permute(2, 3), tr0);
    se_perm<4, double> ap32(permutation<4>().permute(2, 3), tr1);

    if(recip) {
        //  For A/B, B cannot be asymmetric
        if(samesym) {
            ca.req_symmetry().insert(sp10);
            ca.req_symmetry().insert(sp32);
            cb.req_symmetry().insert(sp10);
            cb.req_symmetry().insert(sp32);
            cc.req_symmetry().insert(sp10);
            cc.req_symmetry().insert(sp32);
            sym_ref.insert(sp10);
            sym_ref.insert(sp32);
        } else {
            ca.req_symmetry().insert(ap10);
            ca.req_symmetry().insert(ap32);
            cb.req_symmetry().insert(sp10);
            cb.req_symmetry().insert(sp32);
            cc.req_symmetry().insert(ap10);
            cc.req_symmetry().insert(ap32);
            sym_ref.insert(ap10);
            sym_ref.insert(ap32);
        }
    } else {
        //  For A*B, A and B can be symmetric or asymmetric
        if(samesym) {
            ca.req_symmetry().insert(ap10);
            ca.req_symmetry().insert(ap32);
            cb.req_symmetry().insert(ap10);
            cb.req_symmetry().insert(ap32);
            cc.req_symmetry().insert(sp10);
            cc.req_symmetry().insert(sp32);
            sym_ref.insert(sp10);
            sym_ref.insert(sp32);
        } else {
            ca.req_symmetry().insert(ap10);
            ca.req_symmetry().insert(ap32);
            cb.req_symmetry().insert(sp10);
            cb.req_symmetry().insert(sp32);
            cc.req_symmetry().insert(ap10);
            cc.req_symmetry().insert(ap32);
            sym_ref.insert(ap10);
            sym_ref.insert(ap32);
        }
    }

    if (label) {
        se_label<4, double> sl(bis.get_block_index_dims(), tns);
        block_labeling<4> &bl = sl.get_labeling();
        bl.assign(msk, 0, 0);
        bl.assign(msk, 1, 1);
        bl.assign(msk, 2, 0);
        bl.assign(msk, 3, 1);

        evaluation_rule<4> r1;
        sequence<4, size_t> seq(1);
        product_rule<4> &pr1 = r1.new_product();
        pr1.add(seq, 0);
        sl.set_rule(r1);
        ca.req_symmetry().insert(sl);
        cc.req_symmetry().insert(sl);
        sym_ref.insert(sl);

        product_rule<4> &pr2 = r1.new_product();
        pr2.add(seq, 1);
        sl.set_rule(r1);
        cb.req_symmetry().insert(sl);
    }

    if (part) {
        se_part<4, double> sp1(bis, msk, 2), sp2(bis, msk, 2);
        libtensor::index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
            i0100, i1011, i0101, i1010, i0110, i1001, i0111, i1000;
        i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
        i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
        i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
        i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
        i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
        i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

        sp1.add_map(i0000, i1111);
        sp1.add_map(i0001, i1110);
        sp1.add_map(i0010, i1101);
        sp1.add_map(i0011, i1100);
        sp1.add_map(i0100, i1011);
        sp1.add_map(i0101, i1010);
        sp1.add_map(i0110, i1001);
        sp1.add_map(i0111, i1000);

        ca.req_symmetry().insert(sp1);
        cc.req_symmetry().insert(sp1);
        sym_ref.insert(sp1);

        sp2.add_map(i0000, i0001);
        sp2.add_map(i0001, i0010);
        sp2.add_map(i0010, i0011);
        sp2.add_map(i0011, i0100);
        sp2.add_map(i0100, i0101);
        sp2.add_map(i0101, i0110);
        sp2.add_map(i0110, i0111);
        sp2.add_map(i0111, i1000);
        sp2.add_map(i1000, i1001);
        sp2.add_map(i1001, i1010);
        sp2.add_map(i1010, i1011);
        sp2.add_map(i1011, i1100);
        sp2.add_map(i1100, i1101);
        sp2.add_map(i1101, i1110);
        sp2.add_map(i1110, i1111);

        cb.req_symmetry().insert(sp2);
    }

    }

    //  Fill in random data

    btod_random<4>().perform(bta);
    btod_random<4>().perform(btb);
    btod_random<4>().perform(btc);

    bta.set_immutable();
    btb.set_immutable();

    // Setup reference
    dense_tensor<4, double, allocator_t> ta(dims), tb(dims), tc(dims), tc_ref(dims);
    tod_btconv<4>(bta).perform(ta);
    tod_btconv<4>(btb).perform(tb);

    if (doadd) {
        tod_btconv<4>(btc).perform(tc_ref);
        tod_mult<4>(ta, tb, recip, 0.5).perform(false, tc_ref);
        btod_mult<4>(bta, btb, recip).perform(btc, 0.5);
    }
    else {
        tod_mult<4>(ta, tb, recip).perform(true, tc_ref);
        btod_mult<4>(bta, btb, recip).perform(btc);
    }

    tod_btconv<4>(btc).perform(tc);

    // Compare symmetry
    block_tensor_ctrl<4, double> ctrlc(btc);
    compare_ref<4>::compare(tns.c_str(), ctrlc.req_const_symmetry(), sym_ref);

    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        if (label) product_table_container::get_instance().erase(tns);

        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    } catch (...) {
        if (label) product_table_container::get_instance().erase(tns);
        throw;
    }

    if (label) product_table_container::get_instance().erase(tns);
}

/** \test Elementwise division of two 2-order tensors having 1 element blocks.
 **/
void btod_mult_test::test_8a(bool label, bool part)
        throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "btod_mult_test::test_8a(" << label << ", " << part << ")";

    std::string tns = tnss.str();


    if (label) {
        std::vector<std::string> irn(2);
        irn[0] = "g"; irn[1] = "u";
        point_group_table pg(tns, irn, irn[0]);
        pg.add_product(1, 1, 0);

        product_table_container::get_instance().add(pg);
    }

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 4); bis.split(m, 5); bis.split(m, 9);

    block_tensor<2, double, allocator_t> bta(bis), btb(bis), btc(bis);
    symmetry<2, double> sym_ref(bis);
    scalar_transf<double> tr0, tr1(-1.);

    // set up symmetries
    {
    block_tensor_ctrl<2, double> ca(bta), cb(btb);

    se_perm<2, double> sp(permutation<2>().permute(0, 1), tr0);
    se_perm<2, double> ap(permutation<2>().permute(0, 1), tr1);

    ca.req_symmetry().insert(ap);
    cb.req_symmetry().insert(sp);
    sym_ref.insert(ap);

    if (label) {
        se_label<2, double> sl(bis.get_block_index_dims(), tns);
        block_labeling<2> &bl = sl.get_labeling();
        bl.assign(m, 0, 0); bl.assign(m, 1, 1);
        bl.assign(m, 2, 0); bl.assign(m, 3, 1);
        evaluation_rule<2> r1;
        sequence<2, size_t> seq(1);
        product_rule<2> &pr1 = r1.new_product();
        pr1.add(seq, 0);
        sl.set_rule(r1);

        ca.req_symmetry().insert(sl);
        sym_ref.insert(sl);

        product_rule<2> &pr2 = r1.new_product();
        pr2.add(seq, 1);
        sl.set_rule(r1);
        cb.req_symmetry().insert(sl);
    }

    if (part) {
        se_part<2, double> spa(bis, m, 2), spb(bis, m, 2);
        libtensor::index<2> i00, i11, i01, i10;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;


        spa.add_map(i00, i11, tr0);
        spa.add_map(i01, i10, tr0);

        ca.req_symmetry().insert(spa);
        sym_ref.insert(spa);

        spb.add_map(i00, i01, tr0);
        spb.add_map(i01, i10, tr0);
        spb.add_map(i10, i11, tr0);

        cb.req_symmetry().insert(spb);
    }
    }

    //  Fill in random data
    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);
    bta.set_immutable();
    btb.set_immutable();

    // Setup reference
    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tc(dims), tc_ref(dims);

    tod_btconv<2>(bta).perform(ta);
    tod_btconv<2>(btb).perform(tb);

    tod_mult<2>(ta, tb, true, 4.0).perform(true, tc_ref);
    btod_mult<2> mult(bta, btb, true, 4.0);
    compare_ref<2>::compare(tns.c_str(), mult.get_symmetry(), sym_ref);

    mult.perform(btc);

    tod_btconv<2>(btc).perform(tc);
    block_tensor_ctrl<2, double> cc(btc);
    compare_ref<2>::compare(tns.c_str(), cc.req_const_symmetry(), sym_ref);
    compare_ref<2>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        if (label) product_table_container::get_instance().erase(tns);

        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    } catch (...) {
        if (label) product_table_container::get_instance().erase(tns);
        throw;
    }

    if (label) product_table_container::get_instance().erase(tns);
}



/** \test Elementwise division of two 4-order tensors having 1 element blocks.
 **/
void btod_mult_test::test_8b(bool label, bool part)
        throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "btod_mult_test::test_8b(" << label << ", " << part << ")";

    std::string tns = tnss.str();


    if (label) {
        std::vector<std::string> irn(2);
        irn[0] = "g"; irn[1] = "u";
        point_group_table pg(tns, irn, irn[0]);
        pg.add_product(1, 1, 0);

        product_table_container::get_instance().add(pg);
    }

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<4> i1, i2;
    i2[0] = 5; i2[1] = 5; i2[2] = 7; i2[3] = 7;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1, m2, m;
    m1[0] = true; m1[1] = true;
    m2[2] = true; m2[3] = true;
    m[0] = true; m[1] = true; m[2] = true; m[3] = true;
    bis.split(m1, 2); bis.split(m1, 3); bis.split(m1, 5);
    bis.split(m2, 3); bis.split(m2, 4); bis.split(m2, 7);

    block_tensor<4, double, allocator_t> bta(bis), btb(bis), btc(bis);
    symmetry<4, double> sym_ref(bis);

    // set up symmetries
    scalar_transf<double> tr0, tr1(-1.);
    {
    block_tensor_ctrl<4, double> ca(bta), cb(btb);

    se_perm<4, double> ap10(permutation<4>().permute(0, 1), tr1);
    se_perm<4, double> ap32(permutation<4>().permute(2, 3), tr1);
    se_perm<4, double> sp10(permutation<4>().permute(0, 1), tr0);
    se_perm<4, double> sp32(permutation<4>().permute(2, 3), tr0);

    ca.req_symmetry().insert(ap10);
    ca.req_symmetry().insert(ap32);
    cb.req_symmetry().insert(sp10);
    cb.req_symmetry().insert(sp32);
    sym_ref.insert(ap10);
    sym_ref.insert(ap32);

    if (label) {
        se_label<4, double> sl(bis.get_block_index_dims(), tns);
        block_labeling<4> &bl = sl.get_labeling();
        bl.assign(m, 0, 0); bl.assign(m, 1, 1);
        bl.assign(m, 2, 0); bl.assign(m, 3, 1);
        evaluation_rule<4> r1;
        sequence<4, size_t> seq(1);
        product_rule<4> &pr1 = r1.new_product();
        pr1.add(seq, 0);
        sl.set_rule(r1);

        ca.req_symmetry().insert(sl);
        sym_ref.insert(sl);

        product_rule<4> &pr2 = r1.new_product();
        pr2.add(seq, 1);
        sl.set_rule(r1);
        cb.req_symmetry().insert(sl);
    }

    if (part) {
        se_part<4, double> spa(bis, m, 2), spb(bis, m, 2);
        libtensor::index<4> i0000, i1111, i0001, i1110, i0010, i1101, i0011, i1100,
            i0100, i1011, i0101, i1010, i0110, i1001, i0111, i1000;
        i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
        i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
        i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
        i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
        i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
        i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;

        spa.add_map(i0000, i1111, tr0);
        spa.add_map(i0001, i1110, tr0);
        spa.add_map(i0010, i1101, tr0);
        spa.add_map(i0011, i1100, tr0);
        spa.add_map(i0100, i1011, tr0);
        spa.add_map(i0101, i1010, tr0);
        spa.add_map(i0110, i1001, tr0);
        spa.add_map(i0111, i1000, tr0);

        ca.req_symmetry().insert(spa);
        sym_ref.insert(spa);

        spb.add_map(i0000, i0001, tr0);
        spb.add_map(i0001, i0010, tr0);
        spb.add_map(i0010, i0011, tr0);
        spb.add_map(i0011, i0100, tr0);
        spb.add_map(i0100, i0101, tr0);
        spb.add_map(i0101, i0110, tr0);
        spb.add_map(i0110, i0111, tr0);
        spb.add_map(i0111, i1000, tr0);
        spb.add_map(i1000, i1001, tr0);
        spb.add_map(i1001, i1010, tr0);
        spb.add_map(i1010, i1011, tr0);
        spb.add_map(i1011, i1100, tr0);
        spb.add_map(i1100, i1101, tr0);
        spb.add_map(i1101, i1110, tr0);
        spb.add_map(i1110, i1111, tr0);

        cb.req_symmetry().insert(spb);
    }
    }

    //  Fill in random data
    btod_random<4>().perform(bta);
    btod_random<4>().perform(btb);
    bta.set_immutable();
    btb.set_immutable();

    // Setup reference
    dense_tensor<4, double, allocator_t> ta(dims), tb(dims), tc(dims), tc_ref(dims);

    tod_btconv<4>(bta).perform(ta);
    tod_btconv<4>(btb).perform(tb);

    tod_mult<4>(ta, tb, true, 4.0).perform(true, tc_ref);
    btod_mult<4> mult(bta, btb, true, 4.0);
    compare_ref<4>::compare(tns.c_str(), mult.get_symmetry(), sym_ref);

    mult.perform(btc);

    tod_btconv<4>(btc).perform(tc);

    block_tensor_ctrl<4, double> cc(btc);
    compare_ref<4>::compare(tns.c_str(), cc.req_const_symmetry(), sym_ref);
    compare_ref<4>::compare(tns.c_str(), tc, tc_ref, 1e-15);

    } catch(exception &e) {
        if (label) product_table_container::get_instance().erase(tns);

        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    } catch (...) {
        if (label) product_table_container::get_instance().erase(tns);
        throw;
    }

    if (label) product_table_container::get_instance().erase(tns);
}





} // namespace libtensor
