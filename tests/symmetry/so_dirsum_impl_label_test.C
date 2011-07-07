#include <libtensor/btod/transf_double.h>
#include <libtensor/symmetry/label/point_group_table.h>
#include <libtensor/symmetry/label/so_dirsum_impl_label.h>
#include "../compare_ref.h"
#include "so_dirsum_impl_label_test.h"

namespace libtensor {

const char *so_dirsum_impl_label_test::k_table_id = "S6";

void so_dirsum_impl_label_test::perform() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_impl_label_test::perform()";

    try {

        point_group_table s6(k_table_id, 4);
        point_group_table::label_t ag = 0, eg = 1, au = 2, eu = 3;
        s6.add_product(ag, ag, ag);
        s6.add_product(ag, eg, eg);
        s6.add_product(ag, au, au);
        s6.add_product(ag, eu, eu);
        s6.add_product(eg, eg, ag);
        s6.add_product(eg, eg, eg);
        s6.add_product(eg, au, eu);
        s6.add_product(eg, eu, au);
        s6.add_product(eg, eu, eu);
        s6.add_product(au, au, ag);
        s6.add_product(au, eu, eg);
        s6.add_product(eu, eu, ag);
        s6.add_product(eu, eu, eg);
        s6.check();
        product_table_container::get_instance().add(s6);

    } catch (exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

    try {

        test_empty_1();
        test_empty_2(true);
        test_empty_2(false);
        test_empty_3(true);
        test_empty_3(false);
        test_nn_1();
        test_nn_2();
        test_nn_3();

    } catch (libtest::test_exception) {
        product_table_container::get_instance().erase(k_table_id);
        throw;
    }

    product_table_container::get_instance().erase(k_table_id);
}


/**	\test Tests that the direct sum of two empty group yields an empty
        group of a higher order
 **/
void so_dirsum_impl_label_test::test_empty_1() throw(libtest::test_exception) {

    static const char *testname =
            "so_dirsum_impl_label_test::test_empty_1()";

    typedef se_label<2, double> se2_t;
    typedef se_label<3, double> se3_t;
    typedef se_label<5, double> se5_t;
    typedef so_dirsum<2, 3, double> so_t;
    typedef symmetry_operation_impl<so_t, se5_t> so_impl_t;

    try {

        index<5> i1c, i2c;
        i2c[0] = 3; i2c[1] = 3; i2c[2] = 3; i2c[3] = 3; i2c[4] = 3;
        block_index_space<5> bisc(dimensions<5>(index_range<5>(i1c, i2c)));

        mask<5> mc;
        mc[0] = true; mc[1] = true; mc[2] = true; mc[3] = true; mc[4] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        symmetry_element_set<2, double> seta(se2_t::k_sym_type);
        symmetry_element_set<3, double> setb(se3_t::k_sym_type);
        symmetry_element_set<5, double> setc(se5_t::k_sym_type);
        symmetry_element_set<5, double> setc_ref(se5_t::k_sym_type);

        permutation<5> px;
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_impl_t().perform(params);

        compare_ref<5>::compare(testname, bisc, setc, setc_ref);

        if(! setc.is_empty()) {
            fail_test(testname, __FILE__, __LINE__, "Expected an empty set.");
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/**	\test Direct sum of a group with Au symmetry in 2-space and an empty
        group (1-space) forming a 3-space.
 **/
void so_dirsum_impl_label_test::test_empty_2(
        bool perm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_impl_label_test::test_empty_2(" << perm << ")";
    std::string tns = tnss.str();

    typedef se_label<1, double> se1_t;
    typedef se_label<2, double> se2_t;
    typedef se_label<3, double> se3_t;
    typedef so_dirsum<2, 1, double> so_t;
    typedef symmetry_operation_impl<so_t, se3_t> so_impl_t;

    try {

        index<2> i1a, i2a; i2a[0] = 3; i2a[1] = 3;
        index<3> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3;

        block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
        block_index_space<3> bisc(dimensions<3>(index_range<3>(i1c, i2c)));

        mask<2> ma; ma[0] = true; ma[1] = true;
        bisa.split(ma, 1); bisa.split(ma, 2); bisa.split(ma, 3);
        mask<3> mc; mc[0] = true; mc[1] = true; mc[2] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        dimensions<2> bidimsa = bisa.get_block_index_dims();
        dimensions<3> bidimsc = bisc.get_block_index_dims();

        se2_t elema(bidimsa);
        {
            label_set<2> &ssa = elema.create_subset(k_table_id);
            ssa.add_intrinsic(2);
            for (size_t i = 0; i < 4; i++) {
                ssa.assign(ma, i, i);
            }
        }

        symmetry_element_set<2, double> seta(se2_t::k_sym_type);
        symmetry_element_set<1, double> setb(se1_t::k_sym_type);
        symmetry_element_set<3, double> setc(se3_t::k_sym_type);

        seta.insert(elema);

        permutation<3> px;
        if (perm) px.permute(0, 1).permute(1, 2);
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_impl_t().perform(params);

        if(setc.is_empty()) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        symmetry<3, double> symc(bisc);
        for (symmetry_element_set<3, double>::iterator it = setc.begin();
                it != setc.end(); it++) {

            symc.insert(setc.get_elem(it));
        }

        orbit_list<3, double> olc(symc);
        abs_index<3> ai(bidimsc);
        do {
            size_t i = ai.get_abs_index();
            if (! olc.contains(i)) {

                std::ostringstream oss;
                oss << "Canonical index " << ai.get_index()
                        << " is absent from result.";
                fail_test(tns.c_str(), __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}

/**	\test Direct sum of an empty group (1-space) and a group with Au symmetry
        in 2-space forming a 3-space.
 **/
void so_dirsum_impl_label_test::test_empty_3(
        bool perm) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "so_dirsum_impl_label_test::test_empty_3(" << perm << ")";
    std::string tns = tnss.str();

    typedef se_label<1, double> se1_t;
    typedef se_label<2, double> se2_t;
    typedef se_label<3, double> se3_t;
    typedef so_dirsum<1, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se3_t> so_impl_t;

    try {

        index<2> i1b, i2b; i2b[0] = 3; i2b[1] = 3;
        index<3> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3;

        block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));
        block_index_space<3> bisc(dimensions<3>(index_range<3>(i1c, i2c)));

        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 1); bisb.split(mb, 2); bisb.split(mb, 3);
        mask<3> mc; mc[0] = true; mc[1] = true; mc[2] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        dimensions<2> bidimsb = bisb.get_block_index_dims();
        dimensions<3> bidimsc = bisc.get_block_index_dims();

        se2_t elemb(bidimsb);
        {
            label_set<2> &ssb = elemb.create_subset(k_table_id);
            ssb.add_intrinsic(2);
            for (size_t i = 0; i < 4; i++) {
                ssb.assign(mb, i, i);
            }
        }

        symmetry_element_set<1, double> seta(se1_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<3, double> setc(se3_t::k_sym_type);

        setb.insert(elemb);

        permutation<3> px;
        if (perm) px.permute(0, 1).permute(1, 2);
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_impl_t().perform(params);

        if(setc.is_empty()) {
            fail_test(tns.c_str(), __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        symmetry<3, double> symc(bisc);
        for (symmetry_element_set<3, double>::iterator it = setc.begin();
                it != setc.end(); it++) {

            symc.insert(setc.get_elem(it));
        }

        orbit_list<3, double> olc(symc);
        abs_index<3> ai(bidimsc);
        do {
            size_t i = ai.get_abs_index();
            if (! olc.contains(i)) {

                std::ostringstream oss;
                oss << "Canonical index " << ai.get_index()
                        << " is absent from result.";
                fail_test(tns.c_str(), __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());

    } catch(exception &e) {
        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Direct sum of a group with Ag symmetry in 1-space and a group
        with Eu symmetry in 2-space.
 **/
void so_dirsum_impl_label_test::test_nn_1() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_impl_label_test::test_nn_1()";

    typedef se_label<1, double> se1_t;
    typedef se_label<2, double> se2_t;
    typedef se_label<3, double> se3_t;
    typedef so_dirsum<1, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se3_t> so_impl_t;

    try {

        index<1> i1a, i2a; i2a[0] = 3; ;
        index<2> i1b, i2b; i2b[0] = 3; i2b[1] = 3;
        index<3> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3;

        block_index_space<1> bisa(dimensions<1>(index_range<1>(i1a, i2a)));
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));
        block_index_space<3> bisc(dimensions<3>(index_range<3>(i1c, i2c)));

        mask<1> ma; ma[0] = true;
        bisa.split(ma, 1); bisa.split(ma, 2); bisa.split(ma, 3);
        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 1); bisb.split(mb, 2); bisb.split(mb, 3);
        mask<3> mc; mc[0] = true; mc[1] = true; mc[2] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        dimensions<1> bidimsa = bisa.get_block_index_dims();
        dimensions<2> bidimsb = bisb.get_block_index_dims();
        dimensions<3> bidimsc = bisc.get_block_index_dims();

        se1_t elema(bidimsa);
        {
            label_set<1> &ssa = elema.create_subset(k_table_id);
            ssa.add_intrinsic(0);
            for (size_t i = 0; i < 4; i++) {
                ssa.assign(ma, i, i);
            }
        }

        se2_t elemb(bidimsb);
        {
            label_set<2> &ssb = elemb.create_subset(k_table_id);
            ssb.add_intrinsic(3);
            for (size_t i = 0; i < 4; i++) {
                ssb.assign(mb, i, i);
            }
        }

        symmetry_element_set<1, double> seta(se1_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<3, double> setc(se3_t::k_sym_type);

        seta.insert(elema);
        setb.insert(elemb);

        permutation<3> px;
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_impl_t().perform(params);

        if(setc.is_empty()) {
            fail_test(testname, __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        symmetry<3, double> symc(bisc);
        for (symmetry_element_set<3, double>::iterator it = setc.begin();
                it != setc.end(); it++) {

            symc.insert(setc.get_elem(it));
        }

        orbit_list<3, double> olc(symc);

        std::vector<bool> rx(bidimsc.get_size(), false);
        for (size_t i = 0; i < 4; i++) {
            rx[i * 16 +  3] = rx[i * 16 +  6] = rx[i * 16 +  7] = true;
            rx[i * 16 +  9] = rx[i * 16 + 12] = rx[i * 16 + 13] = true;
        }
        for (size_t i = 0; i < 16; i++) {
            rx[i] = true;
        }

        abs_index<3> ai(bidimsc);
        do {
            size_t i = ai.get_abs_index();
            if (rx[i] != olc.contains(i)) {

                std::ostringstream oss;
                oss << "Canonical index " << ai.get_index();
                if (rx[i])
                    oss << " is absent from result.";
                else
                    oss << " missing in reference.";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Direct sum of a group in 1-space with Eg symmetry and a group in
        2-space with Au symmetry. The result is permuted with [012->120].
 **/
void so_dirsum_impl_label_test::test_nn_2() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_impl_label_test::test_nn_2()";

    typedef se_label<1, double> se1_t;
    typedef se_label<2, double> se2_t;
    typedef se_label<3, double> se3_t;
    typedef so_dirsum<1, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se3_t> so_impl_t;

    try {

        index<1> i1a, i2a; i2a[0] = 3; ;
        index<2> i1b, i2b; i2b[0] = 3; i2b[1] = 3;
        index<3> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3;

        block_index_space<1> bisa(dimensions<1>(index_range<1>(i1a, i2a)));
        block_index_space<2> bisb(dimensions<2>(index_range<2>(i1b, i2b)));
        block_index_space<3> bisc(dimensions<3>(index_range<3>(i1c, i2c)));

        mask<1> ma; ma[0] = true;
        bisa.split(ma, 1); bisa.split(ma, 2); bisa.split(ma, 3);
        mask<2> mb; mb[0] = true; mb[1] = true;
        bisb.split(mb, 1); bisb.split(mb, 2); bisb.split(mb, 3);
        mask<3> mc; mc[0] = true; mc[1] = true; mc[2] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        dimensions<1> bidimsa = bisa.get_block_index_dims();
        dimensions<2> bidimsb = bisb.get_block_index_dims();
        dimensions<3> bidimsc = bisc.get_block_index_dims();

        se1_t elema(bidimsa);
        {
            label_set<1> &ssa = elema.create_subset(k_table_id);
            ssa.add_intrinsic(1);
            for (size_t i = 0; i < 4; i++) {
                ssa.assign(ma, i, i);
            }
        }

        se2_t elemb(bidimsb);
        {
            label_set<2> &ssb = elemb.create_subset(k_table_id);
            ssb.add_intrinsic(2);
            for (size_t i = 0; i < 4; i++) {
                ssb.assign(mb, i, i);
            }
        }

        symmetry_element_set<1, double> seta(se1_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<3, double> setc(se3_t::k_sym_type);

        seta.insert(elema);
        setb.insert(elemb);

        permutation<3> px;
        px.permute(0, 1).permute(1, 2);
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_impl_t().perform(params);

        if(setc.is_empty()) {
            fail_test(testname, __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        symmetry<3, double> symc(bisc);
        for (symmetry_element_set<3, double>::iterator it = setc.begin();
                it != setc.end(); it++) {

            symc.insert(setc.get_elem(it));
        }

        orbit_list<3, double> olc(symc);

        bidimsc.permute(px);
        std::vector<bool> rx(bidimsc.get_size(), false);
        for (size_t i = 0; i < 4; i++) {
            rx[ 8 + i] = rx[28 + i] = true;
            rx[32 + i] = rx[52 + i] = true;
        }
        for (size_t i = 0; i < 16; i++) {
            rx[i * 4 + 1] = true;
        }

        abs_index<3> ai(bidimsc);
        do {
            size_t i = ai.get_abs_index();
            if (rx[i] != olc.contains(i)) {

                std::ostringstream oss;
                oss << "Canonical index " << ai.get_index();
                if (rx[i])
                    oss << " is absent from result.";
                else
                    oss << " missing in reference.";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Direct sum of two groups in 2-space one with two symmetry elements.
 **/
void so_dirsum_impl_label_test::test_nn_3() throw(libtest::test_exception) {

    static const char *testname = "so_dirsum_impl_label_test::test_nn_3()";

    typedef se_label<2, double> se2_t;
    typedef se_label<4, double> se4_t;
    typedef so_dirsum<2, 2, double> so_t;
    typedef symmetry_operation_impl<so_t, se4_t> so_impl_t;

    try {

        index<2> i1a, i2a; i2a[0] = 3; i2a[1] = 3;
        index<4> i1c, i2c; i2c[0] = 3; i2c[1] = 3; i2c[2] = 3; i2c[3] = 3;

        block_index_space<2> bisa(dimensions<2>(index_range<2>(i1a, i2a)));
        block_index_space<4> bisc(dimensions<4>(index_range<4>(i1c, i2c)));

        mask<2> ma; ma[0] = true; ma[1] = true;
        bisa.split(ma, 1); bisa.split(ma, 2); bisa.split(ma, 3);
        mask<4> mc; mc[0] = true; mc[1] = true; mc[2] = true; mc[3] = true;
        bisc.split(mc, 1); bisc.split(mc, 2); bisc.split(mc, 3);

        dimensions<2> bidimsa = bisa.get_block_index_dims();
        dimensions<4> bidimsc = bisc.get_block_index_dims();

        se2_t elema1(bidimsa), elema2(bidimsa);
        {
            mask<2> m10, m01;
            m10[0] = true; m01[1] = true;
            label_set<2> &ssa1 = elema1.create_subset(k_table_id);
            label_set<2> &ssa2 = elema2.create_subset(k_table_id);
            ssa1.add_intrinsic(1);
            ssa2.add_intrinsic(2);
            for (size_t i = 0; i < 4; i++) {
                ssa1.assign(m10, i, i);
                ssa2.assign(m01, i, i);
            }
            ssa1.set_mask(m10);
            ssa2.set_mask(m01);
        }

        se2_t elemb(bidimsa);
        {
            label_set<2> &ssb = elemb.create_subset(k_table_id);
            ssb.add_intrinsic(1);
            for (size_t i = 0; i < 4; i++) {
                ssb.assign(ma, i, i);
            }
        }

        symmetry_element_set<2, double> seta(se2_t::k_sym_type);
        symmetry_element_set<2, double> setb(se2_t::k_sym_type);
        symmetry_element_set<4, double> setc(se4_t::k_sym_type);

        seta.insert(elema1);
        seta.insert(elema2);
        setb.insert(elemb);

        permutation<4> px;
        symmetry_operation_params<so_t> params(seta, setb, px, bisc, setc);

        so_impl_t().perform(params);

        if(setc.is_empty()) {
            fail_test(testname, __FILE__, __LINE__,
                    "Expected a non-empty set.");
        }

        symmetry<4, double> symc(bisc);
        for (symmetry_element_set<4, double>::iterator it = setc.begin();
                it != setc.end(); it++) {

            symc.insert(setc.get_elem(it));
        }

        orbit_list<4, double> olc(symc);

        std::vector<bool> rx(bidimsc.get_size(), false);
        for (size_t i = 0; i < 16; i++) {
            rx[96 + i] = true;
            rx[i * 16 +  1] = rx[i * 16 +  4] = rx[i * 16 +  5] = true;
            rx[i * 16 + 11] = rx[i * 16 + 14] = rx[i * 16 + 15] = true;
        }

        abs_index<4> ai(bidimsc);
        do {
            size_t i = ai.get_abs_index();
            if (rx[i] != olc.contains(i)) {

                std::ostringstream oss;
                oss << "Canonical index " << ai.get_index();
                if (rx[i])
                    oss << " is absent from result.";
                else
                    oss << " missing in reference.";
                fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
            }

        } while (ai.inc());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
