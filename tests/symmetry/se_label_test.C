#include <sstream>
#include <libtensor/btod/transf_double.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/symmetry/label/point_group_table.h>
#include "se_label_test.h"

namespace libtensor {

void se_label_test::perform() throw(libtest::test_exception) {

    std::string s6 = setup_s6_symmetry();
    try {

         test_basic_1(s6);
         test_allowed_1(s6);
         test_allowed_2(s6);
         test_allowed_3(s6);
         test_permute_1(s6);
         test_permute_2(s6);

    } catch (libtest::test_exception) {
        product_table_container::get_instance().erase(s6);
        throw;
    }

    product_table_container::get_instance().erase(s6);
}

/** \test Tests setting evaluation rules
 **/
void se_label_test::test_basic_1(
    const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "se_label_test::test_basic_1(" << table_id << ")";
    std::string tns = tnss.str();

    index<3> i1, i2;
    i2[0] = 1; i2[1] = 1; i2[2] = 1;
    dimensions<3> bidims(index_range<3>(i1, i2));

    se_label<3, double> el(bidims, table_id);

    // Simplest rule
    el.set_rule(0);
    const evaluation_rule &r1 = el.get_rule();
    evaluation_rule::rule_iterator it = r1.begin();
    const evaluation_rule::label_group &intr1 = r1.get_intrinsic(it);
    const std::vector<size_t> &o1 = r1.get_eval_order(it);
    if (intr1.size() != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# intr1");
    if (intr1[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "intr1");
    if (o1.size() != 4)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# o1");
    if (o1[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o1[0]");
    if (o1[1] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o1[1]");
    if (o1[2] != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o1[2]");
    if (o1[3] != evaluation_rule::k_intrinsic)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o1[3]");
    it++;
    if (it != r1.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "# rules");

    // Simple rule with different order
    permutation<3> p2; p2.permute(0, 1).permute(1, 2);
    el.set_rule(2, p2, 1);
    const evaluation_rule &r2 = el.get_rule();
    it = r2.begin();
    const evaluation_rule::label_group &intr2 = r2.get_intrinsic(it);
    const std::vector<size_t> &o2 = r2.get_eval_order(it);
    if (intr2.size() != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# intr2");
    if (intr2[0] != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "intr2");
    if (o2.size() != 4)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# o2");
    if (o2[0] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o2[0]");
    if (o2[1] != evaluation_rule::k_intrinsic)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o2[1]");
    if (o2[2] != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o2[2]");
    if (o2[3] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o2[3]");
    it++;
    if (it != r2.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "# rules");

    evaluation_rule::label_group lg3(2, 0); lg3[1] = 2;
    el.set_rule(lg3, p2, 2);
    const evaluation_rule &r3 = el.get_rule();
    it = r3.begin();
    const evaluation_rule::label_group &intr3 = r3.get_intrinsic(it);
    const std::vector<size_t> &o3 = r3.get_eval_order(it);
    if (intr3.size() != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# intr3");
    if (intr3[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "intr3[0]");
    if (intr3[1] != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "intr3[1]");
    if (o3.size() != 4)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# o3");
    if (o3[0] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o3[0]");
    if (o3[1] != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o3[1]");
    if (o3[2] != evaluation_rule::k_intrinsic)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o3[2]");
    if (o3[3] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o3[3]");
    it++;
    if (it != r3.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "# rules");


    evaluation_rule r4ref;
    evaluation_rule::label_group lg4a(2, 0), lg4b(1, 1);
    lg4a[1] = 2;
    std::vector<size_t> v4a(3,0), v4b(2,0); 
    v4a[0] = 1; v4a[1] = 0; v4a[2] = evaluation_rule::k_intrinsic;
    v4b[0] = 2; v4b[1] = evaluation_rule::k_intrinsic;
    evaluation_rule::rule_id rid4a = r4ref.add_rule(lg4a, v4a);
    evaluation_rule::rule_id rid4b = r4ref.add_rule(lg4b, v4b);
    r4ref.add_product(rid4a);
    r4ref.add_to_product(0, rid4b);
    el.set_rule(r4ref);
    const evaluation_rule &r4 = el.get_rule();
    it = r4.begin();
    const evaluation_rule::label_group &intr4a = r4.get_intrinsic(it);
    const std::vector<size_t> &o4a = r4.get_eval_order(it);
    if (intr4a.size() != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# intr4a");
    if (intr4a[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "intr4a[0]");
    if (intr4a[1] != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "intr4a[1]");
    if (o4a.size() != 3)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# o3");
    if (o4a[0] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o3[0]");
    if (o4a[1] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o3[1]");
    if (o4a[2] != evaluation_rule::k_intrinsic)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o3[2]");
    it++;
    const evaluation_rule::label_group &intr4b = r4.get_intrinsic(it);
    const std::vector<size_t> &o4b = r4.get_eval_order(it);
    if (intr4b.size() != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# intr4b");
    if (intr4b[0] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "intr4b[0]");
    if (o4b.size() != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# o4b");
    if (o4b[0] != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o4b[0]");
    if (o4b[1] != evaluation_rule::k_intrinsic)
        fail_test(tns.c_str(), __FILE__, __LINE__, "o4b[1]");
    it++;
    if (it != r4.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "# rules");
}

/** \test Four blocks, all labeled, different index types, basic rules only
 **/
void se_label_test::test_allowed_1(
    const std::string &table_id) throw(libtest::test_exception) {
    
    std::ostringstream tnss;
    tnss << "se_label_test::test_allowed_1(" << table_id << ")";
    std::string tns = tnss.str();

    index<2> i1, i2;
    i2[0] = 3; i2[1] = 3;
    dimensions<2> bidims(index_range<2>(i1, i2));
    se_label<2, double> el1(bidims, table_id);
    
    { // Add the labels
        mask<2> m01, m10;
        m10[0] = true; m01[1] = true;
        block_labeling<2> &bl = el1.get_labeling();

        for (size_t i = 0; i < 4; i++) bl.assign(m10, i, i);
        bl.assign(m01, 0, 0); // ag
        bl.assign(m01, 1, 2); // au
        bl.assign(m01, 2, 1); // eg
        bl.assign(m01, 3, 3); // eu
    }

    se_label<2, double> el2(el1), el3(el1), el4(el1);
    
    el1.set_rule(0);
    el2.set_rule(1);
    el3.set_rule(2);
    el4.set_rule(3);

    std::vector<bool> ex1(bidims.get_size(), false);
    std::vector<bool> ex2(bidims.get_size(), false);
    std::vector<bool> ex3(bidims.get_size(), false);
    std::vector<bool> ex4(bidims.get_size(), false);

    ex1[0] = ex1[6] = ex1[9] = ex1[15] = true;
    ex2[2] = ex2[4] = ex2[6] = ex2[11] = ex2[13] = ex2[15] = true;
    ex3[1] = ex3[7] = ex3[8] = ex3[14] = true;
    ex4[3] = ex4[5] = ex4[7] = ex4[10] = ex4[12] = ex4[14] = true;

    check_allowed(tns.c_str(), "el1", el1, ex1);
    check_allowed(tns.c_str(), "el2", el2, ex2);
    check_allowed(tns.c_str(), "el3", el3, ex3);
    check_allowed(tns.c_str(), "el4", el4, ex4);
}

/** \test Four blocks, one dim unlabeled, rules including and not including
        unlabeled dimension
 **/
void se_label_test::test_allowed_2(
    const std::string &table_id) throw(libtest::test_exception) {
    
    std::ostringstream tnss;
    tnss << "se_label_test::test_allowed_2(" << table_id << ")";
    std::string tns = tnss.str();

    index<2> i1, i2;
    i2[0] = 3; i2[1] = 3;
    dimensions<2> bidims(index_range<2>(i1, i2));
    se_label<2, double> el1(bidims, table_id);

    { // Add the labels 
        block_labeling<2> &bl = el1.get_labeling();
        mask<2> m01, m10;
        m10[0] = true; m01[1] = true;

        for (size_t i = 0; i < 4; i++) bl.assign(m10, i, i);
    }

    se_label<2, double> el2(el1);        

    el1.set_rule(0);

    evaluation_rule rule;
    std::vector<size_t> order(2, 0);
    order[1] = evaluation_rule::k_intrinsic;
    evaluation_rule::label_group lg(1, 0);
    evaluation_rule::rule_id rid = rule.add_rule(lg, order);
    rule.add_product(rid);
    el2.set_rule(rule);

    std::vector<bool> ex1(bidims.get_size(), true);
    std::vector<bool> ex2(bidims.get_size(), false);

    ex2[0] = ex2[1] = ex2[2] = ex2[3] = true;

    check_allowed(tns.c_str(), "el1", el1, ex1);
    check_allowed(tns.c_str(), "el2", el2, ex2);
}

/** \test Four blocks, all dims labeled, composite rules
 **/
void se_label_test::test_allowed_3(
    const std::string &table_id) throw(libtest::test_exception) {
    
    std::ostringstream tnss;
    tnss << "se_label_test::test_allowed_3(" << table_id << ")";
    std::string tns = tnss.str();

    index<2> i1, i2;
    i2[0] = 3; i2[1] = 3;
    dimensions<2> bidims(index_range<2>(i1, i2));
    se_label<2, double> el1(bidims, table_id);

    { // Add the labels
        block_labeling<2> &bl = el1.get_labeling();
        mask<2> m01, m10;
        m10[0] = true; m01[1] = true;

        for (size_t i = 0; i < 4; i++) bl.assign(m10, i, i);
        bl.assign(m01, 0, 0); // ag
        bl.assign(m01, 1, 2); // au
        bl.assign(m01, 2, 1); // eg
        bl.assign(m01, 3, 3); // eu
    }

    se_label<2, double> el2(el1);        

    evaluation_rule r1, r2;
    std::vector<size_t> oa(2, 0), ob(2, 1);
    oa[1] = ob[1] = evaluation_rule::k_intrinsic;
    evaluation_rule::label_group lga(1, 0), lgb(1, 1);
    evaluation_rule::rule_id rid1a = r1.add_rule(lga, oa);
    evaluation_rule::rule_id rid1b = r1.add_rule(lgb, ob);
    evaluation_rule::rule_id rid2a = r2.add_rule(lga, oa);
    evaluation_rule::rule_id rid2b = r2.add_rule(lgb, ob);
    r1.add_product(rid1a);
    r1.add_to_product(0, rid1b);
    r2.add_product(rid2a);
    r2.add_product(rid2b);

    el1.set_rule(r1);
    el2.set_rule(r2);

    std::vector<bool> ex1(bidims.get_size(), false);
    std::vector<bool> ex2(bidims.get_size(), false);

    ex1[2] = true;
    ex2[0] = ex2[1] = ex2[2] = ex2[3] = 
        ex2[6] = ex2[10] = ex2[14] = true;

    check_allowed(tns.c_str(), "el1", el1, ex1);
    check_allowed(tns.c_str(), "el2", el2, ex2);
}

/** \test Four blocks, all labeled, different index types, basic rules only,
        permute
 **/
void se_label_test::test_permute_1(
    const std::string &table_id) throw(libtest::test_exception) {
    
    std::ostringstream tnss;
    tnss << "se_label_test::test_permute_1(" << table_id << ")";
    std::string tns = tnss.str();

    index<3> i1, i2;
    i2[0] = 3; i2[1] = 3; i2[2] = 3;
    dimensions<3> bidims(index_range<3>(i1, i2));
    se_label<3, double> el1(bidims, table_id);

    {
        block_labeling<3> &bl = el1.get_labeling();
        mask<3> m011, m100;
        m100[0] = true; m011[1] = true; m011[2] = true;

        for (size_t i = 0; i < 4; i++) bl.assign(m100, i, i);
        bl.assign(m011, 0, 0); // ag
        bl.assign(m011, 1, 2); // au
        bl.assign(m011, 2, 1); // eg
        bl.assign(m011, 3, 3); // eu
    }

    se_label<3, double> el2(el1), el3(el1), el4(el1);
    
    el1.set_rule(0);
    el2.set_rule(1);
    el3.set_rule(2);
    el4.set_rule(3);

    permutation<3> p1, p2; 
    p1.permute(0, 1); p2.permute(0, 1).permute(1, 2);

    el1.permute(p1);
    el2.permute(p2);
    el3.permute(p2);
    el4.permute(p1);

    std::vector<bool> ex1(bidims.get_size(), false);
    std::vector<bool> ex2(bidims.get_size(), false);
    std::vector<bool> ex3(bidims.get_size(), false);
    std::vector<bool> ex4(bidims.get_size(), false);

    ex1[0] = ex1[6] = ex1[9] = ex1[15] = 
        ex1[17] = ex1[23] = ex1[24] = ex1[30] = 
        ex1[34] = ex1[36] = ex1[38] = ex1[43] = ex1[45] = ex1[47] = 
        ex1[51] = ex1[53] = ex1[55] = ex1[58] = ex1[60] = ex1[62] = true;
    ex2[1] = ex2[7] = ex2[8] = ex2[9] = ex2[14] = ex2[15] = 
        ex2[19] = ex2[21] = ex2[26] = ex2[27] = ex2[28] = ex2[29] = 
        ex2[32] = ex2[33] = ex2[38] = ex2[39] = 
        ex2[40] = ex2[41] = ex2[46] = ex2[47] = 
        ex2[50] = ex2[51] = ex2[52] = ex2[53] = 
        ex2[58] = ex2[59] = ex2[60] = ex2[61] = true;
    ex3[2] = ex3[4] = ex3[11] = ex3[13] = 
        ex3[16] = ex3[22] = ex3[25] = ex3[31] = 
        ex3[35] = ex3[37] = ex3[42] = ex3[43] = ex3[44] = ex3[45] = 
        ex3[49] = ex3[55] = ex3[56] = ex3[57] = ex3[62] = ex3[63] = true;
    ex4[3] = ex4[5] = ex4[7] = ex4[10] = ex4[12] = ex4[14] = 
        ex4[18] = ex4[20] = ex4[22] = ex4[27] = ex4[29] = ex4[31] = 
        ex4[33] = ex4[35] = ex4[37] = ex4[39] = 
        ex4[40] = ex4[42] = ex4[44] = ex4[46] = 
        ex4[48] = ex4[50] = ex4[52] = ex4[54] = 
        ex4[57] = ex4[59] = ex4[61] = ex4[63] = true;

    check_allowed(tns.c_str(), "el1", el1, ex1);
    check_allowed(tns.c_str(), "el2", el2, ex2);
    check_allowed(tns.c_str(), "el3", el3, ex3);
    check_allowed(tns.c_str(), "el4", el4, ex4);
}

/** \test Four blocks, all dims labeled, composite rules, permuted
 **/
void se_label_test::test_permute_2(
    const std::string &table_id) throw(libtest::test_exception) {
    
    std::ostringstream tnss;
    tnss << "se_label_test::test_permute_2(" << table_id << ")";
    std::string tns = tnss.str();

    index<2> i1, i2;
    i2[0] = 3; i2[1] = 3;
    dimensions<2> bidims(index_range<2>(i1, i2));
    se_label<2, double> el1(bidims, table_id);

    { // Add the labels
        block_labeling<2> &bl = el1.get_labeling();
        mask<2> m01, m10;
        m10[0] = true; m01[1] = true;

        for (size_t i = 0; i < 4; i++) bl.assign(m10, i, i);
        bl.assign(m01, 0, 0); // ag
        bl.assign(m01, 1, 2); // au
        bl.assign(m01, 2, 1); // eg
        bl.assign(m01, 3, 3); // eu
    }
   
    se_label<2, double> el2(el1);        

    evaluation_rule r1, r2;
    std::vector<size_t> oa(2, 0), ob(2, 1);
    oa[1] = ob[1] = evaluation_rule::k_intrinsic;
    evaluation_rule::label_group lga(1, 0), lgb(1, 1);
    evaluation_rule::rule_id rid1a = r1.add_rule(lga, oa);
    evaluation_rule::rule_id rid1b = r1.add_rule(lgb, ob);
    evaluation_rule::rule_id rid2a = r2.add_rule(lga, oa);
    evaluation_rule::rule_id rid2b = r2.add_rule(lgb, ob);
    r1.add_product(rid1a);
    r1.add_to_product(0, rid1b);
    r2.add_product(rid2a);
    r2.add_product(rid2b);

    el1.set_rule(r1);
    el2.set_rule(r2);

    permutation<2> p; p.permute(0, 1);
    el1.permute(p);
    el2.permute(p);

    std::vector<bool> ex1(bidims.get_size(), false);
    std::vector<bool> ex2(bidims.get_size(), false);

    ex1[8] = true;
    ex2[0] = ex2[4] = ex2[8] = ex2[12] = 
        ex2[9] = ex2[10] = ex2[11] = true;

    check_allowed(tns.c_str(), "el1", el1, ex1);
    check_allowed(tns.c_str(), "el2", el2, ex2);
}

/** \brief Setup the product table for S6 point group symmetry

     \return Table ID
 **/
std::string se_label_test::setup_s6_symmetry() {

    try {

        point_group_table s6("s6", 4);
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
        fail_test("se_label_test::setup_s6_symmetry()", __FILE__, __LINE__,
                e.what());
    }

    return "s6";
}

template<size_t N>
void se_label_test::check_allowed(const char *testname, const char *sename,
        const se_label<N, double> &se, const std::vector<bool> &expected) 
    throw(libtest::test_exception) {

    const block_labeling<N> &bl = se.get_labeling();
    const dimensions<N> &bidims = bl.get_block_index_dims();

    if (bidims.get_size() != expected.size())
        throw;

    abs_index<N> ai(bidims);
    do {

        if (se.is_allowed(ai.get_index()) != expected[ai.get_abs_index()]) {
            std::ostringstream oss;
            oss << (expected[ai.get_abs_index()] ? "!" : "")
                << sename << ".is_allowed(" << ai.get_index() << ")";
            fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
        }

    } while (ai.inc());

}

} // namespace libtensor
