#include <sstream>
#include <libtensor/core/abs_index.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include "se_label_test.h"

namespace libtensor {

void se_label_test::perform() throw(libtest::test_exception) {

    std::string s6 = setup_pg_table();
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
    el.set_rule(1);
    const evaluation_rule<3> &r1 = el.get_rule();
    if (r1.get_n_sequences() != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# sequences (r1).");
    if (r1[0][0] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "r1[0][0]");
    if (r1[0][1] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "r1[0][1]");
    if (r1[0][2] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "r1[0][2]");

    if (r1.get_n_products() != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# products (r1).");
    evaluation_rule<3>::iterator it = r1.begin(0);
    if (r1.get_intrinsic(it) != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "Intrisic label (r1).");
    if (r1.get_target(it) != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "Target label (r1).");
    it++;
    if (it != r1.end(0))
        fail_test(tns.c_str(), __FILE__, __LINE__, "# terms (r1).");

    // Simple rule with multiple intrinsic labels
    product_table_i::label_set_t lg2;
    lg2.insert(0); lg2.insert(2);
    el.set_rule(lg2);
    const evaluation_rule<3> &r2 = el.get_rule();

    if (r2.get_n_sequences() != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# sequences (r2).");
    if (r2[0][0] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "r2[0][0]");
    if (r2[0][1] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "r2[0][1]");
    if (r2[0][2] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "r2[0][2]");

    if (r2.get_n_products() != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# products (r2).");
    it = r2.begin(0);
    if (r2.get_intrinsic(it) != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "Intrisic label (r2,0).");
    if (r2.get_target(it) != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "Target label (r2,0).");
    it++;
    if (it != r2.end(0))
        fail_test(tns.c_str(), __FILE__, __LINE__, "# terms (r2,0).");
    it = r2.begin(1);
    if (r2.get_intrinsic(it) != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "Intrisic label (r2,1).");
    if (r2.get_target(it) != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "Target label (r2,1).");
    it++;
    if (it != r2.end(1))
        fail_test(tns.c_str(), __FILE__, __LINE__, "# terms (r2,1).");

    evaluation_rule<3> r3ref;
    sequence<3, size_t> seq3a(1), seq3b(0);
    seq3a[2] = 0; seq3b[2] = 1;
    size_t sno3a = r3ref.add_sequence(seq3a);
    size_t sno3b = r3ref.add_sequence(seq3b);

    size_t pno3a = r3ref.add_product(sno3a, 0, 0);
    size_t pno3b = r3ref.add_product(sno3a, 2, 0);
    r3ref.add_to_product(pno3a, sno3b, 1, 0);
    r3ref.add_to_product(pno3b, sno3b, 1, 0);

    el.set_rule(r3ref);
    const evaluation_rule<3> &r3 = el.get_rule();
    if (r3.get_n_sequences() != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# sequences (r3).");
    if (r3.get_n_products() != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# products (r3).");
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

    evaluation_rule<2> rule;
    sequence<2, size_t> seq(0);
    seq[0] = 1;
    size_t seqno = rule.add_sequence(seq);
    rule.add_product(seqno, 0, 0);
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

    evaluation_rule<2> r1, r2;
    sequence<2, size_t> seqa(0), seqb(0);
    seqa[0] = seqb[1] = 1;
    size_t sno1a = r1.add_sequence(seqa);
    size_t sno1b = r1.add_sequence(seqb);
    size_t sno2a = r2.add_sequence(seqa);
    size_t sno2b = r2.add_sequence(seqb);
    size_t pno1 = r1.add_product(sno1a, 0, 0);
    r1.add_to_product(pno1, sno1b, 1, 0);
    r2.add_product(sno1a, 0, 0);
    r2.add_product(sno1b, 1, 0);

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

    evaluation_rule<2> r1, r2;
    sequence<2, size_t> seqa(0), seqb(0);
    seqa[0] = seqb[1] = 1;
    size_t sno1a = r1.add_sequence(seqa);
    size_t sno1b = r1.add_sequence(seqb);
    size_t sno2a = r2.add_sequence(seqa);
    size_t sno2b = r2.add_sequence(seqb);
    size_t pno1 = r1.add_product(sno1a, 0, 0);
    r1.add_to_product(pno1, sno1b, 1, 0);
    r2.add_product(sno1a, 0, 0);
    r2.add_product(sno1b, 1, 0);

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


} // namespace libtensor
