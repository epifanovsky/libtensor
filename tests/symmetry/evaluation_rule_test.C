#include <libtensor/symmetry/evaluation_rule.h>
#include <libtensor/symmetry/point_group_table.h>
#include "evaluation_rule_test.h"

namespace libtensor {


void evaluation_rule_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_opt_1();
    test_opt_2();
    test_opt_3();
    test_reduce_1();
    test_reduce_2();
    test_reduce_3();
    test_reduce_4();
    test_reduce_5();
    test_reduce_6();
    test_merge_1();
    test_merge_2();
}


/** \test Create product(s), traverse them, clear them
 **/
void evaluation_rule_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_1()";

    typedef product_table_i::label_set_t label_set_t;

    try {

    evaluation_rule<3> rule;
    product_rule<3> &pr1 = rule.new_product();
    product_rule<3> &pr2 = rule.new_product();
    sequence<3, size_t> seq1(1), seq2(1);
    seq2[0] = 0;

    pr1.add(seq1, 0);
    pr2.add(seq1, 1);
    pr1.add(seq2, 1);

    // Check sequences in rule
    const eval_sequence_list<3> &sl = rule.get_sequences();
    if (sl.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    for (size_t i = 0; i < 3; i++) {
        if (seq1[i] != sl[0][i]) {
            fail_test(testname, __FILE__, __LINE__, "sl[0]");
        }
        if (seq2[i] != sl[1][i]) {
            fail_test(testname, __FILE__, __LINE__, "sl[1]");
        }
    }

    // Check products
    evaluation_rule<3>::const_iterator it = rule.begin();
    if (rule.get_product(it) != pr1) {
        fail_test(testname, __FILE__, __LINE__, "1st product rule.");
    }
    it++;
    if (rule.get_product(it) != pr2) {
        fail_test(testname, __FILE__, __LINE__, "2nd product rule.");
    }

    rule.clear();
    if (rule.begin() != rule.end()) {
        fail_test(testname, __FILE__, __LINE__, "Product rules not cleared.");
    }
    if (sl.size() != 0) {
        fail_test(testname, __FILE__, __LINE__,
                "Evaluation sequences not cleared.");
    }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Create products and test is_allowed() function
 **/
void evaluation_rule_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_2()";

    typedef product_table_i::label_set_t label_set_t;

    try {

     // S\f$_6\f$ point group - irreps: Ag, Eg, Au, Eu
     // Product table:
     //      Ag   Eg      Au   Eu
     // Ag   Ag   Eg      Au   Eu
     // Eg   Eg   2Ag+Eg  Eu   2Au+Eu
     // Au   Au   Eu      Ag   Eg
     // Eu   Eu   2Au+Eu  Eg   2Ag+Eg
     point_group_table::label_t ag = 0, eg = 1, au = 2, eu = 3;
     std::vector<std::string> im(4);
     im[ag] = "Ag"; im[eg] = "Eg"; im[au] = "Au"; im[eu] = "Eu";
     point_group_table s6("s6", im, "Ag");
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

    evaluation_rule<3> rule;
    product_rule<3> &pr1 = rule.new_product();
    product_rule<3> &pr2 = rule.new_product();
    sequence<3, size_t> seq1(1);
    sequence<3, size_t> seq2(1);
    seq2[0] = 0;

    pr1.add(seq1, ag);
    pr2.add(seq1, eg);
    pr2.add(seq2, au);

    sequence<3, product_table_i::label_t> blk(ag);
    // Block allowed by both product rules
    blk[0] = eu; blk[1] = eg; blk[2] = eu;
    if (! rule.is_allowed(blk, s6)) {
        fail_test(testname, __FILE__, __LINE__, "is_allowed (1)");
    }
    // Block allowed by 1st product rule only
    blk[0] = eg; blk[1] = eu; blk[2] = au;
    if (! rule.is_allowed(blk, s6)) {
        fail_test(testname, __FILE__, __LINE__, "is_allowed (2)");
    }
    // Block allowed by 2nd product rule only
    blk[0] = eu; blk[1] = au; blk[2] = ag;
    if (! rule.is_allowed(blk, s6)) {
        fail_test(testname, __FILE__, __LINE__, "is_allowed (3)");
    }
    // Block allowed by non of the product rules
    blk[0] = eg; blk[1] = ag; blk[2] = au;
    if (rule.is_allowed(blk, s6)) {
        fail_test(testname, __FILE__, __LINE__, "is_allowed (4)");
    }
    // Invalid block label
    blk[0] = product_table_i::k_invalid; blk[1] = ag; blk[2] = au;
    if (! rule.is_allowed(blk, s6)) {
        fail_test(testname, __FILE__, __LINE__, "is_allowed (5)");
    }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Optimization: remove all allowed terms from product rules
 **/
void evaluation_rule_test::test_opt_1() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_opt_1()";

    typedef product_table_i::label_set_t label_set_t;

    try {

    evaluation_rule<2> rule;
    {
        product_rule<2> &pr = rule.new_product();
        sequence<2, size_t> seq1(1), seq2(1);
        seq2[0] = 0;
        pr.add(seq1, product_table_i::k_identity);
        pr.add(seq2, product_table_i::k_invalid);
    }

    rule.optimize();

    // Check sequence list
    const eval_sequence_list<2> &sl = rule.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq not optimized.");
    }
    if (sl[0][0] != sl[0][1] || sl[0][0] != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong sequence.");
    }

    // Check product list
    evaluation_rule<2>::const_iterator it = rule.begin();
    if (it == rule.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &px = rule.get_product(it);
    it++;
    if (it != rule.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple products.");
    }

    product_rule<2>::iterator ip = px.begin();
    if (ip == px.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product.");
    }
    if (px.get_intrinsic(ip) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic label");
    }
    ip++;
    if (ip != px.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Optimization: Rule simplification due to a sole all allowed term
 **/
void evaluation_rule_test::test_opt_2() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_opt_2()";

    typedef product_table_i::label_set_t label_set_t;

    try {

    evaluation_rule<2> rule;
    {
        sequence<2, size_t> seq1(1), seq2(1);
        seq1[0] = 0; seq2[1] = 0;
        product_rule<2> &pr1 = rule.new_product();
        pr1.add(seq1, product_table_i::k_identity);
        product_rule<2> &pr2 = rule.new_product();
        pr2.add(seq2, product_table_i::k_invalid);
    }

    rule.optimize();

    // Check sequence list
    const eval_sequence_list<2> &sl = rule.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq not optimized.");
    }
    if (sl[0][0] != sl[0][1] || sl[0][0] != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong sequence.");
    }

    // Check product list
    evaluation_rule<2>::const_iterator it = rule.begin();
    if (it == rule.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &px = rule.get_product(it);
    it++;
    if (it != rule.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple products.");
    }

    product_rule<2>::iterator ip = px.begin();
    if (ip == px.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product.");
    }
    if (px.get_intrinsic(ip) != product_table_i::k_invalid) {
        fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic label");
    }
    ip++;
    if (ip != px.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms.");
    }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Optimization: remove products with all forbidden rules
 **/
void evaluation_rule_test::test_opt_3() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_opt_3()";

    typedef product_table_i::label_set_t label_set_t;

    try {

    evaluation_rule<2> rule;
    {
        sequence<2, size_t> seq1(1), seq2(1), seq3(0);
        seq1[0] = 0; seq2[1] = 0;
        product_rule<2> &pr1 = rule.new_product();
        pr1.add(seq1, product_table_i::k_identity);
        pr1.add(seq3, 1);
        product_rule<2> &pr2 = rule.new_product();
        pr2.add(seq2, 2);
    }

    rule.optimize();

    // Check sequence list
    const eval_sequence_list<2> &sl = rule.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq not optimized.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 0) {
        fail_test(testname, __FILE__, __LINE__, "Wrong sequence.");
    }

    // Check product list
    evaluation_rule<2>::const_iterator it = rule.begin();
    if (it == rule.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &px = rule.get_product(it);
    it++;
    if (it != rule.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple products.");
    }

    product_rule<2>::iterator ip = px.begin();
    if (ip == px.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product.");
    }
    if (px.get_intrinsic(ip) != 2) {
        fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic label");
    }
    ip++;
    if (ip != px.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \brief Reduce of two dimensions in one step (complete dim)
 **/
void evaluation_rule_test::test_reduce_1() throw(libtest::test_exception) {

    static const char *testname = "evaluation_rule_test::test_reduce_1()";

    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    try {

    // S\f$_6\f$ point group - irreps: Ag, Eg, Au, Eu
    // Product table:
    //      Ag   Eg      Au   Eu
    // Ag   Ag   Eg      Au   Eu
    // Eg   Eg   2Ag+Eg  Eu   2Au+Eu
    // Au   Au   Eu      Ag   Eg
    // Eu   Eu   2Au+Eu  Eg   2Ag+Eg
    point_group_table::label_t ag = 0, eg = 1, au = 2, eu = 3;
    std::vector<std::string> im(4);
    im[ag] = "Ag"; im[eg] = "Eg"; im[au] = "Au"; im[eu] = "Eu";
    point_group_table s6("s6", im, "Ag");
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

    evaluation_rule<4> r1;
    {
        sequence<4, size_t> seq1(1), seq2(1);
        seq1[2] = 0; seq1[3] = 0;
        seq2[0] = 0; seq2[1] = 0;
        product_rule<4> &pr1 = r1.new_product();
        pr1.add(seq1, eg);
        pr1.add(seq2, eu);
    }

    sequence<4, size_t> rmap(0);
    rmap[0] = 0; rmap[1] = 2; rmap[2] = 1; rmap[3] = 2;
    sequence<2, label_group_t> rdims;
    rdims[0].push_back(ag); rdims[0].push_back(au);
    rdims[0].push_back(eg); rdims[0].push_back(eu);

    evaluation_rule<2> r2;
    r1.reduce(r2, rmap, rdims, s6);

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq not optimized.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong sequence.");
    }

    // Check product list
    evaluation_rule<2>::const_iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Only one product.");
    }
    const product_rule<2> &pr2 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "More than two products.");
    }

    product_rule<2>::iterator ip1 = pr1.begin(), ip2 = pr2.begin();
    if (ip1 == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (ip2 == pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr2.");
    }
    if ((pr1.get_intrinsic(ip1) != au || pr2.get_intrinsic(ip2) != eu) &&
            (pr1.get_intrinsic(ip1) != eu || pr2.get_intrinsic(ip2) != au)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic labels.");
    }

    ip1++; ip2++;
    if (ip1 != pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms in pr1.");
    }
    if (ip2 != pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms in pr2.");
    }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \brief Reduction of four dimensions in two steps (complete dim)
 **/
void evaluation_rule_test::test_reduce_2() throw(libtest::test_exception) {


    static const char *testname = "evaluation_rule_test::test_reduce_2()";

    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    try {

    // S\f$_6\f$ point group - irreps: Ag, Eg, Au, Eu
    // Product table:
    //      Ag   Eg      Au   Eu
    // Ag   Ag   Eg      Au   Eu
    // Eg   Eg   2Ag+Eg  Eu   2Au+Eu
    // Au   Au   Eu      Ag   Eg
    // Eu   Eu   2Au+Eu  Eg   2Ag+Eg
    point_group_table::label_t ag = 0, eg = 1, au = 2, eu = 3;
    std::vector<std::string> im(4);
    im[ag] = "Ag"; im[eg] = "Eg"; im[au] = "Au"; im[eu] = "Eu";
    point_group_table s6("s6", im, "Ag");
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

    evaluation_rule<6> r1;
    {
        sequence<6, size_t> seq1(1), seq2(1);
        seq1[3] = 0; seq1[4] = 0; seq1[5] = 0;
        seq2[0] = 0; seq2[1] = 0; seq2[2] = 0;
        product_rule<6> &pr1 = r1.new_product();
        pr1.add(seq1, eg);
        pr1.add(seq2, eu);
    }

    sequence<6, size_t> rmap(0);
    rmap[0] = 0; rmap[1] = 2; rmap[2] = 3;
    rmap[3] = 1; rmap[4] = 3; rmap[5] = 2;
    sequence<4, label_group_t> rdims;
    rdims[0].push_back(ag); rdims[0].push_back(au);
    rdims[0].push_back(eg); rdims[0].push_back(eu);
    rdims[1].push_back(ag); rdims[1].push_back(au);
    rdims[1].push_back(eg); rdims[1].push_back(eu);

    evaluation_rule<2> r2;
    r1.reduce(r2, rmap, rdims, s6);

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq not optimized.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong sequence.");
    }

    // Check product list
    evaluation_rule<2>::const_iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Only one product.");
    }
    const product_rule<2> &pr2 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "More than two products.");
    }

    product_rule<2>::iterator ip1 = pr1.begin(), ip2 = pr2.begin();
    if (ip1 == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (ip2 == pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr2.");
    }
    if ((pr1.get_intrinsic(ip1) != au || pr2.get_intrinsic(ip2) != eu) &&
            (pr1.get_intrinsic(ip1) != eu || pr2.get_intrinsic(ip2) != au)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic labels.");
    }

    ip1++; ip2++;
    if (ip1 != pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms in pr1.");
    }
    if (ip2 != pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms in pr2.");
    }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \brief Reduction of six dimensions in three steps (complete dims)
 **/
void evaluation_rule_test::test_reduce_3() throw(libtest::test_exception) {


    static const char *testname = "evaluation_rule_test::test_reduce_3()";

    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    try {

    // S\f$_6\f$ point group - irreps: Ag, Eg, Au, Eu
    // Product table:
    //      Ag   Eg      Au   Eu
    // Ag   Ag   Eg      Au   Eu
    // Eg   Eg   2Ag+Eg  Eu   2Au+Eu
    // Au   Au   Eu      Ag   Eg
    // Eu   Eu   2Au+Eu  Eg   2Ag+Eg
    point_group_table::label_t ag = 0, eg = 1, au = 2, eu = 3;
    std::vector<std::string> im(4);
    im[ag] = "Ag"; im[eg] = "Eg"; im[au] = "Au"; im[eu] = "Eu";
    point_group_table s6("s6", im, "Ag");
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

    evaluation_rule<8> r1;
    {
        sequence<8, size_t> seq1(0), seq2(0), seq3(0), seq4(0);
        seq1[0] = 1; seq1[1] = 1; seq2[2] = 1; seq2[3] = 1;
        seq3[4] = 1; seq3[5] = 1; seq4[6] = 1; seq4[7] = 1;
        product_rule<8> &pr1 = r1.new_product();
        pr1.add(seq1, ag);
        pr1.add(seq2, eg);
        pr1.add(seq3, au);
        pr1.add(seq4, ag);
    }

    sequence<8, size_t> rmap(0);
    rmap[0] = 0; rmap[1] = 2; rmap[2] = 4; rmap[3] = 2;
    rmap[4] = 4; rmap[5] = 3; rmap[6] = 3; rmap[7] = 1;
    sequence<6, label_group_t> rdims;
    rdims[0].push_back(ag); rdims[0].push_back(au);
    rdims[0].push_back(eg); rdims[0].push_back(eu);
    rdims[1].push_back(ag); rdims[1].push_back(au);
    rdims[1].push_back(eg); rdims[1].push_back(eu);
    rdims[2].push_back(ag); rdims[2].push_back(au);
    rdims[2].push_back(eg); rdims[2].push_back(eu);

    evaluation_rule<2> r2;
    r1.reduce(r2, rmap, rdims, s6);

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq not optimized.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong sequence.");
    }

    // Check product list
    evaluation_rule<2>::const_iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple products.");
    }
    product_rule<2>::iterator ip1 = pr1.begin();
    if (ip1 == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (pr1.get_intrinsic(ip1) != eu) {
        fail_test(testname, __FILE__, __LINE__, "Wrong intrinsic label.");
    }
    ip1++;
    if (ip1 != pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Multiple terms in pr1.");
    }


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \brief Reduction of four dimensions in two steps (one complete dims)
 **/
void evaluation_rule_test::test_reduce_4() throw(libtest::test_exception) {


    static const char *testname = "evaluation_rule_test::test_reduce_4()";

    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    try {

    // S\f$_6\f$ point group - irreps: Ag, Eg, Au, Eu
    // Product table:
    //      Ag   Eg      Au   Eu
    // Ag   Ag   Eg      Au   Eu
    // Eg   Eg   2Ag+Eg  Eu   2Au+Eu
    // Au   Au   Eu      Ag   Eg
    // Eu   Eu   2Au+Eu  Eg   2Ag+Eg
    point_group_table::label_t ag = 0, eg = 1, au = 2, eu = 3;
    std::vector<std::string> im(4);
    im[ag] = "Ag"; im[eg] = "Eg"; im[au] = "Au"; im[eu] = "Eu";
    point_group_table s6("s6", im, "Ag");
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

    evaluation_rule<6> r1;
    {
        sequence<6, size_t> seq1(0), seq2(0), seq3(0);
        seq1[0] = 1; seq1[1] = 1;
        seq2[2] = 1; seq2[3] = 1;
        seq3[4] = 1; seq3[5] = 1;
        product_rule<6> &pr1 = r1.new_product();
        pr1.add(seq1, ag);
        pr1.add(seq2, eg);
        pr1.add(seq3, au);
    }

    sequence<6, size_t> rmap(0);
    rmap[0] = 0; rmap[1] = 2; rmap[2] = 3;
    rmap[3] = 2; rmap[4] = 3; rmap[5] = 1;

    sequence<4, label_group_t> rdims;
    rdims[0].push_back(ag); rdims[0].push_back(au);
    rdims[0].push_back(eg); rdims[0].push_back(eu);
    rdims[1].push_back(ag); rdims[1].push_back(au);

    evaluation_rule<2> r2;
    r1.reduce(r2, rmap, rdims, s6);

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    size_t isa = 0;
    if (sl[0][0] == 1 && sl[0][1] == 0) {
        if (sl[1][0] != 0 || sl[1][1] != 1)
            fail_test(testname, __FILE__, __LINE__, "2nd seq.");
    }
    else if (sl[0][0] == 0 && sl[0][1] == 1) {
        if (sl[1][0] != 1 || sl[1][1] != 0)
            fail_test(testname, __FILE__, __LINE__, "2nd seq.");

        isa = 1;
    }
    else {
        fail_test(testname, __FILE__, __LINE__, "1st seq.");
    }

    // Check product list
    evaluation_rule<2>::const_iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Only one product.");
    }
    const product_rule<2> &pr2 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "More than two products.");
    }
    product_rule<2>::iterator ip1a = pr1.begin(), ip1b = pr1.begin();
    product_rule<2>::iterator ip2a = pr2.begin(), ip2b = pr2.begin();
    ip1b++; ip2b++;
    if (ip1a == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (ip1b == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Only one term in pr1.");
    }
    if (ip2a == pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr2.");
    }
    if (ip2b == pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Only one term in pr2.");
    }

    label_t l1a(eg), l1b(au), l2a(eu), l2b(ag);
    if (ip1a->first != isa) std::swap(l1a, l1b);
    if (ip2a->first != isa) std::swap(l2a, l2b);

    if (pr1.get_intrinsic(ip1a) != l1a)
        fail_test(testname, __FILE__, __LINE__, "Intrinsic l1a.");
    if (pr1.get_intrinsic(ip1b) != l1b)
        fail_test(testname, __FILE__, __LINE__, "Intrinsic l1b.");
    if (pr1.get_intrinsic(ip2a) != l2a)
        fail_test(testname, __FILE__, __LINE__, "Intrinsic l2a.");
    if (pr1.get_intrinsic(ip2b) != l2b)
        fail_test(testname, __FILE__, __LINE__, "Intrinsic l2b.");

    ip1b++;
    if (ip1b != pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Too many terms in pr1.");
    }
    ip2b++;
    if (ip2b != pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Too many terms in pr2.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \brief Reduction of two dimensions in one steps (complete dims)
 **/
void evaluation_rule_test::test_reduce_5() throw(libtest::test_exception) {


    static const char *testname = "evaluation_rule_test::test_reduce_5()";

    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    try {

    // S\f$_6\f$ point group - irreps: Ag, Eg, Au, Eu
    // Product table:
    //      Ag   Eg      Au   Eu
    // Ag   Ag   Eg      Au   Eu
    // Eg   Eg   2Ag+Eg  Eu   2Au+Eu
    // Au   Au   Eu      Ag   Eg
    // Eu   Eu   2Au+Eu  Eg   2Ag+Eg
    point_group_table::label_t ag = 0, eg = 1, au = 2, eu = 3;
    std::vector<std::string> im(4);
    im[ag] = "Ag"; im[eg] = "Eg"; im[au] = "Au"; im[eu] = "Eu";
    point_group_table s6("s6", im, "Ag");
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

    evaluation_rule<4> r1;
    {
        sequence<4, size_t> seq1(1);
        product_rule<4> &pr1 = r1.new_product();
        pr1.add(seq1, au);
    }

    sequence<4, size_t> rmap(0);
    rmap[0] = 0; rmap[1] = 1; rmap[2] = 2; rmap[3] = 2;
    sequence<2, label_group_t> rdims;
    rdims[0].push_back(ag); rdims[0].push_back(au);
    rdims[0].push_back(eg); rdims[0].push_back(eu);

    evaluation_rule<2> r2;
    r1.reduce(r2, rmap, rdims, s6);

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1) {
        fail_test(testname, __FILE__, __LINE__, "seq.");
    }

    // Check product list
    evaluation_rule<2>::const_iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Only one product.");
    }
    const product_rule<2> &pr2 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "More than two products.");
    }
    product_rule<2>::iterator ip1 = pr1.begin();
    product_rule<2>::iterator ip2 = pr2.begin();
    if (ip1 == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (ip2 == pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr2.");
    }

    if ((pr1.get_intrinsic(ip1) != au || pr2.get_intrinsic(ip2) != eu) &&
            (pr1.get_intrinsic(ip1) != eu || pr2.get_intrinsic(ip2) != au))
        fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");

    ip1++;
    if (ip1 != pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Too many terms in pr1.");
    }
    ip2++;
    if (ip2 != pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Too many terms in pr2.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \brief Reduction of four dimensions in two steps (non-complete dims)
 **/
void evaluation_rule_test::test_reduce_6() throw(libtest::test_exception) {


    static const char *testname = "evaluation_rule_test::test_reduce_6()";

    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    try {

    // C\f$_{2v}\f$ point group - irreps: A1, A2, B1, B2
    // Product table:
    //      A1   A2   B1   B2
    // A1   A1   A2   B1   B2
    // A2   A2   A1   B2   B1
    // B1   B1   B2   A1   A2
    // B2   B2   B1   A2   A1
    point_group_table::label_t a1 = 0, a2 = 1, b1 = 2, b2 = 3;
    std::vector<std::string> im(4);
    im[a1] = "A1"; im[a2] = "A2"; im[b1] = "B1"; im[b2] = "B2";
    point_group_table c2v("c2v", im, "A1");
    c2v.add_product(a2, a2, a1);
    c2v.add_product(a2, b1, b2);
    c2v.add_product(a2, b2, b1);
    c2v.add_product(b1, b1, a1);
    c2v.add_product(b1, b2, a2);
    c2v.add_product(b2, b2, a1);
    c2v.check();

    evaluation_rule<6> r1;
    {
        sequence<6, size_t> seq1(0), seq2(0), seq3(1), seq4(1), seq5(0);
        seq1[2] = 1; seq2[4] = 1; seq3[2] = seq3[4] = 0;
        seq4[2] = seq4[3] = seq4[4] = 0; seq5[3] = 1;
        product_rule<6> &pr1a = r1.new_product();
        pr1a.add(seq1, a1);
        pr1a.add(seq2, a1);
        pr1a.add(seq3, a1);
        product_rule<6> &pr1b = r1.new_product();
        pr1b.add(seq1, a2);
        pr1b.add(seq2, a2);
        pr1b.add(seq3, a1);
        product_rule<6> &pr1c = r1.new_product();
        pr1c.add(seq1, b1);
        pr1c.add(seq2, b1);
        pr1c.add(seq3, a1);
        product_rule<6> &pr1d = r1.new_product();
        pr1d.add(seq1, b2);
        pr1d.add(seq2, b2);
        pr1d.add(seq3, a1);
        product_rule<6> &pr2a = r1.new_product();
        pr2a.add(seq1, a1);
        pr2a.add(seq2, a1);
        pr2a.add(seq4, a1);
        pr2a.add(seq5, a1);
        product_rule<6> &pr2b = r1.new_product();
        pr2b.add(seq1, a2);
        pr2b.add(seq2, a2);
        pr2b.add(seq4, a1);
        pr2b.add(seq5, a1);
        product_rule<6> &pr2c = r1.new_product();
        pr2c.add(seq1, b1);
        pr2c.add(seq2, b1);
        pr2c.add(seq4, a1);
        pr2c.add(seq5, a1);
        product_rule<6> &pr2d = r1.new_product();
        pr2d.add(seq1, b2);
        pr2d.add(seq2, b2);
        pr2d.add(seq4, a1);
        pr2d.add(seq5, a1);
    }

    sequence<6, size_t> rmap(0);
    rmap[0] = 0; rmap[1] = 1; rmap[2] = 2; rmap[3] = 2; rmap[4] = rmap[5] = 3;
    sequence<4, label_group_t> rdims;
    rdims[0].push_back(a1);
    rdims[0].push_back(b1);
    rdims[0].push_back(b2);
    rdims[1].push_back(a1);
    rdims[1].push_back(a2);
    rdims[1].push_back(b1);
    rdims[1].push_back(b2);

    evaluation_rule<2> r2;
    r1.reduce(r2, rmap, rdims, c2v);

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1) {
        fail_test(testname, __FILE__, __LINE__, "seq.");
    }

    // Check product list
    evaluation_rule<2>::const_iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }

    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "More than one product.");
    }
    product_rule<2>::iterator ip1 = pr1.begin();
    if (ip1 == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (pr1.get_intrinsic(ip1) != a1) {
        fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \brief Merge 4 dimensions in 2 steps (merge dims can be simplified)
 **/
void evaluation_rule_test::test_merge_1() throw(libtest::test_exception) {


    static const char *testname = "evaluation_rule_test::test_merge_1()";

    typedef product_table_i::label_set_t label_set_t;

    try {

    evaluation_rule<4> r1;
    {
        sequence<4, size_t> seq1(1), seq2(0), seq3(0);
        seq2[0] = seq2[2] = 1; seq3[1] = seq3[3] = 1;
        product_rule<4> &pr1 = r1.new_product();
        pr1.add(seq1, 0);
        product_rule<4> &pr2 = r1.new_product();
        pr2.add(seq2, 1);
        pr2.add(seq3, 3);
    }

    sequence<4, size_t> mmap(0);
    mmap[0] = 0; mmap[1] = 0; mmap[2] = 1; mmap[3] = 1;
    mask<2> smsk;
    smsk[0] = smsk[1] = true;

    evaluation_rule<2> r2;
    r1.merge(r2, mmap, smsk);

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1) {
        fail_test(testname, __FILE__, __LINE__, "seq.");
    }

    // Check product list
    evaluation_rule<2>::const_iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "More than one products.");
    }

    product_rule<2>::iterator ip1 = pr1.begin();
    if (ip1 == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (pr1.get_intrinsic(ip1) != product_table_i::k_invalid)
        fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");
    ip1++;
    if (ip1 != pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Too many terms in pr1.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \brief Merge 4 dimensions in 2 steps (1 merge cannot be simplified)
 **/
void evaluation_rule_test::test_merge_2() throw(libtest::test_exception) {


    static const char *testname = "evaluation_rule_test::test_merge_2()";

    typedef product_table_i::label_set_t label_set_t;

    try {

    evaluation_rule<5> r1;
    {
        sequence<5, size_t> seq1(1), seq2(0), seq3(0);
        seq1[3] = seq1[4] = 0;
        seq2[3] = seq2[4] = 1;
        seq3[0] = seq3[4] = 1;
        product_rule<5> &pr1 = r1.new_product();
        pr1.add(seq1, 1);
        product_rule<5> &pr2 = r1.new_product();
        pr2.add(seq2, 2);
        pr2.add(seq3, 3);
    }

    sequence<5, size_t> mmap(0);
    mmap[0] = 0; mmap[1] = 0; mmap[2] = 1; mmap[3] = 2; mmap[4] = 2;
    mask<3> smsk;
    smsk[0] = true;

    evaluation_rule<3> r2;
    r1.merge(r2, mmap, smsk);

    // Check sequence list
    const eval_sequence_list<3> &sl = r2.get_sequences();
    if (sl.size() != 3) {
        fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    if (sl[0][0] != 0 || sl[0][1] != 1 || sl[0][2] != 0) {
        fail_test(testname, __FILE__, __LINE__, "seq[0].");
    }
    if (sl[1][0] != 0 || sl[1][1] != 0 || sl[1][2] != 2) {
        fail_test(testname, __FILE__, __LINE__, "seq[1].");
    }
    if (sl[2][0] != 1 || sl[2][1] != 0 || sl[2][2] != 1) {
        fail_test(testname, __FILE__, __LINE__, "seq[2].");
    }

    // Check product list
    evaluation_rule<3>::const_iterator it = r2.begin();
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<3> &pr1 = r2.get_product(it);
    it++;
    if (it == r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Only one product.");
    }
    const product_rule<3> &pr2 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        fail_test(testname, __FILE__, __LINE__, "More than one products.");
    }

    product_rule<3>::iterator ip = pr1.begin();
    if (ip == pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (pr1.get_intrinsic(ip) != 1)
        fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");
    ip++;
    if (ip != pr1.end()) {
        fail_test(testname, __FILE__, __LINE__, "Too many terms in pr1.");
    }

    ip = pr2.begin();
    if (ip == pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Empty product pr2.");
    }
    if (pr2.get_intrinsic(ip) != 2)
        fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");
    ip++;
    if (pr2.get_intrinsic(ip) != 3)
        fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");
    ip++;
    if (ip != pr2.end()) {
        fail_test(testname, __FILE__, __LINE__, "Too many terms in pr2.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
