#include <libtensor/symmetry/inst/er_merge.h>
#include <libtensor/symmetry/inst/er_optimize.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/print_symmetry.h>
#include "se_label_test_base.h"
#include "../test_utils.h"

using namespace libtensor;


/** \brief Merge 4 dimensions in 2 steps (merge dims can be simplified)
 **/
int test_1(const std::string &id) {

    static const char testname[] = "er_merge_test::test_1()";

    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

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

    evaluation_rule<2> r2, tmp;
    er_merge<4, 2>(r1, mmap, smsk).perform(tmp);
    er_optimize<2>(tmp, id).perform(r2);

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        return fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1) {
        return fail_test(testname, __FILE__, __LINE__, "seq.");
    }

    // Check product list
    evaluation_rule<2>::iterator it = r2.begin();
    if (it == r2.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        return fail_test(testname, __FILE__, __LINE__, "More than one products.");
    }

    product_rule<2>::iterator ip1 = pr1.begin();
    if (ip1 == pr1.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (pr1.get_intrinsic(ip1) != product_table_i::k_invalid)
        return fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");
    ip1++;
    if (ip1 != pr1.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Too many terms in pr1.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


/** \brief Merge 4 dimensions in 2 steps (merge dims can be simplified)
 **/
int test_2(const std::string &id) {

    static const char testname[] = "er_merge_test::test_2()";

    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    try {

    evaluation_rule<4> r1;
    {
        sequence<4, size_t> seq1(0), seq2(0);
        seq1[0] = seq1[1] = 1; seq2[2] = seq2[3] = 1;
        product_rule<4> &pr1 = r1.new_product();
        pr1.add(seq1, 2);
        pr1.add(seq2, 1);
    }

    sequence<4, size_t> mmap(0);
    mmap[0] = 0; mmap[1] = 1; mmap[2] = 0; mmap[3] = 1;
    mask<2> smsk;
    smsk[0] = smsk[1] = true;

    evaluation_rule<2> r2, tmp;
    er_merge<4, 2>(r1, mmap, smsk).perform(tmp);
    er_optimize<2>(tmp, id).perform(r2);

    // Check sequence list
    const eval_sequence_list<2> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        return fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    if (sl[0][0] != 1 || sl[0][1] != 1) {
        return fail_test(testname, __FILE__, __LINE__, "seq.");
    }

    // Check product list
    evaluation_rule<2>::iterator it = r2.begin();
    if (it == r2.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<2> &pr1 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        return fail_test(testname, __FILE__, __LINE__, "More than one products.");
    }

    product_rule<2>::iterator ip1 = pr1.begin(), ip2 = pr1.begin();
    if (ip1 == pr1.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    ip2++;
    if (ip2 == pr1.end()) {
        return fail_test(testname, __FILE__, __LINE__, "# terms in pr1.");
    }
    if ((pr1.get_intrinsic(ip1) != 1 || pr1.get_intrinsic(ip2) != 2) &&
            (pr1.get_intrinsic(ip1) != 2 || pr1.get_intrinsic(ip2) != 1)) {
        return fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");
    }
    ip2++;
    if (ip2 != pr1.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Too many terms in pr1.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


/** \brief Merge 4 dimensions in 2 steps (1 merge cannot be simplified)
 **/
int test_3(const std::string &id) {

    static const char testname[] = "er_merge_test::test_3()";

    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

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

    evaluation_rule<3> r2, tmp;
    er_merge<5, 3>(r1, mmap, smsk).perform(tmp);
    er_optimize<3>(tmp, id).perform(r2);

    // Check sequence list
    const eval_sequence_list<3> &sl = r2.get_sequences();
    if (sl.size() != 3) {
        return fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    if (sl[0][0] != 0 || sl[0][1] != 1 || sl[0][2] != 0) {
        return fail_test(testname, __FILE__, __LINE__, "seq[0].");
    }
    if (sl[1][0] != 0 || sl[1][1] != 0 || sl[1][2] != 2) {
        return fail_test(testname, __FILE__, __LINE__, "seq[1].");
    }
    if (sl[2][0] != 1 || sl[2][1] != 0 || sl[2][2] != 1) {
        return fail_test(testname, __FILE__, __LINE__, "seq[2].");
    }

    // Check product list
    evaluation_rule<3>::iterator it = r2.begin();
    if (it == r2.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<3> &pr1 = r2.get_product(it);
    it++;
    if (it == r2.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Only one product.");
    }
    const product_rule<3> &pr2 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        return fail_test(testname, __FILE__, __LINE__, "More than one products.");
    }

    product_rule<3>::iterator ip = pr1.begin();
    if (ip == pr1.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (pr1.get_intrinsic(ip) != 1)
        return fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");
    ip++;
    if (ip != pr1.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Too many terms in pr1.");
    }

    ip = pr2.begin();
    if (ip == pr2.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Empty product pr2.");
    }
    if (pr2.get_intrinsic(ip) != 2)
        return fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");
    ip++;
    if (pr2.get_intrinsic(ip) != 3)
        return fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");
    ip++;
    if (ip != pr2.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Too many terms in pr2.");
    }


    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_4(const std::string &id) {

    static const char testname[] = "er_merge_test::test_4()";

    typedef product_table_i::label_set_t label_set_t;
    typedef product_table_i::label_group_t label_group_t;

    try {

    evaluation_rule<4> r1;
    {
        sequence<4, size_t> seq1(1), seq2(0), seq3(0);
        seq2[0] = seq2[2] = 1; seq3[1] = seq3[3] = 1;
        product_rule<4> &pr1 = r1.new_product();
        pr1.add(seq1, 0);
        product_rule<4> &pr2 = r1.new_product();
        pr2.add(seq2, 0);
        pr2.add(seq3, 0);
    }

    sequence<4, size_t> mmap(0);
    mmap[0] = 0; mmap[1] = 1; mmap[2] = 2; mmap[3] = 1;
    mask<3> smsk;
    smsk[1] = true;

    evaluation_rule<3> r2, tmp;
    er_merge<4, 3>(r1, mmap, smsk).perform(tmp);
    er_optimize<3>(tmp, id).perform(r2);

    // Check sequence list
    const eval_sequence_list<3> &sl = r2.get_sequences();
    if (sl.size() != 1) {
        return fail_test(testname, __FILE__, __LINE__, "# seq.");
    }
    if (sl[0][0] != 1 || sl[0][2] != 1) {
        return fail_test(testname, __FILE__, __LINE__, "seq.");
    }

    // Check product list
    evaluation_rule<3>::iterator it = r2.begin();
    if (it == r2.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Empty product list.");
    }
    const product_rule<3> &pr1 = r2.get_product(it);
    it++;
    if (it != r2.end()) {
        return fail_test(testname, __FILE__, __LINE__, "More than one product.");
    }

    product_rule<3>::iterator ip1 = pr1.begin();
    if (ip1 == pr1.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Empty product pr1.");
    }
    if (pr1.get_intrinsic(ip1) != product_table_i::k_identity)
        return fail_test(testname, __FILE__, __LINE__, "Intrinsic label.");
    ip1++;
    if (ip1 != pr1.end()) {
        return fail_test(testname, __FILE__, __LINE__, "Too many terms in pr1.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    std::string s6 = "S6", c2v = "C2v";
    setup_pg_table(c2v);

    int rc =
        test_1(c2v) ||
        test_4(c2v) ||
        0;

    clear_pg_table(c2v);

    setup_pg_table(s6);

    rc =
        rc ||
        test_2(s6) ||
        test_3(s6) ||
        0;

    clear_pg_table(s6);

    return rc;
}

