#include <libtensor/symmetry/combine_label.h>
#include <libtensor/symmetry/product_table_container.h>
#include "combine_label_test.h"

namespace libtensor {

void combine_label_test::perform() throw(libtest::test_exception) {

    std::string s6 = setup_pg_table();
    try {

         test_1(s6);
         test_2(s6);

    } catch (libtest::test_exception) {
        product_table_container::get_instance().erase(s6);
        throw;
    }

    product_table_container::get_instance().erase(s6);
}

/** \test Tests setting evaluation rules
 **/
void combine_label_test::test_1(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "combine_label_test::test_1(" << table_id << ")";
    std::string tns = tnss.str();

    index<2> i1, i2;
    i2[0] = 3; i2[1] = 3;
    dimensions<2> bidims(index_range<2>(i1, i2));

    se_label<2, double> el1(bidims, table_id), el2(bidims, table_id);
    el1.set_rule(1);

    evaluation_rule<2> r2;
    product_rule<2> &pr2 = r2.new_product();
    sequence<2, size_t> seq2a(0), seq2b(0);
    seq2a[0] = 1; seq2b[1] = 1;
    pr2.add(seq2a, 2);
    pr2.add(seq2b, 3);
    el2.set_rule(r2);

    combine_label<2, double> cl(el1);
    if (cl.get_table_id() != table_id)
        fail_test(tns.c_str(), __FILE__, __LINE__, "Table ID.");

    cl.add(el2);

    const evaluation_rule<2> &rule = cl.get_rule();
    const eval_sequence_list<2> &sl = rule.get_sequences();
    if (sl.size() != 3)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# seq.");

    evaluation_rule<2>::const_iterator it = rule.begin();
    if (it == rule.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "Empty rule");
    const product_rule<2> &pr = rule.get_product(it);
    it++;
    if (it != rule.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "# products.");

    product_rule<2>::iterator ip = pr.begin();
    if (ip == pr.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "# terms.");
    ip++;
    if (ip == pr.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "# terms.");
    ip++;
    if (ip == pr.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "# terms.");
    ip++;
    if (ip != pr.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "# terms.");
}

/** \test Four blocks, all labeled, different index types, basic rules only
 **/
void combine_label_test::test_2(
        const std::string &table_id) throw(libtest::test_exception) {
    
    std::ostringstream tnss;
    tnss << "combine_label_test::test_2(" << table_id << ")";
    std::string tns = tnss.str();

    index<4> i1, i2;
    i2[0] = 3; i2[1] = 3; i2[2] = 3; i2[3] = 3;
    dimensions<4> bidims(index_range<4>(i1, i2));

    se_label<4, double> el1(bidims, table_id), el2(bidims, table_id);
    evaluation_rule<4> r1;
    product_rule<4> &pr1a = r1.new_product();
    product_rule<4> &pr1b = r1.new_product();
    sequence<4, size_t> seq1a(0), seq1b(0), seq1c(0), seq1d(0);
    seq1a[0] = seq1a[1] = 1; seq1b[2] = seq1b[3] = 1;
    seq1c[0] = seq1c[2] = 1; seq1d[1] = seq1d[3] = 1;
    pr1a.add(seq1a, 0); pr1a.add(seq1b, 1);
    pr1b.add(seq1c, 2); pr1b.add(seq1d, 3);
    el1.set_rule(r1);

    evaluation_rule<4> r2;
    product_rule<4> &pr2a = r2.new_product();
    product_rule<4> &pr2b = r2.new_product();
    sequence<4, size_t> seq2a(0), seq2b(0), seq2c(0), seq2d(0);
    seq2a[0] = 1; seq2b[1] = 1; seq2c[2] = 1; seq2d[3] = 1;
    pr2a.add(seq2a, 0); pr2a.add(seq2b, 1);
    pr2b.add(seq2c, 2); pr2b.add(seq2d, 3);
    el2.set_rule(r2);

    combine_label<4, double> cl(el1);
    if (cl.get_table_id() != table_id)
        fail_test(tns.c_str(), __FILE__, __LINE__, "Table ID.");

    cl.add(el2);

    const evaluation_rule<4> &rule = cl.get_rule();
    const eval_sequence_list<4> &sl = rule.get_sequences();
    if (sl.size() != 8)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# seq.");

    evaluation_rule<4>::const_iterator it = rule.begin();
    if (it == rule.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "Empty rule.");
    it++;
    if (it == rule.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "# products.");
    it++;
    if (it == rule.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "# products.");
    it++;
    if (it == rule.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "# products.");
    it++;
    if (it != rule.end())
        fail_test(tns.c_str(), __FILE__, __LINE__, "# products.");

}


} // namespace libtensor
