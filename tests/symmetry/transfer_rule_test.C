#include <libtensor/symmetry/label/transfer_rule.h>
#include "transfer_rule_test.h"

namespace libtensor {

void transfer_rule_test::perform() throw(libtest::test_exception) {

    std::string table_id = setup_pg_table();

    try {

        test_basic_1(table_id);
        test_basic_2(table_id);
        test_basic_3(table_id);
        test_merge_1(table_id);
        test_merge_2(table_id);
        test_merge_3(table_id);

    } catch (libtest::test_exception) {
        product_table_container::get_instance().erase(table_id);
        throw;
    }

    product_table_container::get_instance().erase(table_id);
}

/** \test One basic rule, no simplifications possible
 **/
void transfer_rule_test::test_basic_1(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "transfer_rule_test::test_basic_1(" << table_id << ")";
    std::string tns(tnss.str());

    evaluation_rule<2> from, to;

    product_table_i::label_set_t i1; i1.insert(0);
    basic_rule<2> br1(i1);
    br1[0] = br1[1] = 1;
    from.add_product(from.add_rule(br1));

    transfer_rule<2>(from, table_id).perform(to);

    evaluation_rule<2>::rule_iterator it = to.begin();
    if (it == to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Non-empty result expected.");

    const basic_rule<2> &br2 = to.get_rule(it);
    it++;
    if (it != to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Only one basic rule expected.");

    if (br1 != br2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br1 != br2");
}

/** \test One basic rule, trivial rule (all allowed)
 **/
void transfer_rule_test::test_basic_2(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "transfer_rule_test::test_basic_3(" << table_id << ")";
    std::string tns(tnss.str());


    evaluation_rule<2> from, to;

    product_table_i::label_set_t i1;
    for (product_table_i::label_t i = 0; i < 4; i++) i1.insert(i);
    basic_rule<2> br1(i1);
    br1[0] = br1[1] = 1;
    from.add_product(from.add_rule(br1));

    transfer_rule<2>(from, table_id).perform(to);

    evaluation_rule<2>::rule_iterator it = to.begin();
    if (it == to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Non-empty result expected.");

    const basic_rule<2> &br2 = to.get_rule(it);
    it++;
    if (it != to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Only one basic rule expected.");

    if (br1 != br2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br1 != br2");
}

/** \test One basic rule, trivial rule (all forbidden)
 **/
void transfer_rule_test::test_basic_3(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "transfer_rule_test::test_basic_4(" << table_id << ")";
    std::string tns(tnss.str());

    evaluation_rule<2> from, to;

    product_table_i::label_set_t i1; i1.insert(1);
    basic_rule<2> br1(i1);
    from.add_product(from.add_rule(br1));

    transfer_rule<2>(from, table_id).perform(to);

    evaluation_rule<2>::rule_iterator it = to.begin();
    if (it != to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Empty result expected.");
}

/** \test Multiple rules, merge of two rules possible (in same product)
 **/
void transfer_rule_test::test_merge_1(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "transfer_rule_test::test_merge_1(" << table_id << ")";
    std::string tns(tnss.str());

    evaluation_rule<2> from, to;

    product_table_i::label_set_t i1, i2;
    i1.insert(0); i2.insert(0); i2.insert(1);
    basic_rule<2> br1(i1), br2(i2);
    br1[0] = br1[1] = 1;
    br2[0] = br2[1] = 1;

    evaluation_rule<2>::rule_id_t id1, id2;
    id1 = from.add_rule(br1);
    id2 = from.add_rule(br2);

    from.add_product(id1);
    from.add_to_product(0, id2);

    transfer_rule<2>(from, table_id).perform(to);

    evaluation_rule<2>::rule_iterator it = to.begin();
    if (it == to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Non-empty result expected.");

    const basic_rule<2> &br3 = to.get_rule(it);
    it++;
    if (it != to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Only one basic rule expected.");

    if (br1 != br3)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br1 != br3.");
}

/** \test Multiple rules, merge of two rules possible (in different products)
 **/
void transfer_rule_test::test_merge_2(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "transfer_rule_test::test_merge_2(" << table_id << ")";
    std::string tns(tnss.str());

    evaluation_rule<2> from, to;

    product_table_i::label_set_t i1, i2;
    i1.insert(0); i2.insert(0); i2.insert(1);

    basic_rule<2> br1(i1), br2(i2);
    br1[0] = br1[1] = 1;
    br2[0] = br2[1] = 1;

    evaluation_rule<2>::rule_id_t id1, id2;
    id1 = from.add_rule(br1);
    id2 = from.add_rule(br2);

    from.add_product(id1);
    from.add_product(id2);

    transfer_rule<2>(from, table_id).perform(to);

    evaluation_rule<2>::rule_iterator it = to.begin();
    if (it == to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Non-empty result expected.");

    const basic_rule<2> &br3 = to.get_rule(it);

    it++;
    if (it != to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Only one basic rule expected.");

    if (br2 != br3)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br2 != br3");
}

/** \test Multiple rules, merge of twice the same rule possible
 **/
void transfer_rule_test::test_merge_3(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "transfer_rule_test::test_merge_3(" << table_id << ")";
    std::string tns(tnss.str());

    evaluation_rule<2> from, to;
    product_table_i::label_set_t i1; i1.insert(0);

    basic_rule<2> br1(i1);
    br1[0] = br1[1] = 1;

    evaluation_rule<2>::rule_id_t id1, id2;
    id1 = from.add_rule(br1);
    id2 = from.add_rule(br1);

    from.add_product(id1);
    from.add_product(id2);

    transfer_rule<2>(from, table_id).perform(to);

    evaluation_rule<2>::rule_iterator it = to.begin();
    if (it == to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Non-empty result expected.");

    const basic_rule<2> &br2 = to.get_rule(it);

    it++;
    if (it != to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Only one basic rule expected.");

    if (br1 != br2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br1 != br2");
}


} // namespace libtensor
