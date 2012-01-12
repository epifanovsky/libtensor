#include <libtensor/symmetry/label/transfer_rule.h>
#include "transfer_rule_test.h"

namespace libtensor {

void transfer_rule_test::perform() throw(libtest::test_exception) {

    std::string table_id = setup_pg_table();

    try {

        test_basic_1(table_id);
        test_basic_2(table_id);
        test_basic_3(table_id);
        test_basic_4(table_id);
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

    typedef evaluation_rule::label_group label_group;

    evaluation_rule from, to;

    { // Setup from
        label_group i1(1, 0);
        std::vector<size_t> o1(3, 0);
        o1[0] = 0; o1[1] = 1; o1[2] = evaluation_rule::k_intrinsic;
        from.add_product(from.add_rule(i1, o1));
    }

    transfer_rule(from, 2, table_id).perform(to);

    evaluation_rule::rule_iterator it = to.begin();
    const evaluation_rule::basic_rule &br = to.get_rule(it);
    if (it == to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Non-empty result expected.");

    it++;
    if (it != to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Only one basic rule expected.");

    if (br.intr.size() != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# br.intr");

    if (br.intr[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.intr[0] != 0");

    if (br.order.size() != 3)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# br.order");

    if (br.order[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[0] != 0");

    if (br.order[1] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[1] != 1");

    if (br.order[2] != evaluation_rule::k_intrinsic)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[2]");

}

/** \test One basic rule, simplification of intrinsic labels
 **/
void transfer_rule_test::test_basic_2(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "transfer_rule_test::test_basic_2(" << table_id << ")";
    std::string tns(tnss.str());

    typedef evaluation_rule::label_group label_group;

    evaluation_rule from, to;

    { // Setup from
        label_group i1(3, 2); i1[1] = 1;
        std::vector<size_t> o1(3, 0);
        o1[0] = 0; o1[1] = 1; o1[2] = evaluation_rule::k_intrinsic;
        from.add_product(from.add_rule(i1, o1));
    }

    transfer_rule(from, 2, table_id).perform(to);

    evaluation_rule::rule_iterator it = to.begin();
    const evaluation_rule::basic_rule &br = to.get_rule(it);
    if (it == to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Non-empty result expected.");

    it++;
    if (it != to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Only one basic rule expected.");

    if (br.intr.size() != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# br.intr");

    if (br.intr[0] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.intr[0] != 1");

    if (br.intr[1] != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.intr[1] != 2");

    if (br.order.size() != 3)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# br.order");

    if (br.order[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[0] != 0");

    if (br.order[1] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[1] != 1");

    if (br.order[2] != evaluation_rule::k_intrinsic)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[2]");

}

/** \test One basic rule, trivial rule (all allowed)
 **/
void transfer_rule_test::test_basic_3(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "transfer_rule_test::test_basic_3(" << table_id << ")";
    std::string tns(tnss.str());

    typedef evaluation_rule::label_group label_group;

    evaluation_rule from, to;

    { // Setup from
        evaluation_rule::label_group i1(4, 0);
        for (evaluation_rule::label_t i = 0; i < 4; i++) i1[i] = i;
        std::vector<size_t> o1(3, 0);
        o1[0] = 0; o1[1] = 1;
        o1[2] = evaluation_rule::k_intrinsic;
        from.add_product(from.add_rule(i1, o1));
    }

    transfer_rule(from, 2, table_id).perform(to);

    evaluation_rule::rule_iterator it = to.begin();
    if (it == to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Non-empty result expected.");

    const evaluation_rule::basic_rule &br = to.get_rule(it);
    it++;
    if (it != to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Only one basic rule expected.");

    if (br.intr.size() != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# br.intr");

    if (br.intr[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.intr[0] != 0");

    if (br.order.size() != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# br.order");

    if (br.order[0] != evaluation_rule::k_intrinsic)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[0]");
}

/** \test One basic rule, trivial rule (all forbidden)
 **/
void transfer_rule_test::test_basic_4(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "transfer_rule_test::test_basic_4(" << table_id << ")";
    std::string tns(tnss.str());

    typedef evaluation_rule::label_group label_group;

    evaluation_rule from, to;

    { // Setup from
        label_group i1(1, 1);
        std::vector<size_t> o1(1, evaluation_rule::k_intrinsic);
        from.add_product(from.add_rule(i1, o1));
    }

    transfer_rule(from, 2, table_id).perform(to);

    evaluation_rule::rule_iterator it = to.begin();
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

    typedef evaluation_rule::label_group label_group;

    evaluation_rule from, to;

    { // Setup from
        evaluation_rule::label_group i1(1, 0), i2(2, 1);
        i2[1] = 0;
        std::vector<size_t> o1(3), o2(3);
        o1[0] = 0; o1[1] = 1; o1[2] = evaluation_rule::k_intrinsic;
        o2[0] = 0; o2[1] = 1; o2[2] = evaluation_rule::k_intrinsic;

        evaluation_rule::rule_id id1, id2;
        id1 = from.add_rule(i1, o1);
        id2 = from.add_rule(i2, o2);

        from.add_product(id1);
        from.add_to_product(0, id2);
    }

    transfer_rule(from, 2, table_id).perform(to);

    evaluation_rule::rule_iterator it = to.begin();
    if (it == to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Non-empty result expected.");

    const evaluation_rule::basic_rule &br = to.get_rule(it);
    it++;
    if (it != to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Only one basic rule expected.");

    if (br.intr.size() != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# br.intr");

    if (br.intr[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.intr[0] != 0");

    if (br.order.size() != 3)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# br.order");

    if (br.order[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[0] != 0");

    if (br.order[1] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[1] != 1");

    if (br.order[2] != evaluation_rule::k_intrinsic)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[2]");

}

/** \test Multiple rules, merge of two rules possible (in different products)
 **/
void transfer_rule_test::test_merge_2(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "transfer_rule_test::test_merge_2(" << table_id << ")";
    std::string tns(tnss.str());

    typedef evaluation_rule::label_group label_group;

    evaluation_rule from, to;

    { // Setup from
        evaluation_rule::label_group i1(1, 0), i2(2, 1);
        i2[1] = 0;
        std::vector<size_t> o1(3), o2(3);
        o1[0] = 0; o1[1] = 1; o1[2] = evaluation_rule::k_intrinsic;
        o2[0] = 0; o2[1] = 1; o2[2] = evaluation_rule::k_intrinsic;

        evaluation_rule::rule_id id1, id2;
        id1 = from.add_rule(i1, o1);
        id2 = from.add_rule(i2, o2);

        from.add_product(id1);
        from.add_product(id2);
    }

    transfer_rule(from, 2, table_id).perform(to);

    evaluation_rule::rule_iterator it = to.begin();
    if (it == to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Non-empty result expected.");

    const evaluation_rule::basic_rule &br = to.get_rule(it);

    it++;
    if (it != to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Only one basic rule expected.");

    if (br.intr.size() != 2)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# br.intr");

    if (br.intr[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.intr[0] != 0");

    if (br.intr[1] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.intr[1] != 1");

    if (br.order.size() != 3)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# br.order");

    if (br.order[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[0] != 0");

    if (br.order[1] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[1] != 1");

    if (br.order[2] != evaluation_rule::k_intrinsic)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[2]");

}

/** \test Multiple rules, merge of twice the same rule possible
 **/
void transfer_rule_test::test_merge_3(
        const std::string &table_id) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "transfer_rule_test::test_merge_3(" << table_id << ")";
    std::string tns(tnss.str());

    typedef evaluation_rule::label_group label_group;

    evaluation_rule from, to;

    { // Setup from
        evaluation_rule::label_group i1(1, 0);
        std::vector<size_t> o1(3);
        o1[0] = 0; o1[1] = 1; o1[2] = evaluation_rule::k_intrinsic;

        evaluation_rule::rule_id id1, id2;
        id1 = from.add_rule(i1, o1);
        id2 = from.add_rule(i1, o1);

        from.add_product(id1);
        from.add_product(id2);
    }

    transfer_rule(from, 2, table_id).perform(to);

    evaluation_rule::rule_iterator it = to.begin();
    if (it == to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Non-empty result expected.");

    const evaluation_rule::basic_rule &br = to.get_rule(it);

    it++;
    if (it != to.end())
        fail_test(tns.c_str(), __FILE__, __LINE__,
                "Only one basic rule expected.");

    if (br.intr.size() != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# br.intr");

    if (br.intr[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.intr[0] != 0");

    if (br.order.size() != 3)
        fail_test(tns.c_str(), __FILE__, __LINE__, "# br.order");

    if (br.order[0] != 0)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[0] != 0");

    if (br.order[1] != 1)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[1] != 1");

    if (br.order[2] != evaluation_rule::k_intrinsic)
        fail_test(tns.c_str(), __FILE__, __LINE__, "br.order[2]");

}


} // namespace libtensor
